"""OCR — ARCHITECTURE.md §3 `ocr`: page images -> `list[Block]` with bbox.

PaddleOCR **mobile** detection + recognition through OpenVINO (§3, §7.6 — never the server
models, they will not hit interactive latency on the target i5). One shared detector feeds
per-script recognition heads.

Provenance is the whole job (§0.2). `page` is 1-indexed and `bbox` is normalised to 0-1
against the page image: `core.schema.Block` rejects raw pixel coordinates, and a citation
drawn from an un-normalised bbox highlights the wrong part of the page.

Everything model-shaped sits behind the `TextDetector` / `TextRecognizer` protocols, and
`ingest/ocr_paddle.py` holds the PaddleOCR-through-OpenVINO implementations. The tests drive
this same pipeline with stubs, so bbox arithmetic, reading order, thresholds and caching are
verifiable without an IR on disk.

This stage only runs on scanned pages. `ingest/load.py` sets `has_text_layer=True` when
pymupdf found a native text layer, and §3 skips OCR entirely for those.

Deliberately not attempted: table structure (that is layout analysis; a line-level OCR box
cannot see it, so `kind` is never `table`), skew correction, and handwriting (§11).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog

from core.cache import CacheKey, StageCache, hash_bytes, stage_timer
from core.config import Config
from core.schema import SCRIPTS, BBox, Block

# Backwards, module-order-wise: `normalize` runs after `ocr`. But script detection is a pure
# Unicode-range lookup and re-implementing those ranges here to avoid the import would be a
# second copy of the table that decides how every chunk is tokenised.
from ingest.load import PageImage
from ingest.normalize import detect_script

log = structlog.get_logger(__name__)

STAGE_VERSION = "ocr/1"

#: The recogniser key used when no dedicated per-script head is registered. PP-OCRv5 mobile
#: rec is itself multilingual, so this is the normal case, not a degraded one.
DEFAULT_SCRIPT = "unknown"

_HEADING_HEIGHT_RATIO = 1.25
_HEADING_MAX_CHARS = 120
_CAPTION_MAX_CHARS = 400
_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?|table|chart|plate|exhibit)\.?\s*\d", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:[-•–⁃▪○·*]|\d{1,2}[.)]|[a-z][.)])\s+")


class OcrError(RuntimeError):
    """Base for every OCR failure. A page that cannot be read must say why."""


class OcrModelError(OcrError):
    """The det/rec model is missing, mis-shaped, or missing its character dictionary."""


@dataclass(frozen=True)
class OcrParams:
    """Every tunable in this stage, in one bag, so the cache key can hash all of it.

    These are not in `configs/base.yaml`: §6 fixes that file's shape and `core/config.py`
    forbids extra keys, so an `ocr:` block there is a change to another agent's contract.
    Pass a different `OcrParams` to tune; the stage cache keys on `digest()` and re-OCRs.

    `min_confidence` is deliberately permissive. CLAUDE.md records the measurement from a
    prior run: Tamil recognition returns 0.4-0.6 on clean scans where the output is visibly
    correct while Latin on the same page returns 0.9+, so any uniform threshold at or above
    0.4 drops most correct Tamil. 0.30 sits below the observed correct-Tamil floor and still
    discards near-zero-confidence noise. `confidence_by_script` is **empty by default** —
    per-script numbers are a calibration decision that needs labelled pages, not a guess.
    """

    # Detection defaults are PaddleOCR's own, read from the PP-OCRv5_mobile_det checkpoint
    # metadata (DetResizeForTest.resize_long, DBPostProcess.*) — upstream values, not guesses.
    det_limit_side_len: int = 960
    det_thresh: float = 0.3
    det_box_thresh: float = 0.6
    det_unclip_ratio: float = 1.5
    det_min_box_side: int = 3
    det_max_candidates: int = 1000
    rec_image_height: int = 48
    rec_min_width: int = 320
    rec_max_width: int = 3200
    rec_batch_size: int = 16
    min_confidence: float = 0.30
    confidence_by_script: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.det_limit_side_len < 32:
            raise ValueError(f"det_limit_side_len must be >= 32, got {self.det_limit_side_len}")
        if self.rec_batch_size < 1:
            raise ValueError(f"rec_batch_size must be >= 1, got {self.rec_batch_size}")
        if self.rec_min_width > self.rec_max_width:
            raise ValueError(
                f"rec_min_width {self.rec_min_width} exceeds rec_max_width {self.rec_max_width}"
            )
        for name in ("det_thresh", "det_box_thresh", "min_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be 0-1, got {value}")
        unknown = sorted(set(self.confidence_by_script) - SCRIPTS)
        if unknown:
            raise ValueError(f"confidence_by_script has unknown scripts: {unknown}")

    def threshold_for(self, script: str) -> float:
        return self.confidence_by_script.get(script, self.min_confidence)

    def digest(self) -> str:
        payload = {
            **{k: v for k, v in vars(self).items() if k != "confidence_by_script"},
            "confidence_by_script": dict(sorted(self.confidence_by_script.items())),
        }
        return hash_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class DetBox:
    """A detected text line in **page-image pixel** coordinates. Normalised on the way out."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class RecResult:
    text: str
    confidence: float


class TextDetector(Protocol):
    """`fingerprint` identifies the weights; it rides in the cache key so swapping the
    detector re-OCRs instead of silently serving boxes from the previous model."""

    fingerprint: str

    def detect(self, image: np.ndarray) -> list[DetBox]: ...


class TextRecognizer(Protocol):
    fingerprint: str
    script: str

    def recognize(self, crops: Sequence[np.ndarray]) -> list[RecResult]: ...


def _to_rgb(pixels: np.ndarray) -> np.ndarray:
    """Accept what a rasteriser plausibly hands us; refuse anything we would misread."""
    if not isinstance(pixels, np.ndarray):
        raise OcrError(f"page pixels must be a numpy array, got {type(pixels).__name__}")
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    if pixels.ndim != 3:
        raise OcrError(f"page pixels must be HxW or HxWxC, got shape {pixels.shape}")
    if pixels.shape[0] < 1 or pixels.shape[1] < 1:
        raise OcrError(f"page image is empty: shape {pixels.shape}")
    channels = pixels.shape[2]
    if channels == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif channels == 4:
        pixels = pixels[:, :, :3]
    elif channels != 3:
        raise OcrError(f"page image must have 1, 3 or 4 channels, got {channels}")
    if pixels.dtype != np.uint8:
        raise OcrError(f"page pixels must be uint8, got dtype {pixels.dtype}")
    return np.ascontiguousarray(pixels)


def _classify(text: str, height: float, median_height: float) -> str:
    ratio = height / median_height if median_height > 0 else 1.0
    if len(text) <= _CAPTION_MAX_CHARS and _CAPTION_RE.match(text):
        return "caption"
    if ratio >= _HEADING_HEIGHT_RATIO and len(text) <= _HEADING_MAX_CHARS:
        return "heading"
    if _LIST_RE.match(text):
        return "list"
    return "paragraph"


def _reading_order(boxes: Sequence[DetBox]) -> list[DetBox]:
    """Top-to-bottom, left-to-right within a row band. Column detection is not attempted —
    that is layout analysis, and guessing it wrong scrambles a whole page's reading order."""
    if not boxes:
        return []
    band = max(1.0, float(np.median([b.height for b in boxes])) * 0.6)
    return sorted(boxes, key=lambda b: (round(b.y0 / band), b.x0))


def _normalise_bbox(box: DetBox, height: int, width: int) -> BBox | None:
    x0 = min(max(box.x0 / width, 0.0), 1.0)
    x1 = min(max(box.x1 / width, 0.0), 1.0)
    y0 = min(max(box.y0 / height, 0.0), 1.0)
    y1 = min(max(box.y1 / height, 0.0), 1.0)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


class OcrEngine:
    """Shared detector, per-script recognition heads (§3).

    Head selection is a single pass plus a targeted retry: the default head reads every
    crop, and any line whose text is in a script that *has* a dedicated head is re-read by
    it. The dedicated head's output wins outright rather than winning on confidence —
    confidence is not comparable across heads, which is exactly the calibration trap
    CLAUDE.md documents for Tamil.
    """

    def __init__(
        self,
        detector: TextDetector,
        recognizers: Mapping[str, TextRecognizer],
        params: OcrParams | None = None,
    ) -> None:
        if not recognizers:
            raise OcrError("OcrEngine needs at least one recognition head")
        unknown = sorted(set(recognizers) - SCRIPTS)
        if unknown:
            raise OcrError(f"recognition heads keyed by unknown scripts: {unknown}")
        if DEFAULT_SCRIPT not in recognizers and len(recognizers) > 1:
            raise OcrError(
                f"more than one recognition head but no {DEFAULT_SCRIPT!r} default — "
                f"nothing would read a line whose script has no dedicated head"
            )
        self.detector = detector
        self.recognizers = dict(recognizers)
        self.params = params or OcrParams()
        self._primary = self.recognizers.get(DEFAULT_SCRIPT) or next(
            iter(self.recognizers.values())
        )

    @property
    def fingerprint(self) -> str:
        parts = [self.detector.fingerprint]
        parts += [f"{k}={self.recognizers[k].fingerprint}" for k in sorted(self.recognizers)]
        return hash_bytes("\x1f".join(parts).encode())

    def read_page(self, image: PageImage, doc_id: str) -> list[Block]:
        """One page -> page-local blocks. `reading_order` is renumbered by `ocr_pages`."""
        if image.page < 1:
            raise OcrError(f"page is 1-indexed, got {image.page}")
        pixels = _to_rgb(image.pixels)
        height, width = pixels.shape[:2]
        boxes = _reading_order(self.detector.detect(pixels))
        if not boxes:
            return []

        crops = [_crop(pixels, b) for b in boxes]
        texts, scripts = self._recognize(crops)
        median_height = float(np.median([b.height for b in boxes]))

        blocks: list[Block] = []
        for box, result, script in zip(boxes, texts, scripts, strict=True):
            text = result.text.strip()
            if not text or result.confidence < self.params.threshold_for(script):
                continue
            bbox = _normalise_bbox(box, height, width)
            if bbox is None:
                continue
            blocks.append(
                Block(
                    block_id=f"{doc_id[:12]}:{image.page}:{len(blocks):05d}",
                    doc_id=doc_id,
                    page=image.page,
                    bbox=bbox,
                    kind=_classify(text, box.height, median_height),
                    reading_order=len(blocks),
                    script=script,
                    text=text,
                    ocr_confidence=min(max(result.confidence, 0.0), 1.0),
                )
            )
        return blocks

    def _recognize(self, crops: Sequence[np.ndarray]) -> tuple[list[RecResult], list[str]]:
        results = list(self._primary.recognize(crops))
        scripts = [self._primary.script] * len(results)
        if len(self.recognizers) < 2:
            return results, scripts

        routed: dict[str, list[int]] = {}
        for i, result in enumerate(results):
            head = self.recognizers.get(detect_script(result.text))
            if head is not None and head is not self._primary:
                routed.setdefault(head.script, []).append(i)
        for script, index in routed.items():
            head = self.recognizers[script]
            for i, result in zip(index, head.recognize([crops[i] for i in index]), strict=True):
                results[i] = result
                scripts[i] = head.script
        if routed:
            log.info(
                "ocr.rerouted",
                heads={k: len(v) for k, v in routed.items()},
                n_lines=len(results),
            )
        return results, scripts


def _crop(pixels: np.ndarray, box: DetBox) -> np.ndarray:
    x0 = int(max(0, math.floor(box.x0)))
    y0 = int(max(0, math.floor(box.y0)))
    x1 = int(min(pixels.shape[1], math.ceil(box.x1)))
    y1 = int(min(pixels.shape[0], math.ceil(box.y1)))
    crop = pixels[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)]
    return crop if crop.size else np.zeros((1, 1, 3), dtype=np.uint8)


def _page_hash(doc_id: str, image: PageImage, engine: OcrEngine, params: OcrParams) -> str:
    """Content-addressed per page, so a re-render or a model swap re-OCRs and nothing else does."""
    header = "\x1f".join(
        (
            doc_id,
            str(image.page),
            str(image.dpi),
            str(image.pixels.shape),
            engine.fingerprint,
            params.digest(),
        )
    )
    return hash_bytes(header.encode() + np.ascontiguousarray(image.pixels).tobytes())


def ocr_page(
    image: PageImage,
    *,
    doc_id: str,
    engine: OcrEngine,
    cache: StageCache | None = None,
    config_hash: str = "none",
) -> list[Block]:
    """One page through det + rec, cached on the page's own pixels.

    Caching is per page, not per document: OCR is the slowest stage in the system, and a
    document-level entry means a run that dies on page 150 re-OCRs pages 1-149 (CLAUDE.md).
    """
    params = engine.params
    input_hash = _page_hash(doc_id, image, engine, params)
    with stage_timer("ocr", input_hash) as span:
        span.extra["page"] = image.page

        def compute() -> list[Block]:
            return engine.read_page(image, doc_id)

        if cache is None:
            blocks = compute()
            span.n_out = len(blocks)
            return blocks
        key = CacheKey(
            stage="ocr",
            input_hash=input_hash,
            stage_version=STAGE_VERSION,
            config_hash=f"{config_hash}:{params.digest()}",
        )
        return cache.get_or_compute(key, Block, compute, span=span)


def ocr_pages(
    images: Iterable[PageImage],
    *,
    doc_id: str,
    engine: OcrEngine,
    cache: StageCache | None = None,
    config_hash: str = "none",
) -> list[Block]:
    """Every page of one document, with `reading_order` renumbered document-wide.

    `block_id` stays page-local so a cached page stays valid whatever else is OCR'd
    alongside it; only `reading_order` is global, matching `ingest/load.py`.
    """
    with stage_timer("ocr.document", doc_id) as span:
        blocks: list[Block] = []
        pages = 0
        for image in images:
            pages += 1
            for block in ocr_page(
                image, doc_id=doc_id, engine=engine, cache=cache, config_hash=config_hash
            ):
                blocks.append(replace(block, reading_order=len(blocks)))
        span.n_out = len(blocks)
        span.extra["n_pages"] = pages
    return blocks


def build_engine(
    cfg: Config,
    *,
    params: OcrParams | None = None,
    manifest_path: str | Path | None = None,
    extra_heads: Mapping[str, TextRecognizer] | None = None,
) -> OcrEngine:
    """The production engine: `ocr_det` + `ocr_rec` from the manifest, device fallback included.

    Loading happens here and nowhere else in this module, so every other entry point stays
    testable without an IR on disk. Nothing here touches the network (§0.3) — a missing
    model raises with the setup command to run.
    """
    from ingest.ocr_paddle import PaddleTextDetector, PaddleTextRecognizer, load_charset
    from models.registry import ir_sha256, load_model

    params = params or OcrParams()
    det = load_model("ocr_det", cfg, manifest_path=manifest_path)
    rec = load_model("ocr_rec", cfg, manifest_path=manifest_path)
    log.info(
        "ocr.engine_built",
        det=det.name,
        det_device=det.device,
        det_fell_back=det.fell_back,
        rec=rec.name,
        rec_device=rec.device,
        rec_fell_back=rec.fell_back,
    )

    n_classes = _output_classes(rec.compiled)
    charset = load_charset(rec.entry.ir_dir, n_classes)
    heads: dict[str, TextRecognizer] = {
        DEFAULT_SCRIPT: PaddleTextRecognizer(
            compiled=rec.compiled,
            charset=charset,
            params=params,
            fingerprint=ir_sha256(rec.entry),
            script=DEFAULT_SCRIPT,
        )
    }
    heads.update(extra_heads or {})
    detector = PaddleTextDetector(
        compiled=det.compiled, params=params, fingerprint=ir_sha256(det.entry)
    )
    return OcrEngine(detector=detector, recognizers=heads, params=params)


def _output_classes(compiled: Any) -> int | None:
    try:
        shape = compiled.output(0).get_partial_shape()
        last = shape[len(shape) - 1]
    except (RuntimeError, IndexError, AttributeError, TypeError):
        return None
    return int(last.get_length()) if last.is_static else None
