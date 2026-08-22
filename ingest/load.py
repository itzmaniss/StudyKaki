"""Load a PDF — ARCHITECTURE.md §3 `load`.

`bytes -> Document + native text layer (+ page images on demand)`.

**If the PDF has a text layer, extract it here and set `has_text_layer=True` so OCR is
skipped entirely.** §3 calls that the biggest single speedup available: a 200-page born-digital
textbook goes from minutes of PaddleOCR to under a second of pymupdf.

Text-layer blocks carry `ocr_confidence=None` — the schema uses that to mean "not OCR'd".

Reading order is a global monotonic counter across the whole document, assigned top-to-bottom
then left-to-right within each page. Multi-column layout detection is not attempted; that is
layout analysis, and it belongs to the OCR stage if it is ever needed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import structlog

from core.cache import CacheKey, StageCache, stage_timer
from core.schema import BBox, Block, Document

log = structlog.get_logger(__name__)

STAGE_VERSION = "load/1"
PIPELINE_VERSION = "v1"

PDF_MAGIC = b"%PDF"
PDF_MIME = "application/pdf"

# A born-digital page carries far more than this; a scan with a stray page number carries far
# less. Anything in between is ambiguous and gets OCR'd, which is the safe direction to err.
TEXT_LAYER_MIN_CHARS_PER_PAGE = 50

# Blocks are ordered top-to-bottom; y0 is quantised so that two blocks whose tops differ by
# less than a line's leading are treated as the same row and ordered left-to-right. bboxes are
# already normalised at this point, so the band is expressed as a fraction of page height.
_ROW_BAND_PT = 5.0

_HEADING_MAX_CHARS = 120
_HEADING_SIZE_RATIO = 1.15
_BOLD_HEADING_SIZE_RATIO = 1.02
_BOLD_CHAR_RATIO = 0.6
_CAPTION_SIZE_RATIO = 0.92
_CAPTION_MIN_CHARS = 12
_CAPTION_MAX_CHARS = 400
_BOLD_FLAG = 1 << 4

_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?|table|chart|plate|exhibit)\.?\s*\d", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:[-•–⁃▪○·*]|\d{1,2}[.)]|[a-z][.)])\s+")


class DocumentLoadError(ValueError):
    """Base for anything that makes a file unusable as a source document."""


class UnsupportedMimeError(DocumentLoadError):
    pass


class CorruptDocumentError(DocumentLoadError):
    pass


@dataclass(frozen=True)
class PageImage:
    page: int
    dpi: int
    pixels: np.ndarray

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


@dataclass(frozen=True)
class LoadResult:
    document: Document
    blocks: list[Block]

    def __post_init__(self) -> None:
        if self.blocks and not self.document.has_text_layer:
            raise ValueError("blocks came back from load but has_text_layer is False")


def doc_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open(data: bytes) -> pymupdf.Document:
    if not data:
        raise CorruptDocumentError("empty file")
    if not data.lstrip()[:4].startswith(PDF_MAGIC):
        raise UnsupportedMimeError(
            f"only {PDF_MIME} is ingested; got a file whose header is {data[:8]!r}"
        )
    try:
        return pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise CorruptDocumentError(f"pymupdf could not open the document: {exc}") from exc


def _normalise_bbox(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> BBox:
    width = page_rect.width or 1.0
    height = page_rect.height or 1.0
    xs = sorted(((rect.x0 - page_rect.x0) / width, (rect.x1 - page_rect.x0) / width))
    ys = sorted(((rect.y0 - page_rect.y0) / height, (rect.y1 - page_rect.y0) / height))
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in (xs[0], ys[0], xs[1], ys[1]))
    return (x0, y0, x1, y1)


@dataclass(frozen=True)
class _Candidate:
    page: int
    rect: BBox
    text: str
    size: float
    bold_ratio: float


def _block_text(blk: dict[str, Any]) -> str:
    lines = [
        "".join(span.get("text", "") for span in line.get("spans", ()))
        for line in blk.get("lines", ())
    ]
    return "\n".join(line for line in lines if line.strip())


def _block_typography(blk: dict[str, Any], weights: dict[float, int]) -> tuple[float, float]:
    """Max span size in the block and its bold char fraction; also feeds the doc-wide histogram."""
    size = 0.0
    bold_chars = 0
    total = 0
    for line in blk.get("lines", ()):
        for span in line.get("spans", ()):
            n = len(span.get("text", ""))
            if n == 0:
                continue
            span_size = float(span.get("size", 0.0))
            size = max(size, span_size)
            total += n
            if span_size > 0:
                key = round(span_size, 1)
                weights[key] = weights.get(key, 0) + n
            flags = int(span.get("flags", 0))
            if flags & _BOLD_FLAG or "bold" in str(span.get("font", "")).lower():
                bold_chars += n
    return size, (bold_chars / total if total else 0.0)


def _classify(text: str, size: float, bold_ratio: float, body_size: float, in_table: bool) -> str:
    if in_table:
        return "table"
    ratio = size / body_size if body_size else 1.0
    small_and_short = (
        ratio <= _CAPTION_SIZE_RATIO and _CAPTION_MIN_CHARS <= len(text) <= _CAPTION_MAX_CHARS
    )
    if len(text) <= _CAPTION_MAX_CHARS and (_CAPTION_RE.match(text) or small_and_short):
        return "caption"
    heading_size = ratio >= _HEADING_SIZE_RATIO or (
        bold_ratio >= _BOLD_CHAR_RATIO and ratio >= _BOLD_HEADING_SIZE_RATIO
    )
    if heading_size and len(text) <= _HEADING_MAX_CHARS:
        return "heading"
    if _LIST_RE.match(text):
        return "list"
    return "paragraph"


def _table_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    try:
        return [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    except (AttributeError, ValueError, RuntimeError) as exc:
        log.warning("load.table_detection_failed", page=page.number + 1, error=str(exc))
        return []


def _in_table(rect: pymupdf.Rect, tables: Sequence[pymupdf.Rect]) -> bool:
    area = rect.get_area()
    if area <= 0:
        return False
    return any((table & rect).get_area() > 0.5 * area for table in tables)


def _row_band(page_rect: pymupdf.Rect) -> float:
    return _ROW_BAND_PT / (page_rect.height or 1.0)


def _extract_blocks(doc: pymupdf.Document, doc_id: str) -> list[Block]:
    """Two passes: collect typography doc-wide, then classify against the body font size.

    Only lightweight candidates are retained between passes — the raw pymupdf dicts for a
    200-page book are tens of megabytes and none of it is needed after the first pass.
    """
    weights: dict[float, int] = {}
    candidates: list[tuple[_Candidate, bool]] = []
    for page in doc:
        page_rect = page.rect
        tables = _table_rects(page)
        page_candidates: list[tuple[_Candidate, bool]] = []
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0 or not blk.get("lines"):
                continue
            text = _block_text(blk)
            if not text.strip():
                continue
            size, bold_ratio = _block_typography(blk, weights)
            rect = pymupdf.Rect(blk["bbox"])
            page_candidates.append(
                (
                    _Candidate(
                        page=page.number + 1,
                        rect=_normalise_bbox(rect, page_rect),
                        text=text,
                        size=size,
                        bold_ratio=bold_ratio,
                    ),
                    _in_table(rect, tables),
                )
            )
        page_candidates.sort(
            key=lambda c: (round(c[0].rect[1] / _row_band(page_rect)), c[0].rect[0])
        )
        candidates.extend(page_candidates)

    body_size = max(weights.items(), key=lambda kv: (kv[1], -kv[0]))[0] if weights else 0.0
    return [
        Block(
            block_id=f"{doc_id[:12]}:{cand.page}:{order:05d}",
            doc_id=doc_id,
            page=cand.page,
            bbox=cand.rect,
            kind=_classify(cand.text, cand.size, cand.bold_ratio, body_size, in_table),
            reading_order=order,
            script="unknown",
            text=cand.text,
            ocr_confidence=None,
        )
        for order, (cand, in_table) in enumerate(candidates)
    ]


def _has_text_layer(blocks: Sequence[Block], n_pages: int) -> bool:
    if n_pages == 0:
        return False
    chars = sum(len(b.text.strip()) for b in blocks)
    return chars / n_pages >= TEXT_LAYER_MIN_CHARS_PER_PAGE


def load_pdf(
    data: bytes,
    filename: str,
    *,
    cache: StageCache | None = None,
    config_hash: str = "none",
) -> LoadResult:
    doc_id = doc_id_for(data)
    with stage_timer("load", doc_id) as span:
        keys = {
            name: CacheKey(
                stage=f"load.{name}",
                input_hash=doc_id,
                stage_version=STAGE_VERSION,
                config_hash=config_hash,
            )
            for name in ("document", "blocks")
        }
        if cache is not None:
            documents = cache.load(keys["document"], Document)
            blocks = cache.load(keys["blocks"], Block)
            if documents and blocks is not None:
                span.cached = True
                span.n_out = len(blocks)
                span.extra["has_text_layer"] = documents[0].has_text_layer
                return LoadResult(document=documents[0], blocks=blocks)

        with _open(data) as doc:
            n_pages = doc.page_count
            blocks = _extract_blocks(doc, doc_id)
        has_text_layer = _has_text_layer(blocks, n_pages)
        if not has_text_layer:
            blocks = []
        document = Document(
            doc_id=doc_id,
            filename=filename,
            mime=PDF_MIME,
            n_pages=n_pages,
            has_text_layer=has_text_layer,
            pipeline_version=PIPELINE_VERSION,
        )
        if cache is not None:
            cache.store(keys["document"], [document], Document)
            cache.store(keys["blocks"], blocks, Block)
        span.n_out = len(blocks)
        span.extra["has_text_layer"] = has_text_layer
        span.extra["n_pages"] = n_pages
    return LoadResult(document=document, blocks=blocks)


def load_path(path: str | Path, **kw: Any) -> LoadResult:
    p = Path(path)
    if not p.is_file():
        raise DocumentLoadError(f"not a file: {p}")
    return load_pdf(p.read_bytes(), p.name, **kw)


def render_pages(
    data: bytes, *, dpi: int = 200, pages: Sequence[int] | None = None
) -> Iterator[PageImage]:
    """Page rasters for the OCR stage. Yielded lazily — a 200-page render will not fit in RAM.

    Not cached: images are large, and the OCR stage caches its `Block` output instead.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    with _open(data) as doc:
        wanted = range(1, doc.page_count + 1) if pages is None else pages
        for page_no in wanted:
            if not 1 <= page_no <= doc.page_count:
                raise ValueError(f"page {page_no} out of range 1..{doc.page_count}")
            pix = doc[page_no - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            yield PageImage(page=page_no, dpi=dpi, pixels=pixels)
