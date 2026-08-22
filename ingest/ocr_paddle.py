"""PaddleOCR mobile det + rec through OpenVINO — the backend behind `ingest/ocr.py` (§3).

Split out of `ingest/ocr.py` so the stage stays model-agnostic (and both files stay under
CLAUDE.md's 500-line limit). `ingest/ocr.py` owns the protocols, the `Block` contract and
the cache; everything that knows what a DB probability map or a CTC lattice looks like
lives here.

**Mobile/tiny only** (§3, §7.6). Preprocessing and post-processing reproduce the values in
the checkpoint's own metadata (`inference.yml` / `config.json`: `DetResizeForTest`,
`NormalizeImage`, `DBPostProcess`), so the numbers here are PaddleOCR's, not invented.

No third-party image library is used. cv2, scipy and shapely are not declared dependencies
in §0.1, and the three things they would be used for — bilinear resize, connected
components, polygon offset — are each a few lines against a binarised map.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from core.schema import SCRIPTS
from ingest.ocr import DEFAULT_SCRIPT, DetBox, OcrModelError, OcrParams, RecResult

log = structlog.get_logger(__name__)

# ImageNet constants applied, as upstream does, to a **BGR** array — see `_preprocess`.
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Half-pixel-centre bilinear, matching `cv2.resize(..., INTER_LINEAR)`.

    The Paddle models were trained on cv2-resized inputs; a different sampling convention
    shifts every box by a fraction of a pixel and costs recognition accuracy at small type.
    """
    h, w = img.shape[:2]
    src = img.astype(np.float32, copy=False)
    if (h, w) == (out_h, out_w):
        return src.copy()
    ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = src[y0][:, x0] * (1.0 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1.0 - wx) + src[y1][:, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _components(
    mask: np.ndarray, prob: np.ndarray, max_candidates: int
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Connected components of a binary map as (bbox, mean probability).

    Run-length + union-find rather than `scipy.ndimage.label` or `cv2.findContours`: neither
    is a declared dependency in §0.1, and a binarised DB map is a few thousand runs, so the
    Python loop is over runs, never over pixels.
    """
    height, width = mask.shape
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = mask[y]
        edges = np.diff(row.astype(np.int8))
        starts = np.flatnonzero(edges == 1) + 1
        ends = np.flatnonzero(edges == -1) + 1
        if row[0]:
            starts = np.concatenate(([0], starts))
        if row[-1]:
            ends = np.concatenate((ends, [width]))
        current: list[tuple[int, int, int]] = []
        for x0, x1 in zip(starts, ends, strict=True):
            label = len(parent)
            parent.append(label)
            current.append((int(x0), int(x1), label))
            runs.append((y, int(x0), int(x1), label))

        i = j = 0
        while i < len(current) and j < len(previous):
            a0, a1, la = current[i]
            b0, b1, lb = previous[j]
            if a0 < b1 and b0 < a1:
                union(la, lb)
            if a1 <= b1:
                i += 1
            else:
                j += 1
        previous = current

    boxes: dict[int, list[float]] = {}
    for y, x0, x1, label in runs:
        root = find(label)
        weight = float(prob[y, x0:x1].sum())
        entry = boxes.get(root)
        if entry is None:
            boxes[root] = [x0, y, x1, y + 1, weight, x1 - x0]
        else:
            entry[0] = min(entry[0], x0)
            entry[1] = min(entry[1], y)
            entry[2] = max(entry[2], x1)
            entry[3] = max(entry[3], y + 1)
            entry[4] += weight
            entry[5] += x1 - x0

    out = [
        ((int(x0), int(y0), int(x1), int(y1)), total / count)
        for x0, y0, x1, y1, total, count in boxes.values()
        if count > 0
    ]
    out.sort(key=lambda item: -(item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
    return out[:max_candidates]


def _unclip(box: tuple[int, int, int, int], ratio: float) -> tuple[float, float, float, float]:
    """DB shrinks text regions at training time; the offset puts the box back around the glyphs.

    `area * ratio / perimeter` is the Vatti offset pyclipper computes, which for an
    axis-aligned rectangle reduces to this closed form.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    distance = (w * h * ratio) / max(2.0 * (w + h), 1e-6)
    return (x0 - distance, y0 - distance, x1 + distance, y1 + distance)


class PaddleTextDetector:
    """PP-OCRv5 mobile detection (DB) through an OpenVINO compiled model.

    Preprocessing reproduces the checkpoint's own `inference.yml`: **BGR** channel order
    (PaddleOCR decodes with cv2 and normalises the BGR array with RGB-ordered ImageNet
    constants — reproducing the quirk is required, "fixing" it silently degrades recall),
    long side resized to `det_limit_side_len`, each side rounded to a multiple of 32.
    """

    def __init__(self, compiled: Any, params: OcrParams, fingerprint: str) -> None:
        self.compiled = compiled
        self.params = params
        self.fingerprint = fingerprint

    def detect(self, image: np.ndarray) -> list[DetBox]:
        height, width = image.shape[:2]
        tensor, scale_y, scale_x = self._preprocess(image)
        try:
            outputs = self.compiled([tensor])
            prob = np.asarray(outputs[0])
        except (RuntimeError, KeyError, IndexError) as exc:
            raise OcrModelError(f"detection inference failed: {exc}") from exc
        if prob.ndim != 4 or prob.shape[1] != 1:
            raise OcrModelError(
                f"detector produced shape {prob.shape}, expected [N,1,H,W] probability map"
            )
        return self._postprocess(prob[0, 0].astype(np.float32), height, width, scale_y, scale_x)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        p = self.params
        height, width = image.shape[:2]
        percent = p.det_limit_side_len / max(height, width)
        out_h = max(int(round(height * percent / 32.0)) * 32, 32)
        out_w = max(int(round(width * percent / 32.0)) * 32, 32)
        resized = _resize_bilinear(image[:, :, ::-1], out_h, out_w) / 255.0
        normalised = (resized - _DET_MEAN) / _DET_STD
        tensor = np.ascontiguousarray(normalised.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor, out_h / height, out_w / width

    def _postprocess(
        self, prob: np.ndarray, height: int, width: int, scale_y: float, scale_x: float
    ) -> list[DetBox]:
        p = self.params
        mask = prob > p.det_thresh
        if not mask.any():
            return []
        out: list[DetBox] = []
        for box, score in _components(mask, prob, p.det_max_candidates):
            if score < p.det_box_thresh:
                continue
            x0, y0, x1, y1 = _unclip(box, p.det_unclip_ratio)
            x0, x1 = x0 / scale_x, x1 / scale_x
            y0, y1 = y0 / scale_y, y1 / scale_y
            if min(x1 - x0, y1 - y0) < p.det_min_box_side:
                continue
            out.append(
                DetBox(
                    x0=max(0.0, x0),
                    y0=max(0.0, y0),
                    x1=min(float(width), x1),
                    y1=min(float(height), y1),
                    score=float(score),
                )
            )
        return out


def load_charset(ir_dir: Path, n_classes: int | None = None) -> list[str]:
    """Character dictionary for CTC decoding, from what `models/convert.py` copies next to the IR.

    `convert_paddle` copies the checkpoint's `config.json` / `inference.yml`, both of which
    carry `PostProcess.character_dict`. JSON is preferred: the YAML form of the PP-OCRv5
    dictionary is 18k lines and parsing it on every model load is pure latency.

    Index 0 is the CTC blank. PaddleOCR appends a space class when `use_space_char` is on;
    `n_classes` (the model's real output width) decides whether this checkpoint has it,
    rather than a flag that can disagree with the weights.
    """
    chars: list[str] | None = None
    for name in ("config.json", "inference.yml"):
        path = ir_dir / name
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text()) if name.endswith(".json") else _read_yaml(path)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("ocr.charset_unreadable", path=str(path), error=str(exc))
            continue
        found = (raw.get("PostProcess") or {}).get("character_dict")
        if found:
            chars = [str(c) for c in found]
            break
    if chars is None:
        dict_files = sorted(ir_dir.glob("*dict*.txt"))
        if dict_files:
            chars = dict_files[0].read_text(encoding="utf-8").splitlines()
    if not chars:
        raise OcrModelError(
            f"no recognition character dictionary in {ir_dir} — looked for "
            f"PostProcess.character_dict in config.json / inference.yml and for *dict*.txt. "
            f"models/convert.py copies these next to the IR; re-run "
            f"`uv run python -m scripts.setup`"
        )

    charset = ["<blank>", *chars]
    if n_classes is not None:
        if n_classes == len(charset) + 1:
            charset.append(" ")
        elif n_classes != len(charset):
            raise OcrModelError(
                f"recogniser emits {n_classes} classes but the dictionary in {ir_dir} yields "
                f"{len(charset)} — the IR and its character dictionary are from different models"
            )
    return charset


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not parse to a mapping")
    return loaded


class PaddleTextRecognizer:
    """PP-OCRv5 mobile recognition (CTC) through an OpenVINO compiled model.

    Crops are batched (§7.5 applies here for the same reason it applies to embeddings — a
    per-crop call is ~10x the overhead for no benefit) and sorted by aspect ratio so a batch
    pads to the width its own widest crop needs, not to the widest crop on the page.
    """

    def __init__(
        self,
        compiled: Any,
        charset: Sequence[str],
        params: OcrParams,
        fingerprint: str,
        script: str = DEFAULT_SCRIPT,
    ) -> None:
        if script not in SCRIPTS:
            raise OcrModelError(f"recogniser script {script!r} is not one of {sorted(SCRIPTS)}")
        self.compiled = compiled
        self.charset = list(charset)
        self.params = params
        self.fingerprint = fingerprint
        self.script = script

    def recognize(self, crops: Sequence[np.ndarray]) -> list[RecResult]:
        out: list[RecResult] = [RecResult("", 0.0)] * len(crops)
        if not crops:
            return out
        order = sorted(range(len(crops)), key=lambda i: _aspect(crops[i]))
        for start in range(0, len(order), self.params.rec_batch_size):
            index = order[start : start + self.params.rec_batch_size]
            probs = self._infer(self._batch([crops[i] for i in index]))
            for position, result in zip(index, self._decode(probs), strict=True):
                out[position] = result
        return out

    def _batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        p = self.params
        height = p.rec_image_height
        widest = max(_aspect(c) for c in crops)
        width = int(np.clip(math.ceil(height * widest), p.rec_min_width, p.rec_max_width))
        batch = np.zeros((len(crops), 3, height, width), dtype=np.float32)
        for i, crop in enumerate(crops):
            target = min(width, max(1, math.ceil(height * _aspect(crop))))
            resized = _resize_bilinear(crop, height, target) / 255.0
            batch[i, :, :, :target] = ((resized - 0.5) / 0.5).transpose(2, 0, 1)
        return batch

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        try:
            probs = np.asarray(self.compiled([batch])[0], dtype=np.float32)
        except (RuntimeError, KeyError, IndexError) as exc:
            raise OcrModelError(f"recognition inference failed: {exc}") from exc
        if probs.ndim != 3:
            raise OcrModelError(f"recogniser produced shape {probs.shape}, expected [N,T,C]")
        # PaddleOCR's exported rec graph ends in softmax and its confidences are read
        # straight off the output. An IR exported without it would otherwise report logits
        # as probabilities, which reads as a systematic confidence collapse.
        if not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-2):
            shifted = probs - probs.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs

    def _decode(self, probs: np.ndarray) -> list[RecResult]:
        indices = probs.argmax(axis=2)
        scores = probs.max(axis=2)
        results: list[RecResult] = []
        for row, score in zip(indices, scores, strict=True):
            keep = np.concatenate(([True], row[1:] != row[:-1])) & (row != 0)
            picked = row[keep]
            if picked.size == 0:
                results.append(RecResult("", 0.0))
                continue
            text = "".join(
                self.charset[i] if i < len(self.charset) else "" for i in picked.tolist()
            )
            results.append(RecResult(text, float(score[keep].mean())))
        return results


def _aspect(crop: np.ndarray) -> float:
    height = max(1, crop.shape[0])
    return float(crop.shape[1]) / float(height)
