"""`ingest/ocr_paddle.py` — the numeric core behind the OCR stage (ARCHITECTURE.md §3, §7.6).

Detection post-processing, CTC decoding and charset loading are driven with fake compiled
models returning hand-built probability arrays, so none of this needs an IR on disk. The
stage-level contract (bbox, provenance, thresholds, caching) is in `tests/test_ocr.py`.

The only text asserted on is what our own CTC decoder must produce from an array written
here by hand — never model output (CLAUDE.md).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ingest.ocr import OcrModelError, OcrParams
from ingest.ocr_paddle import (
    PaddleTextDetector,
    PaddleTextRecognizer,
    _components,
    _resize_bilinear,
    load_charset,
)

WIDTH, HEIGHT = 400, 600


class FakeDetCompiled:
    """Returns a DB probability map with one rectangular blob, in the shape it was handed."""

    def __init__(self, rect: tuple[float, float, float, float]) -> None:
        self.rect = rect

    def __call__(self, inputs):
        _, _, h, w = np.asarray(inputs[0]).shape
        prob = np.zeros((1, 1, h, w), dtype=np.float32)
        x0, y0, x1, y1 = self.rect
        prob[0, 0, int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 0.9
        return [prob]


class FakeRecCompiled:
    def __init__(self, row: np.ndarray) -> None:
        self.row = row
        self.batch_sizes: list[int] = []

    def __call__(self, inputs):
        batch = np.asarray(inputs[0])
        self.batch_sizes.append(int(batch.shape[0]))
        return [np.repeat(self.row[None], batch.shape[0], axis=0)]


def test_resize_bilinear_preserves_shape_and_constant_regions():
    src = np.full((30, 20, 3), 128, dtype=np.uint8)
    out = _resize_bilinear(src, 48, 64)
    assert out.shape == (48, 64, 3)
    assert out == pytest.approx(128.0)


def test_components_separates_disjoint_blobs():
    prob = np.zeros((6, 6), dtype=np.float32)
    prob[0:2, 0:3] = 0.8
    prob[4:6, 4:6] = 0.4
    found = dict(_components(prob > 0.1, prob, 10))
    assert found.pop((0, 0, 3, 2)) == pytest.approx(0.8)
    assert found.pop((4, 4, 6, 6)) == pytest.approx(0.4)
    assert found == {}


def test_detected_boxes_come_back_in_original_page_pixels():
    """The whole point of the det post-process: resize + unclip must invert exactly enough
    that the box centre lands where the ink is."""
    detector = PaddleTextDetector(
        FakeDetCompiled((0.25, 0.4, 0.75, 0.5)), OcrParams(), fingerprint="det/fake"
    )
    (box,) = detector.detect(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8))
    assert (box.x0 + box.x1) / 2 == pytest.approx(0.5 * WIDTH, abs=1.0)
    assert (box.y0 + box.y1) / 2 == pytest.approx(0.45 * HEIGHT, abs=1.0)
    assert 0.0 <= box.x0 < box.x1 <= WIDTH
    assert 0.0 <= box.y0 < box.y1 <= HEIGHT
    assert box.score == pytest.approx(0.9)


def test_a_blank_page_detects_nothing():
    detector = PaddleTextDetector(
        FakeDetCompiled((0.0, 0.0, 0.0, 0.0)), OcrParams(), fingerprint="det/fake"
    )
    assert detector.detect(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)) == []


def test_a_detector_that_is_not_a_probability_map_is_refused():
    class Wrong:
        def __call__(self, inputs):
            return [np.zeros((1, 3, 8, 8), dtype=np.float32)]

    detector = PaddleTextDetector(Wrong(), OcrParams(), fingerprint="det/fake")
    with pytest.raises(OcrModelError, match=r"expected \[N,1,H,W\]"):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))


def ctc_probs() -> np.ndarray:
    """argmax path blank/a/a/blank/b — CTC must collapse it to two characters."""
    return np.array(
        [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]], dtype=np.float32
    )


def test_ctc_decode_collapses_repeats_and_blanks_and_averages_their_scores():
    compiled = FakeRecCompiled(ctc_probs())
    rec = PaddleTextRecognizer(compiled, ["<blank>", "a", "b"], OcrParams(), "rec/fake")
    (result,) = rec.recognize([np.full((20, 100, 3), 30, dtype=np.uint8)])
    assert result.text == "ab"
    assert result.confidence == pytest.approx(0.7)
    assert 0.0 <= result.confidence <= 1.0


def test_crops_are_batched_and_results_return_in_input_order():
    compiled = FakeRecCompiled(ctc_probs())
    rec = PaddleTextRecognizer(
        compiled, ["<blank>", "a", "b"], OcrParams(rec_batch_size=2), "rec/fake"
    )
    crops = [np.full((20, 40 * (i + 1), 3), 30, dtype=np.uint8) for i in range(5)]
    results = rec.recognize(crops)
    assert compiled.batch_sizes == [2, 2, 1]
    assert len(results) == 5
    assert {r.text for r in results} == {"ab"}


def test_a_recogniser_output_of_the_wrong_rank_is_refused():
    class Wrong:
        def __call__(self, inputs):
            return [np.zeros((2, 3), dtype=np.float32)]

    rec = PaddleTextRecognizer(Wrong(), ["<blank>", "a"], OcrParams(), "rec/fake")
    with pytest.raises(OcrModelError, match=r"expected \[N,T,C\]"):
        rec.recognize([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_a_recogniser_head_must_name_a_known_script():
    with pytest.raises(OcrModelError, match="not one of"):
        PaddleTextRecognizer(None, ["<blank>"], OcrParams(), "rec/fake", script="klingon")


def write_charset(tmp_path, chars: list[str]):
    (tmp_path / "config.json").write_text(json.dumps({"PostProcess": {"character_dict": chars}}))
    return tmp_path


def test_charset_prepends_the_ctc_blank_and_appends_the_space_class(tmp_path):
    ir_dir = write_charset(tmp_path, ["a", "b"])
    assert load_charset(ir_dir) == ["<blank>", "a", "b"]
    assert load_charset(ir_dir, 4) == ["<blank>", "a", "b", " "]


def test_a_charset_that_does_not_match_the_ir_is_refused(tmp_path):
    with pytest.raises(OcrModelError, match="different models"):
        load_charset(write_charset(tmp_path, ["a", "b"]), 9)


def test_a_missing_charset_names_the_setup_command(tmp_path):
    with pytest.raises(OcrModelError, match="scripts.setup"):
        load_charset(tmp_path)


class TestSoftmaxDetection:
    """Regression: INT8 rounding must not be mistaken for logits.

    PaddleOCR's exported rec graph already ends in softmax. INT8 weight compression pushes the
    row sums a few percent off 1.0 — measured at 1.026 on PP-OCRv5_mobile_rec. A tight
    tolerance reads that as "these are logits" and softmaxes a second time, flattening a real
    distribution to near-uniform: peak probability fell 0.97 -> 1.4e-4, every confidence
    rounded to 0.00, `min_confidence` dropped every block, and the stage returned zero blocks
    while raising nothing. The whole OCR suite stayed green throughout, because every test
    used a fake whose rows summed to exactly 1.0.
    """

    @staticmethod
    def _rec(compiled, charset=("<blank>", "a", "b", "c")):
        return PaddleTextRecognizer(compiled, list(charset), OcrParams(), "rec/fake")

    @staticmethod
    def _compiled(array):
        class Compiled:
            def __call__(self, _inputs):
                return {0: array}

            def __getitem__(self, _key):
                return array

        return Compiled()

    def test_int8_drifted_probabilities_are_not_resoftmaxed(self):
        """Rows summing to 1.026 are probabilities, not logits — confidence must survive."""
        probs = np.zeros((1, 4, 4), dtype=np.float32)
        probs[0, :, 1] = 0.996
        probs[0, :, 0] = 0.030
        assert np.allclose(probs.sum(axis=-1), 1.026), "fixture must reproduce the measured drift"

        out = self._rec(self._compiled(probs))._infer(np.zeros((1, 3, 48, 320), np.float32))
        assert out.max() > 0.5, "already-softmaxed rows must pass through untouched"

    def test_genuine_logits_are_softmaxed(self):
        """The guard must still fire for a graph exported without its softmax."""
        logits = np.array([[[-4.0, 9.0, -2.0, -3.0]] * 3], dtype=np.float32)
        out = self._rec(self._compiled(logits))._infer(np.zeros((1, 3, 48, 320), np.float32))
        assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-4)
        assert out.max() > 0.9

    def test_confidence_survives_decode(self):
        """End to end through _decode: a confident row must not report ~0.0."""
        probs = np.full((1, 3, 4), 0.01, dtype=np.float32)
        probs[0, :, 1] = 0.97
        rec = self._rec(self._compiled(probs))
        results = rec._decode(rec._infer(np.zeros((1, 3, 48, 320), np.float32)))
        assert results[0].confidence > 0.5, "a 0.97 peak must not collapse to zero"
        assert results[0].text == "a"
