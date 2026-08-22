"""OCR stage tests — ARCHITECTURE.md §3 `ocr`, §0.2 provenance.

The stage is driven with a stub detector and stub recogniser (the protocols in `ingest/ocr.py`
exist for exactly this), so bbox arithmetic, reading order, thresholds and caching are
verified without an IR on disk. `ingest/ocr_paddle.py`'s numeric core is driven with fake
compiled models in `tests/test_ocr_paddle.py`.

Per CLAUDE.md, nothing here asserts on recognised text produced by a *model* — the only text
asserted on is text the stub was told to return, or text our own CTC decoder must produce
from a probability array we wrote by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from core.cache import StageCache
from core.schema import BLOCK_KINDS, Block
from ingest.load import PageImage
from ingest.ocr import (
    DEFAULT_SCRIPT,
    DetBox,
    OcrEngine,
    OcrError,
    OcrModelError,
    OcrParams,
    RecResult,
    ocr_page,
    ocr_pages,
)

DOC_ID = "f" * 64
WIDTH, HEIGHT = 400, 600


@dataclass(frozen=True)
class Line:
    """A text line in page-image **pixels**, painted into the page so a stub can find it."""

    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    confidence: float = 0.9


class StubDetector:
    def __init__(self, boxes: list[DetBox], fingerprint: str = "det/stub", error=None) -> None:
        self.boxes = boxes
        self.fingerprint = fingerprint
        self.error = error
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[DetBox]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.boxes)


class StubRecognizer:
    """Reads the tag painted into each crop, so a result follows its crop, not the call order."""

    def __init__(self, table, script: str = DEFAULT_SCRIPT, fingerprint: str = "rec/stub") -> None:
        self.table = dict(table)
        self.script = script
        self.fingerprint = fingerprint
        self.batches: list[int] = []

    def recognize(self, crops):
        self.batches.append(len(crops))
        return [self.table.get(int(c[0, 0, 0]), RecResult("", 0.0)) for c in crops]


def build_page(lines: list[Line], *, page: int = 1, width: int = WIDTH, height: int = HEIGHT):
    """A white page with each line painted as a solid grey tag, plus stubs that agree with it."""
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    table: dict[int, RecResult] = {}
    boxes: list[DetBox] = []
    for i, line in enumerate(lines):
        tag = i + 1
        pixels[line.y0 : line.y1, line.x0 : line.x1] = tag
        table[tag] = RecResult(line.text, line.confidence)
        boxes.append(DetBox(*(float(v) for v in (line.x0, line.y0, line.x1, line.y1)), 0.95))
    image = PageImage(page=page, dpi=200, pixels=pixels)
    return image, StubDetector(boxes), StubRecognizer(table)


def engine_for(lines: list[Line], *, params: OcrParams | None = None, page: int = 1, **kw):
    image, detector, recognizer = build_page(lines, page=page, **kw)
    engine = OcrEngine(detector, {DEFAULT_SCRIPT: recognizer}, params=params)
    return image, engine, detector, recognizer


BODY = [
    Line(40, 60, 200, 90, "Chapter 3"),
    Line(40, 150, 360, 174, "Chloroplasts capture light energy."),
]


# --- provenance: bbox, page, reading order -------------------------------------------------


def test_bbox_is_normalised_against_the_page_image():
    image, engine, _, _ = engine_for([Line(40, 60, 200, 90, "Chapter 3")])
    (block,) = engine.read_page(image, DOC_ID)
    assert block.bbox == pytest.approx((40 / WIDTH, 60 / HEIGHT, 200 / WIDTH, 90 / HEIGHT))
    assert block.bbox == pytest.approx((0.1, 0.1, 0.5, 0.15))


def test_raw_pixel_bbox_is_rejected_by_the_schema():
    """Why the normalisation above is load-bearing: the un-normalised box cannot be stored."""
    with pytest.raises(ValueError, match="normalised to 0-1"):
        Block(
            block_id="b",
            doc_id=DOC_ID,
            page=1,
            bbox=(40.0, 60.0, 200.0, 90.0),
            kind="paragraph",
            reading_order=0,
            script="latn",
            text="Chapter 3",
            ocr_confidence=0.9,
        )


def test_bbox_is_clamped_to_the_page_and_degenerate_boxes_are_dropped():
    image, _, recognizer = build_page([Line(10, 10, 50, 40, "in bounds")])
    detector = StubDetector(
        [
            DetBox(-20.0, -30.0, float(WIDTH) + 99, float(HEIGHT) + 99, 0.9),
            DetBox(10.0, 10.0, 10.0, 40.0, 0.9),
        ]
    )
    recognizer.table[255] = RecResult("overflowing line", 0.9)
    blocks = OcrEngine(detector, {DEFAULT_SCRIPT: recognizer}).read_page(image, DOC_ID)
    assert [b.bbox for b in blocks] == [(0.0, 0.0, 1.0, 1.0)]


def test_page_is_one_indexed_and_carried_onto_every_block():
    image, engine, _, _ = engine_for(BODY, page=7)
    blocks = engine.read_page(image, DOC_ID)
    assert {b.page for b in blocks} == {7}


def test_page_below_one_is_refused():
    image, engine, _, _ = engine_for(BODY)
    with pytest.raises(OcrError, match="1-indexed"):
        engine.read_page(PageImage(page=0, dpi=200, pixels=image.pixels), DOC_ID)


def test_reading_order_is_top_to_bottom_then_left_to_right():
    lines = [
        Line(200, 300, 360, 324, "second row right"),
        Line(40, 60, 180, 84, "first row left"),
        Line(40, 300, 180, 324, "second row left"),
        Line(200, 60, 360, 84, "first row right"),
    ]
    image, engine, _, _ = engine_for(lines)
    blocks = engine.read_page(image, DOC_ID)
    assert [b.text for b in blocks] == [
        "first row left",
        "first row right",
        "second row left",
        "second row right",
    ]
    assert [b.reading_order for b in blocks] == [0, 1, 2, 3]


def test_reading_order_is_renumbered_across_pages_and_pages_are_preserved():
    images, detectors, recognizers = zip(
        *(build_page(BODY, page=p) for p in (1, 2, 3)), strict=True
    )
    engine = OcrEngine(detectors[0], {DEFAULT_SCRIPT: recognizers[0]})
    blocks = ocr_pages(images, doc_id=DOC_ID, engine=engine)
    assert [b.reading_order for b in blocks] == list(range(6))
    assert [b.page for b in blocks] == [1, 1, 2, 2, 3, 3]


def test_block_ids_are_page_local_unique_and_deterministic():
    image, engine, _, _ = engine_for(BODY, page=4)
    first = engine.read_page(image, DOC_ID)
    second = engine.read_page(image, DOC_ID)
    assert [b.block_id for b in first] == [b.block_id for b in second]
    assert len({b.block_id for b in first}) == len(first)
    assert all(b.block_id.startswith(f"{DOC_ID[:12]}:4:") for b in first)


# --- confidence ----------------------------------------------------------------------------


def test_ocr_confidence_is_populated_and_in_range():
    """`None` is reserved for native-text-layer blocks (§2); an OCR'd block must carry a number."""
    image, engine, _, _ = engine_for(BODY)
    for block in engine.read_page(image, DOC_ID):
        assert block.ocr_confidence is not None
        assert 0.0 <= block.ocr_confidence <= 1.0


def test_out_of_range_recogniser_confidence_is_clamped():
    image, engine, _, _ = engine_for([Line(40, 60, 200, 90, "loud", confidence=1.4)])
    (block,) = engine.read_page(image, DOC_ID)
    assert block.ocr_confidence == 1.0


def test_lines_below_the_threshold_are_dropped_and_the_rest_stay_contiguous():
    lines = [
        Line(40, 60, 200, 90, "kept", confidence=0.9),
        Line(40, 150, 200, 180, "noise", confidence=0.05),
        Line(40, 250, 200, 280, "also kept", confidence=0.4),
    ]
    image, engine, _, _ = engine_for(lines)
    blocks = engine.read_page(image, DOC_ID)
    assert [b.text for b in blocks] == ["kept", "also kept"]
    assert [b.reading_order for b in blocks] == [0, 1]


def test_default_threshold_does_not_drop_correct_low_confidence_tamil():
    """CLAUDE.md: Tamil reads 0.4-0.6 on clean scans where Latin reads 0.9+. A uniform 0.7
    would silently drop most correct Tamil, so the default sits below that floor."""
    assert OcrParams().min_confidence < 0.4
    assert OcrParams().confidence_by_script == {}
    image, engine, _, _ = engine_for([Line(40, 60, 300, 90, "தமிழ் உரை", confidence=0.45)])
    (block,) = engine.read_page(image, DOC_ID)
    assert block.ocr_confidence == pytest.approx(0.45)


def test_per_script_threshold_reaches_lines_read_by_the_multilingual_default_head():
    """The shipping engine has one head, so a threshold keyed on the *head's* script would
    make `confidence_by_script` a silent no-op — the exact trap CLAUDE.md warns about."""
    params = OcrParams(confidence_by_script={"taml": 0.6})
    lines = [
        Line(40, 60, 300, 90, "தமிழ் உரை", confidence=0.45),
        Line(40, 150, 300, 180, "latin line", confidence=0.45),
    ]
    image, engine, _, _ = engine_for(lines, params=params)
    assert [b.text for b in engine.read_page(image, DOC_ID)] == ["latin line"]


def test_blank_recognitions_are_dropped():
    image, engine, _, _ = engine_for([Line(40, 60, 200, 90, "   ", confidence=0.99)])
    assert engine.read_page(image, DOC_ID) == []


# --- classification ------------------------------------------------------------------------


def test_kind_is_classified_from_geometry_and_is_never_table():
    """A line-level OCR box cannot see table structure, so `kind` is never `table` here."""
    lines = [
        Line(40, 40, 300, 90, "Chapter 3"),
        Line(40, 150, 360, 174, "Chloroplasts capture light energy."),
        Line(40, 200, 360, 224, "- water splitting"),
        Line(40, 250, 360, 274, "Figure 3.1 Structure of a chloroplast."),
    ]
    image, engine, _, _ = engine_for(lines)
    kinds = {b.text: b.kind for b in engine.read_page(image, DOC_ID)}
    assert kinds["Chapter 3"] == "heading"
    assert kinds["Chloroplasts capture light energy."] == "paragraph"
    assert kinds["- water splitting"] == "list"
    assert kinds["Figure 3.1 Structure of a chloroplast."] == "caption"
    assert set(kinds.values()) <= BLOCK_KINDS - {"table"}


# --- malformed input and model failure ------------------------------------------------------


@pytest.mark.parametrize(
    ("pixels", "match"),
    [
        (np.zeros((10, 10, 3), dtype=np.float32), "uint8"),
        (np.zeros((2, 10, 10, 3), dtype=np.uint8), "HxW"),
        (np.zeros((0, 10, 3), dtype=np.uint8), "empty"),
        (np.zeros((10, 10, 2), dtype=np.uint8), "channels"),
        ([[0, 0], [0, 0]], "numpy array"),
    ],
)
def test_unreadable_page_images_are_refused_with_a_reason(pixels, match):
    _, engine, _, _ = engine_for(BODY)
    with pytest.raises(OcrError, match=match):
        engine.read_page(PageImage(page=1, dpi=200, pixels=pixels), DOC_ID)


@pytest.mark.parametrize("shape", [(HEIGHT, WIDTH), (HEIGHT, WIDTH, 4)])
def test_greyscale_and_rgba_pages_are_accepted(shape):
    _, engine, _, recognizer = engine_for(BODY)
    pixels = np.full(shape, 7, dtype=np.uint8)
    recognizer.table[7] = RecResult("read anyway", 0.9)
    blocks = engine.read_page(PageImage(page=1, dpi=200, pixels=pixels), DOC_ID)
    assert [b.text for b in blocks] == ["read anyway", "read anyway"]


def test_model_failure_is_not_swallowed():
    image, _, recognizer = build_page(BODY)
    detector = StubDetector([], error=OcrModelError("detection inference failed"))
    engine = OcrEngine(detector, {DEFAULT_SCRIPT: recognizer})
    with pytest.raises(OcrModelError, match="detection inference failed"):
        engine.read_page(image, DOC_ID)


def test_a_page_with_no_detections_yields_no_blocks_and_no_recognition():
    image, _, recognizer = build_page(BODY)
    engine = OcrEngine(StubDetector([]), {DEFAULT_SCRIPT: recognizer})
    assert engine.read_page(image, DOC_ID) == []
    assert recognizer.batches == []


# --- engine construction and per-script heads ----------------------------------------------


@pytest.mark.parametrize(
    ("heads", "match"),
    [
        ({}, "at least one"),
        ({"klingon": "x"}, "unknown scripts"),
        ({"taml": "t", "latn": "l"}, "no 'unknown' default"),
        ({DEFAULT_SCRIPT: "u", "taml": "mislabelled"}, "disagrees"),
    ],
)
def test_engine_refuses_incoherent_recognition_heads(heads, match):
    built = {
        key: StubRecognizer({}, script=DEFAULT_SCRIPT if value == "mislabelled" else key)
        for key, value in heads.items()
    }
    with pytest.raises(OcrError, match=match):
        OcrEngine(StubDetector([]), built)


def test_a_dedicated_script_head_re_reads_the_lines_it_owns():
    image, detector, primary = build_page([Line(40, 60, 300, 90, "தமிழ", confidence=0.5)])
    tamil = StubRecognizer({1: RecResult("தமிழ் உரை", 0.55)}, script="taml", fingerprint="rec/taml")
    engine = OcrEngine(detector, {DEFAULT_SCRIPT: primary, "taml": tamil})
    (block,) = engine.read_page(image, DOC_ID)
    assert block.script == "taml"
    assert block.text == "தமிழ் உரை"
    assert block.ocr_confidence == pytest.approx(0.55)


# --- caching -------------------------------------------------------------------------------


def test_the_second_call_hits_the_cache_and_returns_identical_blocks(tmp_path):
    image, engine, detector, _ = engine_for(BODY)
    cache = StageCache(tmp_path)
    first = ocr_page(image, doc_id=DOC_ID, engine=engine, cache=cache)
    second = ocr_page(image, doc_id=DOC_ID, engine=engine, cache=cache)
    assert detector.calls == 1
    assert first == second
    assert [b.bbox for b in second] == [b.bbox for b in first]
    assert all(isinstance(b.bbox, tuple) and len(b.bbox) == 4 for b in second)


def test_changed_ocr_params_invalidate_the_cache(tmp_path):
    image, detector, recognizer = build_page(BODY)
    cache = StageCache(tmp_path)
    ocr_page(
        image, doc_id=DOC_ID, engine=OcrEngine(detector, {DEFAULT_SCRIPT: recognizer}), cache=cache
    )
    tuned = OcrEngine(detector, {DEFAULT_SCRIPT: recognizer}, params=OcrParams(min_confidence=0.5))
    ocr_page(image, doc_id=DOC_ID, engine=tuned, cache=cache)
    assert detector.calls == 2


def test_a_swapped_model_invalidates_the_cache(tmp_path):
    image, detector, recognizer = build_page(BODY)
    cache = StageCache(tmp_path)
    ocr_page(
        image, doc_id=DOC_ID, engine=OcrEngine(detector, {DEFAULT_SCRIPT: recognizer}), cache=cache
    )
    swapped = StubRecognizer(recognizer.table, fingerprint="rec/v2")
    ocr_page(
        image,
        doc_id=DOC_ID,
        engine=OcrEngine(detector, {DEFAULT_SCRIPT: swapped}),
        cache=cache,
    )
    assert detector.calls == 2


def test_a_different_page_is_a_different_cache_entry(tmp_path):
    cache = StageCache(tmp_path)
    image, engine, detector, _ = engine_for(BODY, page=1)
    ocr_page(image, doc_id=DOC_ID, engine=engine, cache=cache)
    other = PageImage(page=2, dpi=200, pixels=image.pixels)
    ocr_page(other, doc_id=DOC_ID, engine=engine, cache=cache)
    assert detector.calls == 2


# --- params --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"det_limit_side_len": 16}, "det_limit_side_len"),
        ({"rec_batch_size": 0}, "rec_batch_size"),
        ({"rec_min_width": 400, "rec_max_width": 100}, "exceeds"),
        ({"min_confidence": 1.5}, "min_confidence must be 0-1"),
        ({"det_thresh": -0.1}, "det_thresh must be 0-1"),
        ({"confidence_by_script": {"klingon": 0.5}}, "unknown scripts"),
    ],
)
def test_params_validate_on_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OcrParams(**kwargs)


def test_params_digest_is_stable_and_independent_of_mapping_order():
    a = OcrParams(confidence_by_script={"taml": 0.3, "latn": 0.8})
    b = OcrParams(confidence_by_script={"latn": 0.8, "taml": 0.3})
    assert a.digest() == b.digest()
    assert a.digest() != OcrParams().digest()
