from __future__ import annotations

import hashlib

import pymupdf
import pytest

from core.cache import CacheKey, StageCache
from core.schema import BLOCK_KINDS, Document
from ingest.load import (
    PDF_MIME,
    STAGE_VERSION,
    CorruptDocumentError,
    DocumentLoadError,
    LoadResult,
    UnsupportedMimeError,
    doc_id_for,
    load_path,
    load_pdf,
    render_pages,
)

A4 = (595, 842)


def build_text_pdf() -> bytes:
    """Three pages: headings, body with a hyphen-broken word, a list, a caption, and CJK."""
    doc = pymupdf.open()
    p1 = doc.new_page(width=A4[0], height=A4[1])
    p1.insert_textbox(pymupdf.Rect(60, 60, 535, 110), "Chapter 3", fontsize=24, fontname="hebo")
    p1.insert_textbox(
        pymupdf.Rect(60, 150, 535, 190), "3.1 Light Reactions", fontsize=15, fontname="hebo"
    )
    p1.insert_textbox(
        pymupdf.Rect(60, 230, 300, 400),
        "Chloroplasts capture light energy and drive photo-\n"
        "synthesis in the thylakoid membrane. The process depends on chlorophyll pigments "
        "absorbing photons of specific wavelengths.",
        fontsize=11,
        fontname="helv",
    )
    p1.insert_textbox(
        pymupdf.Rect(60, 500, 400, 600),
        "- water splitting\n- oxygen release\n- ATP synthesis",
        fontsize=11,
        fontname="helv",
    )

    p2 = doc.new_page(width=A4[0], height=A4[1])
    p2.insert_textbox(
        pymupdf.Rect(60, 60, 535, 100), "3.2 The Calvin Cycle", fontsize=15, fontname="hebo"
    )
    p2.insert_textbox(
        pymupdf.Rect(60, 160, 535, 320),
        "Carbon fixation converts carbon dioxide into glucose using the ATP and NADPH "
        "produced by the light reactions. RuBisCO catalyses the first committed step.",
        fontsize=11,
        fontname="helv",
    )
    p2.insert_textbox(
        pymupdf.Rect(60, 420, 535, 460),
        "Figure 3.1 Structure of a chloroplast, showing stacked thylakoids.",
        fontsize=8,
        fontname="helv",
    )

    p3 = doc.new_page(width=A4[0], height=A4[1])
    p3.insert_textbox(
        pymupdf.Rect(60, 60, 535, 300),
        "光合作用是植物利用光能把二氧化碳和水转化为葡萄糖的过程。"
        "这个过程发生在叶绿体中，并且释放氧气。",
        fontsize=12,
        fontname="china-s",
    )
    return doc.tobytes()


def build_table_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=A4[0], height=A4[1])
    page.insert_textbox(pymupdf.Rect(60, 50, 535, 90), "Results", fontsize=16, fontname="hebo")
    rows = [("Sample", "Rate", "Yield"), ("A", "12.4", "0.81"), ("B", "9.7", "0.64")]
    x0, y0, width, height = 60, 140, 140, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * width, y0 + r * height, x0 + (c + 1) * width, y0 + (r + 1) * height
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (4, 5, -4, -2), cell, fontsize=10, fontname="helv")
    page.insert_textbox(
        pymupdf.Rect(60, 300, 535, 400),
        "The table above summarises the measured conversion rates for each sample under "
        "identical illumination conditions across the full experimental series.",
        fontsize=11,
        fontname="helv",
    )
    return doc.tobytes()


def build_scanned_pdf() -> bytes:
    """No text layer at all — a drawn shape stands in for a page scan."""
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=A4[0], height=A4[1])
        page.draw_rect(pymupdf.Rect(80, 80, 500, 700), color=(0, 0, 0), fill=(0.85, 0.85, 0.85))
    return doc.tobytes()


@pytest.fixture(scope="module")
def text_pdf() -> bytes:
    return build_text_pdf()


@pytest.fixture(scope="module")
def scanned_pdf() -> bytes:
    return build_scanned_pdf()


def test_doc_id_is_sha256_of_file_bytes(text_pdf):
    assert doc_id_for(text_pdf) == hashlib.sha256(text_pdf).hexdigest()
    assert doc_id_for(text_pdf) != doc_id_for(build_scanned_pdf())


def test_text_layer_is_extracted_and_ocr_is_skipped(text_pdf):
    result = load_pdf(text_pdf, "bio.pdf")
    assert result.document.has_text_layer is True
    assert result.document.n_pages == 3
    assert result.document.mime == PDF_MIME
    assert result.document.filename == "bio.pdf"
    assert result.document.doc_id == doc_id_for(text_pdf)
    assert len(result.blocks) >= 7


def test_text_layer_blocks_carry_no_ocr_confidence(text_pdf):
    """`ocr_confidence is None` is how the schema says 'this came from the native text layer'."""
    result = load_pdf(text_pdf, "bio.pdf")
    assert {b.ocr_confidence for b in result.blocks} == {None}


def test_bboxes_are_normalised_not_pdf_points(text_pdf):
    for block in load_pdf(text_pdf, "bio.pdf").blocks:
        x0, y0, x1, y1 = block.bbox
        assert 0.0 <= x0 <= x1 <= 1.0
        assert 0.0 <= y0 <= y1 <= 1.0
        assert (x1 - x0) > 0.0


def test_pages_are_one_indexed_and_within_the_document(text_pdf):
    result = load_pdf(text_pdf, "bio.pdf")
    pages = {b.page for b in result.blocks}
    assert min(pages) == 1
    assert max(pages) <= result.document.n_pages


def test_reading_order_is_unique_and_monotonic_across_pages(text_pdf):
    blocks = load_pdf(text_pdf, "bio.pdf").blocks
    orders = [b.reading_order for b in blocks]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)
    assert [b.page for b in blocks] == sorted(b.page for b in blocks)


def test_block_ids_are_unique_and_deterministic(text_pdf):
    first = load_pdf(text_pdf, "bio.pdf").blocks
    second = load_pdf(text_pdf, "bio.pdf").blocks
    assert [b.block_id for b in first] == [b.block_id for b in second]
    assert len({b.block_id for b in first}) == len(first)


def test_script_is_left_for_the_normalize_stage(text_pdf):
    assert {b.script for b in load_pdf(text_pdf, "bio.pdf").blocks} == {"unknown"}


def kind_of(blocks, needle: str) -> str:
    return next(b for b in blocks if needle in b.text).kind


def test_typography_drives_block_kind(text_pdf):
    blocks = load_pdf(text_pdf, "bio.pdf").blocks
    assert kind_of(blocks, "Chapter 3") == "heading"
    assert kind_of(blocks, "3.1 Light Reactions") == "heading"
    assert kind_of(blocks, "water splitting") == "list"
    assert kind_of(blocks, "Chloroplasts capture") == "paragraph"
    assert kind_of(blocks, "Figure 3.1") == "caption"
    assert {b.kind for b in blocks} <= BLOCK_KINDS


def test_ruled_tables_are_labelled_as_tables():
    blocks = load_pdf(build_table_pdf(), "t.pdf").blocks
    assert kind_of(blocks, "Sample") == "table"
    assert kind_of(blocks, "12.4") == "table"
    assert kind_of(blocks, "Results") == "heading"
    assert kind_of(blocks, "summarises the measured") == "paragraph"


def test_line_breaks_survive_for_the_normalize_stage(text_pdf):
    """De-hyphenation happens in `normalize`, so `load` must not collapse lines first."""
    body = next(b for b in load_pdf(text_pdf, "bio.pdf").blocks if "Chloroplasts" in b.text)
    assert "photo-\nsynthesis" in body.text


def test_a_scan_reports_no_text_layer_and_no_blocks(scanned_pdf):
    result = load_pdf(scanned_pdf, "scan.pdf")
    assert result.document.has_text_layer is False
    assert result.blocks == []
    assert result.document.n_pages == 2


def test_load_result_rejects_blocks_without_a_text_layer(text_pdf):
    real = load_pdf(text_pdf, "bio.pdf")
    scanned_doc = Document(
        doc_id=real.document.doc_id,
        filename="bio.pdf",
        mime=PDF_MIME,
        n_pages=3,
        has_text_layer=False,
        pipeline_version="v1",
    )
    with pytest.raises(ValueError, match="has_text_layer is False"):
        LoadResult(document=scanned_doc, blocks=real.blocks)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", CorruptDocumentError),
        (b"this is plainly not a pdf", UnsupportedMimeError),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, UnsupportedMimeError),
        (b"%PDF-1.7\ntruncated garbage without a trailer", CorruptDocumentError),
    ],
)
def test_malformed_input_raises_a_document_load_error(data, expected):
    with pytest.raises(expected) as exc:
        load_pdf(data, "bad.bin")
    assert isinstance(exc.value, DocumentLoadError)


def test_load_path_reads_the_file_and_names_the_document(tmp_path, text_pdf):
    path = tmp_path / "bio.pdf"
    path.write_bytes(text_pdf)
    result = load_path(path)
    assert result.document.filename == "bio.pdf"
    assert result.document.doc_id == doc_id_for(text_pdf)


def test_load_path_rejects_a_missing_file(tmp_path):
    with pytest.raises(DocumentLoadError, match="not a file"):
        load_path(tmp_path / "nope.pdf")


def test_render_pages_yields_one_raster_per_page(text_pdf):
    images = list(render_pages(text_pdf, dpi=72))
    assert [im.page for im in images] == [1, 2, 3]
    for image in images:
        assert image.pixels.shape == (image.height, image.width, 3)
        assert image.pixels.dtype.name == "uint8"
        assert image.height > 0 and image.width > 0


def test_render_pages_honours_a_page_subset(text_pdf):
    assert [im.page for im in render_pages(text_pdf, dpi=72, pages=[2])] == [2]


@pytest.mark.parametrize(
    ("kw", "match"),
    [({"dpi": 0}, "dpi must be positive"), ({"pages": [9]}, "out of range")],
)
def test_render_pages_rejects_bad_arguments(text_pdf, kw, match):
    with pytest.raises(ValueError, match=match):
        list(render_pages(text_pdf, **kw))


def test_second_load_is_served_from_cache(tmp_path, text_pdf, monkeypatch):
    cache = StageCache(tmp_path / "cache")
    first = load_pdf(text_pdf, "bio.pdf", cache=cache)

    def explode(_data):
        raise AssertionError("cache hit should not reopen the PDF")

    monkeypatch.setattr("ingest.load._open", explode)
    second = load_pdf(text_pdf, "bio.pdf", cache=cache)
    assert second == first


def test_cache_entries_land_under_the_documented_key(tmp_path, text_pdf):
    cache = StageCache(tmp_path / "cache")
    load_pdf(text_pdf, "bio.pdf", cache=cache)
    for stage in ("load.document", "load.blocks"):
        key = CacheKey(
            stage=stage,
            input_hash=doc_id_for(text_pdf),
            stage_version=STAGE_VERSION,
            config_hash="none",
        )
        assert cache.has(key)


def test_a_scan_caches_its_empty_block_list(tmp_path, scanned_pdf):
    """Otherwise every run re-parses a document already known to need OCR."""
    cache = StageCache(tmp_path / "cache")
    load_pdf(scanned_pdf, "scan.pdf", cache=cache)
    key = CacheKey(
        stage="load.blocks",
        input_hash=doc_id_for(scanned_pdf),
        stage_version=STAGE_VERSION,
        config_hash="none",
    )
    assert cache.has(key)
    assert load_pdf(scanned_pdf, "scan.pdf", cache=cache).blocks == []


def build_rotated_pdf(rotation: int) -> bytes:
    """A marker at the top-left of the unrotated page, plus filler to clear the text threshold."""
    doc = pymupdf.open()
    page = doc.new_page(width=A4[0], height=A4[1])
    page.insert_textbox(
        pymupdf.Rect(60, 60, 350, 100), "TOPLEFT MARKER", fontsize=14, fontname="helv"
    )
    page.insert_textbox(
        pymupdf.Rect(60, 400, 500, 700),
        "Filler body text so this page clears the text-layer threshold. " * 4,
        fontsize=11,
        fontname="helv",
    )
    page.set_rotation(rotation)
    return doc.tobytes()


@pytest.mark.parametrize(
    ("rotation", "corner"),
    [(0, "top-left"), (90, "top-right"), (180, "bottom-right"), (270, "bottom-left")],
)
def test_page_rotation_moves_the_bbox_to_the_displayed_corner(rotation, corner):
    """`get_text` reports unrotated coordinates while `page.rect` is rotated.

    Left uncorrected, a /Rotate 180 page cites the opposite corner and the bbox still passes
    schema validation — a wrong citation that looks entirely healthy.
    """
    blocks = load_pdf(build_rotated_pdf(rotation), "r.pdf").blocks
    x0, y0, x1, y1 = next(b for b in blocks if "TOPLEFT" in b.text).bbox
    left, top = x0 < 0.5, y0 < 0.5
    assert (("top" if top else "bottom") + "-" + ("left" if left else "right")) == corner
    assert 0.0 <= x0 <= x1 <= 1.0
    assert 0.0 <= y0 <= y1 <= 1.0


def test_an_upright_page_is_unaffected_by_the_rotation_correction():
    upright = load_pdf(build_rotated_pdf(0), "r.pdf").blocks
    assert [b.bbox for b in upright] == [
        b.bbox for b in load_pdf(build_rotated_pdf(0), "r.pdf").blocks
    ]
    marker = next(b for b in upright if "TOPLEFT" in b.text)
    assert marker.bbox[0] == pytest.approx(0.101, abs=0.01)
    assert marker.bbox[1] == pytest.approx(0.071, abs=0.01)
