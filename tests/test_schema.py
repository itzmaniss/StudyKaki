from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from core.schema import Answer, Block, Chunk, Document, Retrieved


def make_block(**kw) -> Block:
    base = dict(
        block_id="b1",
        doc_id="d1",
        page=1,
        bbox=(0.1, 0.2, 0.8, 0.9),
        kind="paragraph",
        reading_order=0,
        script="taml",
        text="ஒளிச்சேர்க்கை",
        ocr_confidence=0.42,
    )
    return Block(**{**base, **kw})


def make_chunk(**kw) -> Chunk:
    base = dict(
        chunk_id="c1",
        doc_id="d1",
        page_start=1,
        page_end=2,
        block_ids=["b1", "b2"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3", "3.2 Photosynthesis"],
        text="hello",
        token_count=5,
        lang="ta",
        script="taml",
    )
    return Chunk(**{**base, **kw})


@pytest.mark.parametrize(
    "obj",
    [
        make_block(),
        make_block(ocr_confidence=None, script="latn", text="native text layer"),
        make_chunk(),
        Document(
            doc_id="d1",
            filename="bio.pdf",
            mime="application/pdf",
            n_pages=3,
            has_text_layer=True,
            pipeline_version="v1",
        ),
    ],
)
def test_parquet_round_trip_is_lossless(obj, tmp_path):
    """dataclass -> parquet -> dataclass, unchanged (CLAUDE.md testing expectations)."""
    path = tmp_path / "rt.parquet"
    pl.DataFrame([obj.to_row()]).write_parquet(path, compression="zstd")
    row = pl.read_parquet(path).to_dicts()[0]
    assert type(obj).from_row(row) == obj


def test_bbox_survives_round_trip_as_tuple(tmp_path):
    """Parquet stores tuples as lists; a list bbox would break equality and hashing."""
    path = tmp_path / "b.parquet"
    block = make_block()
    pl.DataFrame([block.to_row()]).write_parquet(path)
    restored = Block.from_row(pl.read_parquet(path).to_dicts()[0])
    assert isinstance(restored.bbox, tuple)
    assert restored.bbox == block.bbox


def test_frozen_dataclasses_are_immutable():
    block = make_block()
    with pytest.raises(FrozenInstanceError):
        block.page = 2  # type: ignore[misc]


class TestMalformedInput:
    def test_page_must_be_one_indexed(self):
        with pytest.raises(ValueError, match="1-indexed"):
            make_block(page=0)

    def test_bbox_must_be_normalised(self):
        with pytest.raises(ValueError, match="normalised"):
            make_block(bbox=(0.0, 0.0, 612.0, 792.0))

    def test_bbox_must_not_be_inverted(self):
        with pytest.raises(ValueError, match="inverted"):
            make_block(bbox=(0.9, 0.9, 0.1, 0.1))

    def test_unknown_script_rejected(self):
        with pytest.raises(ValueError, match="unknown script"):
            make_block(script="klingon")

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown block kind"):
            make_block(kind="footnote")

    def test_ocr_confidence_bounds(self):
        with pytest.raises(ValueError, match="ocr_confidence"):
            make_block(ocr_confidence=1.5)

    def test_chunk_page_range_must_not_invert(self):
        with pytest.raises(ValueError, match="precedes"):
            make_chunk(page_start=5, page_end=2)

    def test_chunk_needs_block_provenance(self):
        with pytest.raises(ValueError, match="at least one block"):
            make_chunk(block_ids=[])

    def test_abstained_answer_cannot_cite(self):
        hit = Retrieved(chunk=make_chunk(), score=0.9, rank=1)
        with pytest.raises(ValueError, match="abstained"):
            Answer(text="...", citations=[hit], abstained=True, trace_id="t1")

    def test_rank_is_one_indexed(self):
        with pytest.raises(ValueError, match="1-indexed"):
            Retrieved(chunk=make_chunk(), score=0.9, rank=0)
