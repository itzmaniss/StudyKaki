"""Rank fusion interface — ARCHITECTURE.md §1, §10.

V1 wires dense only, so most of what is asserted here is that the *seam* behaves: one list
passes through with its similarity scores intact (so `tau` keeps meaning what §4 says), and
the moment a second list appears the scores change units and the result says so.
"""

from __future__ import annotations

import pytest

from core.schema import Chunk, Retrieved
from retrieve.fusion import RRF_K, FusionResult, RankedList, fuse, reciprocal_rank_fusion
from retrieve.retriever import abstains


def chunk(cid: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc-1",
        page_start=page,
        page_end=page,
        block_ids=[f"b-{cid}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 1"],
        text=f"text of {cid}",
        token_count=10,
        lang="en",
        script="latn",
    )


def ranked(name: str, cids: list[str], *, weight: float = 1.0, scores: list[float] | None = None):
    if scores is None:
        scores = [1.0 - 0.1 * i for i in range(len(cids))]
    hits = [
        Retrieved(chunk=chunk(cid), score=s, rank=i)
        for i, (cid, s) in enumerate(zip(cids, scores, strict=True), start=1)
    ]
    return RankedList(name=name, hits=hits, weight=weight)


# --- passthrough: the only path V1 actually exercises ------------------------------------


def test_single_list_passes_through_unchanged():
    dense = ranked("dense", ["c1", "c2", "c3"])
    out = fuse([dense])

    assert isinstance(out, FusionResult)
    assert out.method == "passthrough"
    assert out.scores_are_similarities is True
    assert out.sources == ("dense",)
    assert [h.chunk.chunk_id for h in out.hits] == ["c1", "c2", "c3"]
    assert [h.score for h in out.hits] == [h.score for h in dense.hits]
    assert [h.rank for h in out.hits] == [1, 2, 3]


def test_passthrough_truncates_to_k():
    out = fuse([ranked("dense", ["c1", "c2", "c3", "c4"])], k=2)
    assert [h.chunk.chunk_id for h in out.hits] == ["c1", "c2"]


def test_passthrough_preserves_the_abstain_decision():
    """§4: with only dense wired, `tau` must still mean a cosine similarity."""
    strong = fuse([ranked("dense", ["c1"], scores=[0.81])])
    weak = fuse([ranked("dense", ["c1"], scores=[0.12])])

    assert abstains(strong.hits, 0.35) is False
    assert abstains(weak.hits, 0.35) is True


def test_rrf_scores_are_not_similarities_and_must_not_meet_tau():
    """The trap the module docstring warns about, pinned as a test.

    A perfect dense hit (cosine 0.99) becomes an RRF score of ~0.016 once fused, which is
    below tau=0.35. Anything that thresholds fused scores abstains on every query.
    """
    a = ranked("dense", ["c1"], scores=[0.99])
    b = ranked("bm25", ["c1"], scores=[0.99])
    out = fuse([a, b])

    assert out.scores_are_similarities is False
    assert out.hits[0].score == pytest.approx(2 / (RRF_K + 1))
    assert abstains(out.hits, 0.35) is True


# --- reciprocal rank fusion ---------------------------------------------------------------


def test_rrf_ranks_agreement_above_a_single_top_hit():
    """c2 is rank 2 then rank 1; c1 is rank 1 then rank 3. Agreement wins — by arithmetic."""
    a = ranked("dense", ["c1", "c2", "c3"])
    b = ranked("bm25", ["c2", "c4", "c1"])
    out = fuse([a, b])

    assert out.method == "rrf"
    assert [h.chunk.chunk_id for h in out.hits] == ["c2", "c1", "c4", "c3"]
    assert out.hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert out.hits[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert out.hits[2].score == pytest.approx(1 / 62)
    assert out.hits[3].score == pytest.approx(1 / 63)


def test_rrf_deduplicates_on_chunk_id_and_renumbers_ranks():
    out = fuse([ranked("a", ["c1", "c2"]), ranked("b", ["c2", "c1"])])

    assert len({h.chunk.chunk_id for h in out.hits}) == len(out.hits) == 2
    assert [h.rank for h in out.hits] == [1, 2]


def test_rrf_truncates_to_k():
    out = fuse([ranked("a", ["c1", "c2", "c3"]), ranked("b", ["c3", "c2", "c1"])], k=2)
    assert len(out.hits) == 2
    assert [h.rank for h in out.hits] == [1, 2]


def test_rrf_weight_shifts_the_order():
    a = ranked("dense", ["c1", "c2"], weight=3.0)
    b = ranked("bm25", ["c2", "c1"], weight=1.0)
    out = fuse([a, b])

    assert [h.chunk.chunk_id for h in out.hits] == ["c1", "c2"]
    assert out.hits[0].score == pytest.approx(3 / 61 + 1 / 62)


def test_rrf_uses_stated_rank_not_list_position():
    """A sliced list carries ranks 3..4; fusion must honour them, not renumber to 1..2."""
    full = ranked("dense", ["c1", "c2", "c3", "c4"])
    tail = RankedList(name="dense", hits=full.hits[2:])

    fused = reciprocal_rank_fusion([tail])
    assert fused[0].chunk.chunk_id == "c3"
    assert fused[0].score == pytest.approx(1 / 63)


def test_rrf_ties_break_on_first_seen_order_deterministically():
    a = ranked("a", ["c1", "c2"])
    b = ranked("b", ["c2", "c1"])
    first = [h.chunk.chunk_id for h in fuse([a, b]).hits]

    assert first == ["c1", "c2"]
    assert first == [h.chunk.chunk_id for h in fuse([a, b]).hits]


def test_rrf_over_empty_lists_returns_nothing():
    out = fuse([RankedList(name="dense"), RankedList(name="bm25")])
    assert out.hits == []
    assert abstains(out.hits, 0.35) is True


def test_ranks_are_contiguous_from_one():
    out = fuse([ranked("a", ["c1", "c2", "c3"]), ranked("b", ["c4", "c5"])])
    assert [h.rank for h in out.hits] == list(range(1, len(out.hits) + 1))


# --- malformed input ----------------------------------------------------------------------


def test_fuse_rejects_zero_lists():
    with pytest.raises(ValueError, match="at least one ranked list"):
        fuse([])


def test_fuse_rejects_duplicate_list_names():
    with pytest.raises(ValueError, match="unique"):
        fuse([ranked("dense", ["c1"]), ranked("dense", ["c2"])])


def test_ranked_list_rejects_an_empty_name():
    with pytest.raises(ValueError, match="named"):
        RankedList(name="", hits=[])


def test_ranked_list_rejects_a_negative_weight():
    with pytest.raises(ValueError, match="weight"):
        RankedList(name="dense", weight=-1.0)


def test_rrf_rejects_a_nonsense_constant():
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([ranked("a", ["c1"])], rrf_k=0)
