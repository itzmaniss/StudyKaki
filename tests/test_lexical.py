"""BM25 + per-script tokenization, and the hybrid arm — §10."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk, Retrieved
from retrieve.fusion import HybridRetriever
from retrieve.lexical import BM25Index, LexicalRetriever, tokenize

FIVE = [
    ("c1", "photosynthesis converts light into chemical energy"),
    ("c2", "boolean algebra was formalised by George Boole"),
    ("c3", "the mitochondrion is the powerhouse of the cell"),
    ("c4", "section 2.4.1 covers logic gates and truth tables"),
    ("c5", "light dependent reactions occur in the thylakoid"),
]


def make_chunk(cid: str, text: str, script: str = "latn", lang: str = "en") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc-1",
        page_start=1,
        page_end=1,
        block_ids=[f"b-{cid}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 1"],
        text=text,
        token_count=len(text.split()),
        lang=lang,
        script=script,
    )


class FakeIndex:
    """Just what LexicalRetriever touches: frame columns + chunk_at."""

    def __init__(self, chunks: list[Chunk]) -> None:
        import polars as pl

        self._chunks = chunks
        self.frame = pl.DataFrame([c.to_row() for c in chunks])
        self.index_id = "sha256:test"

    def chunk_at(self, row: int) -> Chunk:
        return self._chunks[row]


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={"paths": PathsConfig(data_dir=tmp_path / "d", ov_cache_dir=tmp_path / "o")}
    )


@pytest.fixture
def lexical(cfg):
    return LexicalRetriever(FakeIndex([make_chunk(c, t) for c, t in FIVE]), cfg)


# --- tokenization (§10: the actual work) ---------------------------------------------


def test_latin_splits_on_whitespace_and_casefolds():
    assert tokenize("Photosynthesis Converts Light", "latn") == [
        "photosynthesis",
        "converts",
        "light",
    ]


def test_section_numbers_survive_tokenization():
    """ "2.4.1" is the query this arm exists to win; splitting it loses the point."""
    assert "2.4.1" in tokenize("see section 2.4.1 now", "latn")


def test_punctuation_is_dropped_but_digits_kept():
    assert tokenize("H2O, really?", "latn") == ["h2o", "really"]


def test_tamil_splits_on_whitespace():
    assert tokenize("ஜார்ஜ் பூல்", "taml") == ["ஜார்ஜ்", "பூல்"]


def test_cjk_falls_back_to_character_bigrams():
    toks = tokenize("光合作用", "hans")
    assert toks == ["光合", "合作", "作用"]


def test_empty_text_has_no_terms():
    assert tokenize("", "latn") == []


# --- BM25 arithmetic -----------------------------------------------------------------


def test_idf_is_never_negative_for_a_term_in_every_document():
    bm25 = BM25Index([["a"], ["a"], ["a"]])
    assert bm25.idf["a"] >= 0.0


def test_unmatched_rows_score_exactly_zero():
    bm25 = BM25Index([["a", "b"], ["c"]])
    assert bm25.scores(["a"])[1] == 0.0


def test_rarer_term_outscores_a_common_one():
    bm25 = BM25Index([["rare", "common"], ["common"], ["common"]])
    assert bm25.scores(["rare"])[0] > bm25.scores(["common"])[0]


def test_bm25_rejects_nonsense_params():
    with pytest.raises(ValueError, match="k1"):
        BM25Index([["a"]], k1=-1.0)
    with pytest.raises(ValueError, match="b"):
        BM25Index([["a"]], b=1.5)


# --- retriever over 5 fixture chunks, known-correct -----------------------------------


def test_exact_term_query_finds_the_right_chunk(lexical):
    hits = lexical.retrieve("George Boole", 5)
    assert hits[0].chunk.chunk_id == "c2"


def test_section_number_query_finds_the_right_chunk(lexical):
    hits = lexical.retrieve("2.4.1", 5)
    assert hits[0].chunk.chunk_id == "c4"


def test_ranks_are_one_based_and_scores_descend(lexical):
    hits = lexical.retrieve("light", 5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))


def test_no_shared_vocabulary_returns_nothing_rather_than_padding(lexical):
    """Tamil query, English corpus. Empty beats handing RRF unearned rank credit."""
    assert lexical.retrieve("ஒளிச்சேர்க்கை", 5) == []


def test_unmatched_chunks_are_not_padded_out_to_k(lexical):
    hits = lexical.retrieve("mitochondrion", 5)
    assert len(hits) == 1


def test_retriever_rejects_bad_input(lexical):
    with pytest.raises(ValueError, match="k must be"):
        lexical.retrieve("q", 0)
    with pytest.raises(ValueError, match="empty"):
        lexical.retrieve("   ", 5)


# --- hybrid (§10) --------------------------------------------------------------------


class FakeDense:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        return self._hits[:k]


def test_hybrid_fuses_both_arms(cfg, lexical):
    chunks = [make_chunk(c, t) for c, t in FIVE]
    dense = FakeDense([Retrieved(chunk=chunks[2], score=0.8, rank=1)])

    hits = HybridRetriever(dense, lexical, cfg).retrieve("George Boole", 5)
    found = {h.chunk.chunk_id for h in hits}

    assert "c3" in found, "dense-only hit was dropped"
    assert "c2" in found, "lexical-only hit was dropped"


def test_hybrid_scores_are_rrf_not_similarities(cfg, lexical):
    """RRF tops out near 1/(60+1); feeding these to abstains(tau=0.45) abstains always."""
    chunks = [make_chunk(c, t) for c, t in FIVE]
    dense = FakeDense([Retrieved(chunk=chunks[1], score=0.9, rank=1)])

    hits = HybridRetriever(dense, lexical, cfg).retrieve("George Boole", 5)

    assert hits[0].score < 0.1
    assert hits[0].score < cfg.retrieve.tau


def test_hybrid_keeps_the_dense_top_score_for_the_abstain_decision(cfg, lexical):
    chunks = [make_chunk(c, t) for c, t in FIVE]
    dense = FakeDense([Retrieved(chunk=chunks[1], score=0.77, rank=1)])

    r = HybridRetriever(dense, lexical, cfg)
    r.retrieve("George Boole", 5)

    assert r.last_dense_top_score == pytest.approx(0.77)


def test_hybrid_survives_an_arm_returning_nothing(cfg, lexical):
    hits = HybridRetriever(FakeDense([]), lexical, cfg).retrieve("George Boole", 5)
    assert [h.chunk.chunk_id for h in hits] == ["c2"]


def test_hybrid_rejects_bad_k(cfg, lexical):
    with pytest.raises(ValueError, match="k must be"):
        HybridRetriever(FakeDense([]), lexical, cfg).retrieve("q", 0)


def test_lexical_index_row_order_matches_the_frame(lexical):
    """fusion dedups on chunk_id; that is only sound if row i is chunk i."""
    assert lexical.bm25.n_docs == len(FIVE)
    assert np.count_nonzero(lexical.bm25.scores(tokenize("mitochondrion", "latn"))) == 1
