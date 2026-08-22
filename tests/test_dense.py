"""Dense retrieval — ARCHITECTURE.md §4, §3.1.

Everything runs against a **synthetic index built in `tmp_path`**: five chunks whose vectors
are written by hand so the correct ranking is arithmetic, not opinion. The real BGE-M3 INT8
IR is 544 MB and is not committed (§7.3), so the embedder fingerprint is exercised through
the same stub IR `tests/test_registry.py` uses — the fingerprint is a hash of weights and a
JSON block, and neither cares that the weights are a stub.

The fixture vectors are deliberately **not** unit-norm, and the one with the largest raw dot
product against the query is not the closest by angle. Any regression from cosine back to a
plain dot product flips the expected order and fails loudly.

No test touches `data/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk, Retrieved
from eval.metrics import GoldQuestion
from eval.run import evaluate
from models.registry import FingerprintMismatch, embedder_fingerprint
from retrieve.dense import (
    CHUNKS_NAME,
    INDEX_MANIFEST_NAME,
    VECTORS_NAME,
    DenseIndex,
    DenseIndexError,
    DenseRetriever,
    IndexCorrupt,
    IndexNotFound,
    default_index_dir,
    load_index,
    query_encoder,
)
from retrieve.fusion import RankedList, fuse
from retrieve.retriever import Retriever, abstains
from tests.test_registry import EMBEDDING, write_manifest

DIM = EMBEDDING["dim"]

#: (chunk_id, page, vector). Row order on disk is deliberately not score order.
FIVE_CHUNKS = [
    ("c1", 11, (1.0, 3.0, 0.0)),
    ("c2", 12, (1.0, 0.0, 0.0)),
    ("c3", 13, (0.0, 1.0, 0.0)),
    ("c4", 14, (3.0, 1.0, 0.0)),
    ("c5", 15, (1.0, 1.0, 0.0)),
]

#: Against query (1, 0, 0): c2 is collinear; c4 has the biggest dot product but not angle.
EXPECTED_ORDER = ["c2", "c4", "c5", "c1", "c3"]
EXPECTED_SCORES = [1.0, 3 / np.sqrt(10), 1 / np.sqrt(2), 1 / np.sqrt(10), 0.0]


def vec(*values: float) -> np.ndarray:
    """A readable low-dimensional vector padded into the embedder's real width."""
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return v


def make_chunk(chunk_id: str, page: int, doc_id: str = "doc-bio") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page_start=page,
        page_end=page,
        block_ids=[f"blk-{chunk_id}-a", f"blk-{chunk_id}-b"],
        bbox_union=(0.1, 0.2, 0.9, 0.8),
        heading_path=["Chapter 3", "3.2 Photosynthesis"],
        text=f"body text of {chunk_id}",
        token_count=42,
        lang="en",
        script="latn",
    )


def write_index(
    root: Path,
    *,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedder: dict,
    index_id: str = "sha256:testindex",
    n_vectors: int | None = None,
    omit: tuple[str, ...] = (),
    manifest_text: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    if CHUNKS_NAME not in omit:
        rows = [c.to_row() for c in chunks]
        frame = pl.DataFrame(rows) if rows else pl.DataFrame([make_chunk("x", 1).to_row()]).clear()
        frame.write_parquet(root / CHUNKS_NAME, compression="zstd")

    if VECTORS_NAME not in omit:
        np.save(root / VECTORS_NAME, vectors)

    if INDEX_MANIFEST_NAME not in omit:
        text = manifest_text
        if text is None:
            text = json.dumps(
                {
                    "index_id": index_id,
                    "n_vectors": len(chunks) if n_vectors is None else n_vectors,
                    "chunk_config_hash": "sha256:chunkcfg",
                    "embedder": embedder,
                }
            )
        (root / INDEX_MANIFEST_NAME).write_text(text)
    return root


class FakeEncoder:
    """Stands in for `ingest/embed.py`. Records what it was asked to embed."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.seen: list[str] = []
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        self.seen.extend(texts)
        return np.asarray([self.vector], dtype=np.float32)


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=tmp_path / "ov_cache")
        }
    )


@pytest.fixture
def models_manifest(tmp_path):
    return write_manifest(tmp_path / "models", embedding=EMBEDDING)


@pytest.fixture
def fingerprint(cfg, models_manifest):
    """Exactly what `ingest/index.py` must stamp into `index_manifest.json` (§3.1)."""
    return embedder_fingerprint(cfg, manifest_path=models_manifest)


@pytest.fixture
def index(tmp_path, fingerprint):
    chunks = [make_chunk(cid, page) for cid, page, _ in FIVE_CHUNKS]
    vectors = np.stack([vec(*v) for _, _, v in FIVE_CHUNKS])
    write_index(tmp_path / "index", chunks=chunks, vectors=vectors, embedder=fingerprint)
    return load_index(tmp_path / "index")


@pytest.fixture
def retriever(index, cfg, models_manifest):
    return DenseRetriever(
        index, cfg, FakeEncoder(vec(1.0, 0.0, 0.0)), models_manifest_path=models_manifest
    )


# --- the known-correct answer over five fixture chunks -------------------------------------


def test_five_chunks_rank_in_the_arithmetically_certain_order(retriever):
    hits = retriever.retrieve("what is photosynthesis", k=5)

    assert [h.chunk.chunk_id for h in hits] == EXPECTED_ORDER
    for hit, expected in zip(hits, EXPECTED_SCORES, strict=True):
        assert hit.score == pytest.approx(expected, abs=1e-6)


def test_ranking_is_cosine_not_dot_product(retriever):
    """c4 wins on raw dot product (3.0 vs 1.0) and still must lose on angle."""
    hits = retriever.retrieve("photosynthesis", k=2)

    assert hits[0].chunk.chunk_id == "c2"
    assert hits[1].chunk.chunk_id == "c4"


def test_ranks_are_contiguous_from_one(retriever):
    hits = retriever.retrieve("photosynthesis", k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(isinstance(h, Retrieved) for h in hits)


def test_scores_are_monotonically_descending(retriever):
    scores = [h.score for h in retriever.retrieve("photosynthesis", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_k_truncates_without_changing_the_order(retriever):
    top2 = [h.chunk.chunk_id for h in retriever.retrieve("photosynthesis", k=2)]
    assert top2 == EXPECTED_ORDER[:2]


def test_k_larger_than_the_corpus_returns_everything(retriever):
    hits = retriever.retrieve("photosynthesis", k=500)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_provenance_rides_on_every_hit(retriever):
    """§0.2 — citations depend on doc_id/page/bbox and cannot be retrofitted."""
    top = retriever.retrieve("photosynthesis", k=1)[0]

    assert top.chunk.doc_id == "doc-bio"
    assert (top.chunk.page_start, top.chunk.page_end) == (12, 12)
    assert top.chunk.bbox_union == (0.1, 0.2, 0.9, 0.8)
    assert top.chunk.heading_path == ["Chapter 3", "3.2 Photosynthesis"]
    assert top.chunk.block_ids == ["blk-c2-a", "blk-c2-b"]
    assert top.chunk.text == "body text of c2"


def test_ties_break_on_row_order_deterministically(tmp_path, cfg, models_manifest, fingerprint):
    chunks = [make_chunk(f"t{i}", i + 1) for i in range(4)]
    vectors = np.stack([vec(1.0, 0.0, 0.0) for _ in chunks])
    write_index(tmp_path / "tied", chunks=chunks, vectors=vectors, embedder=fingerprint)
    r = DenseRetriever(
        load_index(tmp_path / "tied"),
        cfg,
        FakeEncoder(vec(1.0, 0.0, 0.0)),
        models_manifest_path=models_manifest,
    )

    assert [h.chunk.chunk_id for h in r.retrieve("q", k=4)] == ["t0", "t1", "t2", "t3"]
    assert [h.chunk.chunk_id for h in r.retrieve("q", k=4)] == ["t0", "t1", "t2", "t3"]


def test_blocked_scan_matches_an_unblocked_one(index, cfg, models_manifest):
    """Peak RSS is bounded by blocking the scan; the answer must not depend on block size."""
    args = (index, cfg, FakeEncoder(vec(1.0, 2.0, 0.5)))
    fine = DenseRetriever(*args, models_manifest_path=models_manifest, block_rows=1)
    coarse = DenseRetriever(*args, models_manifest_path=models_manifest, block_rows=10_000)

    a = fine.retrieve("q", k=5)
    b = coarse.retrieve("q", k=5)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]
    assert [h.score for h in a] == pytest.approx([h.score for h in b], abs=1e-6)


def test_zero_norm_vectors_score_zero_not_nan(tmp_path, cfg, models_manifest, fingerprint):
    chunks = [make_chunk("degenerate", 1), make_chunk("real", 2)]
    vectors = np.stack([vec(), vec(1.0, 0.0, 0.0)])
    write_index(tmp_path / "degen", chunks=chunks, vectors=vectors, embedder=fingerprint)
    r = DenseRetriever(
        load_index(tmp_path / "degen"),
        cfg,
        FakeEncoder(vec(1.0, 0.0, 0.0)),
        models_manifest_path=models_manifest,
    )

    hits = r.retrieve("q", k=2)
    assert [h.chunk.chunk_id for h in hits] == ["real", "degenerate"]
    assert not any(np.isnan(h.score) for h in hits)


# --- the fingerprint: hard fail, never a quiet degrade (§3.1 rule 2) ------------------------


def test_mismatched_embedder_raises_rather_than_returning_results(
    tmp_path, cfg, models_manifest, fingerprint
):
    """The single most important property in this file.

    An index built by a different embedder must not return hits at all. Degrading quietly
    here looks exactly like a chunking bug and costs a day to find.
    """
    stale = {**fingerprint, "ir_sha256": "0" * 64}
    chunks = [make_chunk(cid, page) for cid, page, _ in FIVE_CHUNKS]
    vectors = np.stack([vec(*v) for _, _, v in FIVE_CHUNKS])
    write_index(tmp_path / "stale", chunks=chunks, vectors=vectors, embedder=stale)
    r = DenseRetriever(
        load_index(tmp_path / "stale"),
        cfg,
        FakeEncoder(vec(1.0, 0.0, 0.0)),
        models_manifest_path=models_manifest,
    )

    with pytest.raises(FingerprintMismatch, match="re-index required"):
        r.retrieve("photosynthesis", k=5)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hf_revision", "main"),
        ("precision", "int4"),
        ("dim", 768),
        ("pooling", "mean"),
        ("normalize", False),
        ("max_len", 512),
        ("query_prefix", "query: "),
        ("ov_version", "2025.1.0"),
    ],
)
def test_every_fingerprint_field_is_load_bearing(
    tmp_path, cfg, models_manifest, fingerprint, key, value
):
    """§3.1 lists eleven fields; a change to any one of them invalidates the index."""
    drifted = {**fingerprint, key: value}
    assert drifted != fingerprint, f"{key} already equals {value!r} — the case proves nothing"
    # A drifted `dim` also has to be a real width, or `load_index` catches it first.
    width = int(drifted["dim"])
    vectors = np.zeros((1, width), dtype=np.float32)
    vectors[0, 0] = 1.0
    write_index(
        tmp_path / f"drift-{key}",
        chunks=[make_chunk("c1", 1)],
        vectors=vectors,
        embedder=drifted,
        n_vectors=1,
    )
    r = DenseRetriever(
        load_index(tmp_path / f"drift-{key}"),
        cfg,
        FakeEncoder(vec(1.0)),
        models_manifest_path=models_manifest,
    )

    with pytest.raises(FingerprintMismatch):
        r.retrieve("q", k=1)


def test_an_index_without_a_fingerprint_is_refused(tmp_path, cfg, models_manifest):
    write_index(
        tmp_path / "nofp",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder={},
        n_vectors=1,
    )
    r = DenseRetriever(
        load_index(tmp_path / "nofp"),
        cfg,
        FakeEncoder(vec(1.0)),
        models_manifest_path=models_manifest,
    )

    with pytest.raises(FingerprintMismatch, match="no embedder fingerprint"):
        r.retrieve("q", k=1)


def test_the_fingerprint_is_rechecked_on_every_query(retriever):
    """§3.1 rule 2 says *on every query* — a check that only runs at load is not enough.

    A model swapped underneath a long-lived process is the realistic case: the UI holds one
    retriever open all session while the developer re-converts the embedder in another shell.
    """
    assert retriever.retrieve("photosynthesis", k=1)

    retriever.index.manifest["embedder"]["precision"] = "int4"

    with pytest.raises(FingerprintMismatch):
        retriever.retrieve("photosynthesis", k=1)


def test_no_hits_leak_out_of_a_mismatched_query(tmp_path, cfg, models_manifest, fingerprint):
    """The encoder must not even run once the fingerprint has failed."""
    write_index(
        tmp_path / "stale2",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder={**fingerprint, "hf_id": "sentence-transformers/all-MiniLM-L6-v2"},
        n_vectors=1,
    )
    encoder = FakeEncoder(vec(1.0))
    r = DenseRetriever(
        load_index(tmp_path / "stale2"), cfg, encoder, models_manifest_path=models_manifest
    )

    with pytest.raises(FingerprintMismatch):
        r.retrieve("q", k=1)
    assert encoder.calls == 0


# --- the query prefix travels with the index (§3.1 rule 3) ---------------------------------


def test_query_prefix_is_read_from_the_manifest_not_hardcoded(tmp_path, cfg):
    """BGE-M3 needs no prefix; an E5-family swap needs `query: `. The index decides."""
    prefixed = {**EMBEDDING, "query_prefix": "query: ", "passage_prefix": "passage: "}
    manifest_path = write_manifest(tmp_path / "e5", embedding=prefixed)
    fp = embedder_fingerprint(cfg, manifest_path=manifest_path)
    assert fp["query_prefix"] == "query: "

    write_index(
        tmp_path / "e5index",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder=fp,
        n_vectors=1,
    )
    encoder = FakeEncoder(vec(1.0))
    r = DenseRetriever(
        load_index(tmp_path / "e5index"), cfg, encoder, models_manifest_path=manifest_path
    )
    r.retrieve("what is photosynthesis", k=1)

    assert encoder.seen == ["query: what is photosynthesis"]


def test_bge_m3_gets_no_prefix(retriever):
    retriever.retrieve("what is photosynthesis", k=1)
    assert retriever.encode.seen == ["what is photosynthesis"]


# --- abstain (§4, §0.6) --------------------------------------------------------------------


def test_does_not_abstain_when_the_top_hit_clears_tau(retriever, cfg):
    hits = retriever.retrieve("photosynthesis", k=5)
    assert hits[0].score > cfg.retrieve.tau
    assert retriever.abstains(hits) is False


def test_abstains_when_every_hit_is_below_tau(index, cfg, models_manifest):
    """Query orthogonal to the whole corpus — cosine 0 everywhere, well under tau=0.35."""
    r = DenseRetriever(
        index, cfg, FakeEncoder(vec(0.0, 0.0, 1.0)), models_manifest_path=models_manifest
    )
    hits = r.retrieve("something the corpus never mentions", k=5)

    assert all(h.score == pytest.approx(0.0, abs=1e-6) for h in hits)
    assert r.abstains(hits) is True
    assert abstains(hits, cfg.retrieve.tau) is True


def test_the_retriever_delegates_to_the_shared_abstain_rule(retriever, cfg):
    hits = retriever.retrieve("photosynthesis", k=5)
    assert retriever.abstains(hits) == abstains(hits, cfg.retrieve.tau)


# --- an empty index ------------------------------------------------------------------------


def test_an_empty_index_returns_no_hits_and_abstains(tmp_path, cfg, models_manifest, fingerprint):
    write_index(
        tmp_path / "empty",
        chunks=[],
        vectors=np.zeros((0, DIM), dtype=np.float32),
        embedder=fingerprint,
    )
    idx = load_index(tmp_path / "empty")
    r = DenseRetriever(idx, cfg, FakeEncoder(vec(1.0)), models_manifest_path=models_manifest)

    assert idx.n_vectors == 0
    hits = r.retrieve("anything", k=20)
    assert hits == []
    assert r.abstains(hits) is True


# --- opening an index ----------------------------------------------------------------------


def test_vectors_are_memmapped_not_loaded(index):
    """§0.1 — a 500 MB index must not cost 500 MB of RAM to open."""
    assert isinstance(index.vectors, np.memmap)
    assert index.vectors.dtype == np.float32
    assert index.vectors.shape == (5, DIM)


def test_index_exposes_its_identity(index, fingerprint):
    assert isinstance(index, DenseIndex)
    assert index.index_id == "sha256:testindex"
    assert index.n_vectors == 5
    assert index.dim == DIM
    assert index.embedder == fingerprint
    assert index.query_prefix == ""


def test_chunks_decode_lazily_by_row(index):
    assert index.chunk_at(0).chunk_id == "c1"
    assert index.chunk_at(4).chunk_id == "c5"


def test_missing_index_directory_is_named(tmp_path):
    with pytest.raises(IndexNotFound, match="index directory not found"):
        load_index(tmp_path / "nothing-here")


@pytest.mark.parametrize("missing", [CHUNKS_NAME, VECTORS_NAME, INDEX_MANIFEST_NAME])
def test_an_incomplete_index_is_refused(tmp_path, fingerprint, missing):
    write_index(
        tmp_path / "partial",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder=fingerprint,
        n_vectors=1,
        omit=(missing,),
    )
    with pytest.raises(IndexNotFound, match=missing):
        load_index(tmp_path / "partial")


def test_a_corrupt_index_manifest_is_refused(tmp_path, fingerprint):
    write_index(
        tmp_path / "badjson",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder=fingerprint,
        manifest_text="{not json",
    )
    with pytest.raises(IndexCorrupt, match="not valid JSON"):
        load_index(tmp_path / "badjson")


def test_row_count_mismatch_is_refused(tmp_path, fingerprint):
    """Vectors and chunks join by row position — a mismatch mis-cites every answer."""
    write_index(
        tmp_path / "skew",
        chunks=[make_chunk("c1", 1), make_chunk("c2", 2)],
        vectors=np.stack([vec(1.0)]),
        embedder=fingerprint,
    )
    with pytest.raises(IndexCorrupt, match="2 chunks but 1 vectors"):
        load_index(tmp_path / "skew")


def test_declared_n_vectors_must_match_the_array(tmp_path, fingerprint):
    write_index(
        tmp_path / "lying",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder=fingerprint,
        n_vectors=99,
    )
    with pytest.raises(IndexCorrupt, match="n_vectors=99"):
        load_index(tmp_path / "lying")


def test_declared_dim_must_match_the_array(tmp_path, fingerprint):
    write_index(
        tmp_path / "narrow",
        chunks=[make_chunk("c1", 1)],
        vectors=np.zeros((1, 8), dtype=np.float32),
        embedder=fingerprint,
        n_vectors=1,
    )
    with pytest.raises(IndexCorrupt, match="8-wide"):
        load_index(tmp_path / "narrow")


def test_non_float32_vectors_are_refused(tmp_path, fingerprint):
    write_index(
        tmp_path / "f64",
        chunks=[make_chunk("c1", 1)],
        vectors=np.zeros((1, DIM), dtype=np.float64),
        embedder=fingerprint,
        n_vectors=1,
    )
    with pytest.raises(IndexCorrupt, match="float32"):
        load_index(tmp_path / "f64")


def test_one_dimensional_vectors_are_refused(tmp_path, fingerprint):
    write_index(
        tmp_path / "flat",
        chunks=[make_chunk("c1", 1)],
        vectors=np.zeros(DIM, dtype=np.float32),
        embedder=fingerprint,
        n_vectors=1,
    )
    with pytest.raises(IndexCorrupt, match="expected 2-D"):
        load_index(tmp_path / "flat")


def test_chunks_missing_provenance_columns_are_refused(tmp_path, fingerprint):
    root = tmp_path / "noprov"
    root.mkdir(parents=True)
    pl.DataFrame({"chunk_id": ["c1"], "text": ["hello"]}).write_parquet(root / CHUNKS_NAME)
    np.save(root / VECTORS_NAME, np.stack([vec(1.0)]))
    (root / INDEX_MANIFEST_NAME).write_text(json.dumps({"embedder": fingerprint}))

    with pytest.raises(IndexCorrupt, match="missing Chunk columns"):
        load_index(root)


# --- default_index_dir ---------------------------------------------------------------------


def test_default_index_dir_finds_the_only_index(cfg, fingerprint):
    root = cfg.resolve(cfg.paths.data_dir) / "index"
    write_index(
        root / "abc123",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder=fingerprint,
        n_vectors=1,
    )
    assert default_index_dir(cfg) == root / "abc123"


def test_default_index_dir_refuses_to_guess_between_several(cfg, fingerprint):
    root = cfg.resolve(cfg.paths.data_dir) / "index"
    for name in ("abc123", "def456"):
        write_index(
            root / name,
            chunks=[make_chunk("c1", 1)],
            vectors=np.stack([vec(1.0)]),
            embedder=fingerprint,
            n_vectors=1,
        )
    with pytest.raises(IndexNotFound, match="several indexes"):
        default_index_dir(cfg)


def test_default_index_dir_without_an_index_root(cfg):
    with pytest.raises(IndexNotFound, match="no index root"):
        default_index_dir(cfg)


def test_default_index_dir_with_an_empty_index_root(cfg):
    (cfg.resolve(cfg.paths.data_dir) / "index").mkdir(parents=True)
    with pytest.raises(IndexNotFound, match="contains no index"):
        default_index_dir(cfg)


# --- malformed calls -----------------------------------------------------------------------


def test_k_must_be_positive(retriever):
    with pytest.raises(ValueError, match="k must be >= 1"):
        retriever.retrieve("photosynthesis", k=0)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_is_refused(retriever, query):
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.retrieve(query, k=5)


def test_block_rows_must_be_positive(index, cfg):
    with pytest.raises(ValueError, match="block_rows"):
        DenseRetriever(index, cfg, FakeEncoder(vec(1.0)), block_rows=0)


def test_a_wrong_width_query_embedding_is_refused(index, cfg, models_manifest):
    r = DenseRetriever(
        index,
        cfg,
        FakeEncoder(np.ones(8, dtype=np.float32)),
        models_manifest_path=models_manifest,
    )
    with pytest.raises(DenseIndexError, match="different model"):
        r.retrieve("q", k=1)


def test_a_batched_query_embedding_is_refused(index, cfg, models_manifest):
    class BatchEncoder:
        def __call__(self, texts):
            return np.zeros((3, DIM), dtype=np.float32)

    r = DenseRetriever(index, cfg, BatchEncoder(), models_manifest_path=models_manifest)
    with pytest.raises(DenseIndexError, match="3 vectors for one query"):
        r.retrieve("q", k=1)


def test_a_bare_vector_from_the_encoder_is_accepted(index, cfg, models_manifest):
    """`[dim]` and `[1, dim]` are both reasonable encoder outputs; accept either."""

    class BareEncoder:
        def __call__(self, texts):
            return vec(1.0, 0.0, 0.0)

    r = DenseRetriever(index, cfg, BareEncoder(), models_manifest_path=models_manifest)
    assert [h.chunk.chunk_id for h in r.retrieve("q", k=2)] == ["c2", "c4"]


# --- it plugs into everything downstream ---------------------------------------------------


def test_satisfies_the_retriever_protocol(retriever):
    assert isinstance(retriever, Retriever)


def test_drives_the_eval_harness(retriever, cfg):
    """§5 — the harness was built against a random stub; the real thing must slot in."""
    golden = [
        GoldQuestion(q="what is photosynthesis", lang="en", doc_id="doc-bio", gold_pages=[12]),
        GoldQuestion(q="what is respiration", lang="en", doc_id="doc-bio", gold_pages=[14]),
    ]
    result, frame = evaluate(retriever, golden, cfg, label="dense")

    assert result.n_questions == 2
    assert result.recall_at_1 == pytest.approx(0.5)  # page 12 is rank 1, page 14 is rank 2
    assert result.recall_at_5 == pytest.approx(1.0)
    assert result.mrr_at_10 == pytest.approx((1.0 + 0.5) / 2)
    assert frame.height == 2


def test_dense_hits_pass_through_fusion_with_their_scores_intact(retriever, cfg):
    """§1 — with only dense wired, fusion must not disturb the abstain decision."""
    hits = retriever.retrieve("photosynthesis", k=5)
    fused = fuse([RankedList(name="dense", hits=hits)], k=cfg.retrieve.n_context)

    assert fused.scores_are_similarities is True
    assert [h.chunk.chunk_id for h in fused.hits] == EXPECTED_ORDER[: cfg.retrieve.n_context]
    assert abstains(fused.hits, cfg.retrieve.tau) is False


# --- production wiring onto ingest/embed.py -------------------------------------------------


class FakeEmbedder:
    """The `ingest.embed.Embedder` surface `retrieve/dense.py` actually uses."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls: list[tuple[list[str], str]] = []

    def embed_texts(self, texts, *, prefix: str = "") -> np.ndarray:
        self.calls.append((list(texts), prefix))
        return np.asarray([self.vector for _ in texts], dtype=np.float32)


def test_query_encoder_leaves_prefixing_to_the_index_manifest(cfg):
    """`Embedder.embed_queries` would apply the *models* manifest prefix — that is one too many.

    `DenseRetriever` has already applied the index's `query_prefix`, so the encoder must be
    handed the text verbatim or an E5-family index gets `query: query: ...`.
    """
    embedder = FakeEmbedder(vec(1.0))
    encode = query_encoder(cfg, embedder=embedder)
    out = encode(["query: what is photosynthesis"])

    assert embedder.calls == [(["query: what is photosynthesis"], "")]
    assert out.shape == (1, DIM)


def test_open_wires_an_index_and_an_embedder_together(tmp_path, cfg, models_manifest, fingerprint):
    chunks = [make_chunk(cid, page) for cid, page, _ in FIVE_CHUNKS]
    vectors = np.stack([vec(*v) for _, _, v in FIVE_CHUNKS])
    write_index(tmp_path / "wired", chunks=chunks, vectors=vectors, embedder=fingerprint)

    r = DenseRetriever.open(
        cfg,
        tmp_path / "wired",
        models_manifest_path=models_manifest,
        embedder=FakeEmbedder(vec(1.0, 0.0, 0.0)),
    )

    assert [h.chunk.chunk_id for h in r.retrieve("photosynthesis", k=5)] == EXPECTED_ORDER


def test_open_resolves_the_only_index_under_data_dir(cfg, models_manifest, fingerprint):
    root = cfg.resolve(cfg.paths.data_dir) / "index"
    chunks = [make_chunk(cid, page) for cid, page, _ in FIVE_CHUNKS]
    vectors = np.stack([vec(*v) for _, _, v in FIVE_CHUNKS])
    write_index(root / "abc123", chunks=chunks, vectors=vectors, embedder=fingerprint)

    r = DenseRetriever.open(
        cfg, models_manifest_path=models_manifest, embedder=FakeEmbedder(vec(1.0, 0.0, 0.0))
    )

    assert r.index.path == root / "abc123"
    assert r.retrieve("photosynthesis", k=1)[0].chunk.chunk_id == "c2"


def test_open_with_a_prefixed_index_applies_the_prefix_exactly_once(tmp_path, cfg):
    prefixed = {**EMBEDDING, "query_prefix": "query: ", "passage_prefix": "passage: "}
    manifest_path = write_manifest(tmp_path / "e5models", embedding=prefixed)
    fp = embedder_fingerprint(cfg, manifest_path=manifest_path)
    write_index(
        tmp_path / "e5wired",
        chunks=[make_chunk("c1", 1)],
        vectors=np.stack([vec(1.0)]),
        embedder=fp,
        n_vectors=1,
    )
    embedder = FakeEmbedder(vec(1.0))

    r = DenseRetriever.open(
        cfg, tmp_path / "e5wired", models_manifest_path=manifest_path, embedder=embedder
    )
    r.retrieve("what is photosynthesis", k=1)

    assert embedder.calls == [(["query: what is photosynthesis"], "")]
