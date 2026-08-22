"""Index writer — ARCHITECTURE.md §3 `index`, §3.1, §0.1 storage layout.

The highest-value test in this file is `test_an_index_written_here_is_readable_by_dense`:
`ingest/index.py` is the only writer of a format `retrieve/dense.py` is the only reader of,
and a disagreement between them is not a crash, it is every citation pointing at the wrong
page. So the two are pinned together here — constants, layout, and a real round trip.

Vectors are hand-written so the expected ranking is arithmetic, and the embedder fingerprint
comes from the stub IR `tests/test_registry.py` builds: the fingerprint is a hash of weights
plus a JSON block, and neither cares that the weights are a stub. No test touches `data/`.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk
from ingest.index import (
    CHUNKS_NAME,
    INDEX_MANIFEST_NAME,
    STAGE_VERSION,
    VECTORS_NAME,
    BuiltIndex,
    IndexBuildError,
    build_index,
    index_dir_name,
    index_id_for,
    index_path,
    index_root,
)
from models.registry import EMBEDDER_FINGERPRINT_KEYS, FingerprintMismatch, embedder_fingerprint
from retrieve import dense
from retrieve.dense import DenseRetriever, load_index
from tests.test_registry import EMBEDDING, write_manifest

DIM = EMBEDDING["dim"]

#: (chunk_id, page, vector). Against query (1, 0, 0) the collinear row wins on angle even
#: though it does not have the largest raw dot product.
THREE_CHUNKS = [
    ("c1", 11, (1.0, 3.0, 0.0)),
    ("c2", 12, (1.0, 0.0, 0.0)),
    ("c3", 13, (3.0, 1.0, 0.0)),
]
EXPECTED_ORDER = ["c2", "c3", "c1"]


def vec(*values: float) -> np.ndarray:
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


class FakeEncoder:
    """Stands in for `ingest/embed.py` on the query side."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def __call__(self, texts):
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
    return embedder_fingerprint(cfg, manifest_path=models_manifest)


@pytest.fixture
def chunks():
    return [make_chunk(cid, page) for cid, page, _ in THREE_CHUNKS]


@pytest.fixture
def vectors():
    return np.stack([vec(*v) for _, _, v in THREE_CHUNKS])


@pytest.fixture
def built(tmp_path, cfg, chunks, vectors, fingerprint) -> BuiltIndex:
    return build_index(chunks, vectors, cfg, root=tmp_path / "index", fingerprint=fingerprint)


# --- the layout on disk (§0.1) --------------------------------------------------------------


def test_the_three_artifacts_land_where_the_architecture_says(built):
    assert built.chunks_path.is_file()
    assert built.vectors_path.is_file()
    assert built.manifest_path.is_file()
    assert built.path.name == index_dir_name(built.index_id)


def test_filenames_match_the_reader_that_opens_them(tmp_path):
    """`ingest/index.py` writes what `retrieve/dense.py` reads. A rename on either side is
    a broken index, not a broken import, so it is asserted rather than shared."""
    assert (CHUNKS_NAME, VECTORS_NAME, INDEX_MANIFEST_NAME) == (
        dense.CHUNKS_NAME,
        dense.VECTORS_NAME,
        dense.INDEX_MANIFEST_NAME,
    )


def test_index_root_follows_the_configured_data_dir(cfg, tmp_path):
    assert index_root(cfg) == tmp_path / "data" / "index"


def test_vectors_round_trip_as_a_float32_memmap(built, vectors):
    stored = np.load(built.vectors_path, mmap_mode="r")
    assert stored.dtype == np.float32
    assert stored.shape == (len(THREE_CHUNKS), DIM)
    assert isinstance(stored, np.memmap)
    np.testing.assert_array_equal(np.asarray(stored), vectors.astype(np.float32))


def test_vectors_are_npy_not_parquet(built):
    """§0.1 pins `.npy` — a parquet round trip costs RAM this machine does not have."""
    assert built.vectors_path.read_bytes()[:6] == b"\x93NUMPY"


def test_chunk_rows_keep_their_order_and_their_provenance(built, chunks):
    frame = pl.read_parquet(built.chunks_path)
    assert frame["chunk_id"].to_list() == [c.chunk_id for c in chunks]
    first = Chunk.from_row(frame.row(0, named=True))
    assert first == chunks[0]
    assert first.bbox_union == (0.1, 0.2, 0.9, 0.8)


def test_row_i_of_the_vectors_is_row_i_of_the_chunks(built, vectors):
    frame = pl.read_parquet(built.chunks_path)
    stored = np.load(built.vectors_path, mmap_mode="r")
    assert frame.height == stored.shape[0]
    for row, (chunk_id, _, values) in enumerate(THREE_CHUNKS):
        assert frame["chunk_id"][row] == chunk_id
        np.testing.assert_array_equal(np.asarray(stored[row]), vec(*values))


# --- the manifest (§3.1 — MANDATORY) --------------------------------------------------------


def test_the_manifest_carries_the_full_embedder_fingerprint(built, fingerprint):
    manifest = json.loads(built.manifest_path.read_text())
    assert manifest["embedder"] == fingerprint
    for key in EMBEDDER_FINGERPRINT_KEYS:
        assert key in manifest["embedder"], f"§3.1 requires {key}"


def test_the_manifest_carries_the_index_identity(built, cfg, chunks):
    manifest = json.loads(built.manifest_path.read_text())
    assert manifest["index_id"] == built.index_id
    assert manifest["index_id"].startswith("sha256:")
    assert manifest["n_vectors"] == len(chunks)
    assert manifest["chunk_config_hash"] == cfg.chunk_config_hash
    assert manifest["stage_version"] == STAGE_VERSION


def test_the_manifest_is_written_last_so_a_partial_build_is_refused(built):
    """`load_index` treats the manifest as proof of completeness (§0.1)."""
    built.manifest_path.unlink()
    with pytest.raises(dense.IndexNotFound, match="not a complete index"):
        load_index(built.path)


def test_prefixes_travel_with_the_index_not_the_code(built):
    """§3.1 rule 3 — an E5-style model's asymmetric prefixes must survive in the manifest."""
    manifest = json.loads(built.manifest_path.read_text())
    assert "query_prefix" in manifest["embedder"]
    assert "passage_prefix" in manifest["embedder"]


# --- the end-to-end contract: what this writes, dense.py opens ------------------------------


def test_an_index_written_here_is_readable_by_dense(built, chunks):
    index = load_index(built.path)
    assert index.n_vectors == len(chunks)
    assert index.dim == DIM
    assert index.index_id == built.index_id
    assert index.chunk_at(0) == chunks[0]


def test_a_query_against_a_written_index_ranks_by_cosine(built, cfg, models_manifest):
    retriever = DenseRetriever(
        load_index(built.path),
        cfg,
        FakeEncoder(vec(1.0, 0.0, 0.0)),
        models_manifest_path=models_manifest,
    )
    hits = retriever.retrieve("photosynthesis", k=3)
    assert [h.chunk.chunk_id for h in hits] == EXPECTED_ORDER
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].chunk.page_start == 12  # provenance survived the round trip


def test_a_mismatched_embedder_raises_instead_of_returning_results(
    tmp_path, cfg, chunks, vectors, fingerprint, models_manifest
):
    """§3.1 rule 2 — drift must hard-fail, never degrade quietly."""
    stale = dict(fingerprint) | {"ir_sha256": "sha256:some-other-quantization"}
    built = build_index(chunks, vectors, cfg, root=tmp_path / "stale", fingerprint=stale)
    retriever = DenseRetriever(
        load_index(built.path),
        cfg,
        FakeEncoder(vec(1.0, 0.0, 0.0)),
        models_manifest_path=models_manifest,
    )
    with pytest.raises(FingerprintMismatch, match="re-index required"):
        retriever.retrieve("photosynthesis", k=3)


# --- refusals: every way an index can be silently wrong -------------------------------------


def test_a_vector_per_chunk_mismatch_is_refused(cfg, chunks, fingerprint, tmp_path):
    with pytest.raises(IndexBuildError, match="every citation is wrong"):
        build_index(
            chunks,
            np.stack([vec(1.0), vec(2.0)]),
            cfg,
            root=tmp_path / "index",
            fingerprint=fingerprint,
        )


def test_a_width_that_contradicts_the_fingerprint_is_refused(cfg, chunks, fingerprint, tmp_path):
    narrow = np.ones((len(chunks), 8), dtype=np.float32)
    with pytest.raises(IndexBuildError, match="did not come from the manifest's embedder"):
        build_index(chunks, narrow, cfg, root=tmp_path / "index", fingerprint=fingerprint)


def test_a_one_dimensional_array_is_refused(cfg, fingerprint, tmp_path):
    with pytest.raises(IndexBuildError, match="expected 2-D"):
        build_index(
            [make_chunk("c1", 1)],
            vec(1.0),
            cfg,
            root=tmp_path / "index",
            fingerprint=fingerprint,
        )


def test_non_finite_vectors_are_refused(cfg, chunks, vectors, fingerprint, tmp_path):
    poisoned = vectors.copy()
    poisoned[1, 0] = np.nan
    with pytest.raises(IndexBuildError, match="NaN or inf"):
        build_index(chunks, poisoned, cfg, root=tmp_path / "index", fingerprint=fingerprint)


def test_duplicate_chunk_ids_are_refused(cfg, fingerprint, tmp_path):
    twice = [make_chunk("c1", 1), make_chunk("c1", 1)]
    with pytest.raises(IndexBuildError, match="duplicate chunk_id"):
        build_index(
            twice,
            np.stack([vec(1.0), vec(1.0)]),
            cfg,
            root=tmp_path / "index",
            fingerprint=fingerprint,
        )


def test_nothing_is_written_when_the_build_is_refused(cfg, chunks, fingerprint, tmp_path):
    root = tmp_path / "index"
    with pytest.raises(IndexBuildError):
        build_index(chunks, np.zeros((1, DIM), np.float32), cfg, root=root, fingerprint=fingerprint)
    assert not root.exists()


# --- empty and idempotent -------------------------------------------------------------------


def test_an_empty_corpus_writes_a_readable_but_empty_index(cfg, fingerprint, tmp_path):
    built = build_index(
        [], np.zeros((0, DIM), np.float32), cfg, root=tmp_path, fingerprint=fingerprint
    )
    index = load_index(built.path)
    assert index.n_vectors == 0
    assert set(pl.read_parquet(built.chunks_path).columns) >= {"chunk_id", "doc_id", "bbox_union"}


def test_the_same_corpus_lands_on_the_same_index_id(cfg, chunks, vectors, fingerprint, tmp_path):
    a = build_index(chunks, vectors, cfg, root=tmp_path / "a", fingerprint=fingerprint)
    b = build_index(chunks, vectors, cfg, root=tmp_path / "b", fingerprint=fingerprint)
    assert a.index_id == b.index_id


def test_a_different_embedder_is_a_different_index(cfg, chunks, fingerprint):
    other = dict(fingerprint) | {"ir_sha256": "sha256:different"}
    assert index_id_for(chunks, fingerprint, "h") != index_id_for(chunks, other, "h")


def test_a_different_chunk_config_is_a_different_index(chunks, fingerprint):
    assert index_id_for(chunks, fingerprint, "h1") != index_id_for(chunks, fingerprint, "h2")


def test_rebuilding_an_unchanged_index_rewrites_nothing(
    cfg, chunks, vectors, fingerprint, tmp_path
):
    first = build_index(chunks, vectors, cfg, root=tmp_path / "index", fingerprint=fingerprint)
    stamp = first.manifest_path.stat().st_mtime_ns
    again = build_index(chunks, vectors, cfg, root=tmp_path / "index", fingerprint=fingerprint)
    assert again.reused is True
    assert again.path == first.path
    assert again.manifest_path.stat().st_mtime_ns == stamp


def test_an_interrupted_build_is_rewritten_not_reused(cfg, chunks, vectors, fingerprint, tmp_path):
    first = build_index(chunks, vectors, cfg, root=tmp_path / "index", fingerprint=fingerprint)
    first.manifest_path.unlink()
    again = build_index(chunks, vectors, cfg, root=tmp_path / "index", fingerprint=fingerprint)
    assert again.reused is False
    assert load_index(again.path).n_vectors == len(chunks)


def test_a_manifest_from_another_embedder_is_never_reused(
    cfg, chunks, vectors, fingerprint, tmp_path
):
    root = tmp_path / "index"
    built = build_index(chunks, vectors, cfg, root=root, fingerprint=fingerprint)
    poisoned = json.loads(built.manifest_path.read_text())
    poisoned["embedder"]["ir_sha256"] = "sha256:someone-elses-weights"
    built.manifest_path.write_text(json.dumps(poisoned))

    again = build_index(chunks, vectors, cfg, root=root, fingerprint=fingerprint)
    assert again.reused is False
    assert json.loads(again.manifest_path.read_text())["embedder"] == fingerprint


def test_a_corrupt_manifest_is_rewritten_rather_than_trusted(
    cfg, chunks, vectors, fingerprint, tmp_path
):
    root = tmp_path / "index"
    built = build_index(chunks, vectors, cfg, root=root, fingerprint=fingerprint)
    built.manifest_path.write_text("{not json")
    again = build_index(chunks, vectors, cfg, root=root, fingerprint=fingerprint)
    assert again.reused is False
    assert load_index(again.path).index_id == built.index_id


def test_the_default_root_is_the_configured_data_dir(cfg, chunks, vectors, fingerprint):
    built = build_index(chunks, vectors, cfg, fingerprint=fingerprint)
    assert built.path == index_path(index_root(cfg), built.index_id)
    assert built.path.is_dir()
