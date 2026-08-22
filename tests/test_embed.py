"""Embedding stage — ARCHITECTURE.md §3 `embed`, §3.1, §7.5.

Everything runs against a **stub encoder IR** built in `tmp_path`: an embedding-table gather
that maps `[batch, seq]` int64 to `[batch, seq, dim]` float32, the same shape contract as
BGE-M3. The real INT8 IR is 544 MB and is not committed (§7.3), so a test that needed it
would be a test that never runs on a clean checkout. Set `INTEL2026_REAL_MODELS=1` to also
run the one test that does use it.

Nothing here asserts on specific embedding values — models drift (CLAUDE.md). The assertions
are on shape, dtype, normalisation, row alignment, prefix provenance, caching and failure.

No test may touch `data/`; caches are redirected into `tmp_path`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from core.cache import CacheKey, StageCache
from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk
from ingest.embed import (
    DEFAULT_BATCH_SIZE,
    STAGE_VERSION,
    Embedder,
    EmbedError,
    TokenizerUnavailable,
    VectorCache,
    _l2_normalize,
    embed_chunks,
    embed_config_hash,
    hash_texts,
)
from models.registry import IR_XML_NAME, MANIFEST_SCHEMA_VERSION, ov_version

STUB_DIM = 8
STUB_VOCAB = 97

TEXTS = [
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "ஒளிச்சேர்க்கை என்பது ஒளி ஆற்றலை வேதி ஆற்றலாக மாற்றும் செயல்முறை ஆகும்.",
    "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖的过程。",
    "The Calvin cycle fixes carbon dioxide in the stroma of the chloroplast.",
    "Chlorophyll absorbs red and blue light and reflects green light.",
]


def make_stub_encoder(ir_dir: Path, *, dim: int = STUB_DIM, seed: int = 0) -> Path:
    """`(input_ids, attention_mask) -> last_hidden_state`, the BGE-M3 I/O contract in miniature.

    A gather from a fixed table, masked like a real encoder attends: distinct tokens give
    distinct hidden states, so pooled vectors actually differ between texts.
    """
    ir_dir.mkdir(parents=True, exist_ok=True)
    table = np.random.default_rng(seed).standard_normal((STUB_VOCAB, dim)).astype(np.float32)

    ids = ops.parameter([-1, -1], ov.Type.i64, name="input_ids")
    mask = ops.parameter([-1, -1], ov.Type.i64, name="attention_mask")
    hidden = ops.gather(ops.constant(table), ids, ops.constant(0, ov.Type.i32))
    keep = ops.unsqueeze(ops.convert(mask, ov.Type.f32), ops.constant([-1], ov.Type.i32))
    out = ops.multiply(hidden, keep)
    out.output(0).get_tensor().set_names({"last_hidden_state"})

    model = ov.Model([out], [ids, mask], "stub-encoder")
    ov.save_model(model, ir_dir / IR_XML_NAME, compress_to_fp16=False)
    return ir_dir


class StubTokenizer:
    """Deterministic, offline, and counts its calls so cache tests can prove a miss.

    Position 0 stands in for `<s>` but carries a hash of the text's opening, so CLS pooling
    produces a different vector per text — a fixed CLS id would make every stub vector
    identical. Hashing the *opening* rather than the whole string keeps two texts that share
    a prefix identical up to the point where they diverge, which is what makes the truncation
    test mean anything.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, texts):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        seqs = [[self._head(t), *[ord(c) % STUB_VOCAB for c in t]] for t in texts]
        width = max(len(s) for s in seqs)
        ids = np.ones((len(seqs), width), dtype=np.int64)
        mask = np.zeros((len(seqs), width), dtype=np.int64)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = seq
            mask[i, : len(seq)] = 1
        return ids, mask

    @staticmethod
    def _head(text: str) -> int:
        return int(hashlib.sha256(text[:4].encode()).hexdigest()[:8], 16) % STUB_VOCAB


def write_manifest(
    root: Path,
    *,
    dim: int = STUB_DIM,
    pooling: str = "cls",
    normalize: bool = True,
    max_len: int = 512,
    query_prefix: str = "",
    passage_prefix: str = "",
    build_ir: bool = True,
    seed: int = 0,
    ir_dir: str = "ir/stub-int8",
) -> Path:
    if build_ir:
        make_stub_encoder(root / ir_dir, dim=dim if dim <= STUB_DIM else STUB_DIM, seed=seed)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "ov_version": ov_version(),
        "models": {
            "bge-m3": {
                "role": "embedder",
                "hf_id": "BAAI/bge-m3",
                "hf_revision": "5617a9f61b028005a4858fdac845db406aefb181",
                "precision": "int8",
                "ir_dir": ir_dir,
                "ir_sha256": "",
                "ov_version": ov_version(),
                "converted_at": "2026-08-22T00:00:00Z",
                "embedding": {
                    "dim": dim,
                    "pooling": pooling,
                    "normalize": normalize,
                    "max_len": max_len,
                    "query_prefix": query_prefix,
                    "passage_prefix": passage_prefix,
                },
            }
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=tmp_path / "ov_cache")
        }
    )


@pytest.fixture
def tokenizer():
    return StubTokenizer()


def make_embedder(tmp_path, cfg, tokenizer, **kw) -> Embedder:
    manifest = write_manifest(tmp_path, **kw)
    return Embedder.load(cfg, manifest_path=manifest, tokenizer=tokenizer)


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        doc_id="doc",
        page_start=1,
        page_end=1,
        block_ids=[f"b{i}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 1"],
        text=text,
        token_count=len(text.split()),
        lang="en",
        script="latn",
    )


CHUNKS = [make_chunk(i, t) for i, t in enumerate(TEXTS)]


# --- shape, dtype, normalisation -----------------------------------------------------


def test_embed_passages_returns_float32_rows_of_manifest_dim(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    vectors = emb.embed_passages(TEXTS)
    assert vectors.shape == (len(TEXTS), STUB_DIM)
    assert vectors.dtype == np.float32
    assert emb.dim == STUB_DIM


def test_vectors_are_l2_normalised_when_the_manifest_says_so(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, normalize=True)
    norms = np.linalg.norm(emb.embed_passages(TEXTS), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_normalisation_is_not_hardcoded(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, normalize=False)
    norms = np.linalg.norm(emb.embed_passages(TEXTS), axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_distinct_texts_get_distinct_vectors(tmp_path, cfg, tokenizer):
    vectors = make_embedder(tmp_path, cfg, tokenizer).embed_passages(TEXTS)
    assert len({v.tobytes() for v in vectors}) == len(TEXTS)


def test_empty_input_returns_an_empty_matrix_of_the_right_width(tmp_path, cfg, tokenizer):
    vectors = make_embedder(tmp_path, cfg, tokenizer).embed_passages([])
    assert vectors.shape == (0, STUB_DIM)
    assert vectors.dtype == np.float32


def test_a_zero_vector_normalises_to_zero_not_nan():
    out = _l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert not np.isnan(out).any()
    assert np.allclose(out[0], 0.0)
    assert np.allclose(np.linalg.norm(out[1]), 1.0)


# --- batching (§7.5) -----------------------------------------------------------------


def test_batching_produces_the_same_vectors_as_a_single_batch(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    one_batch = emb.embed_passages(TEXTS, batch_size=len(TEXTS))
    small = emb.embed_passages(TEXTS, batch_size=2)
    assert np.allclose(one_batch, small, atol=1e-6)


def test_texts_are_batched_not_embedded_one_at_a_time(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    emb.embed_passages(TEXTS, batch_size=DEFAULT_BATCH_SIZE)
    assert tokenizer.calls == 1
    assert tokenizer.batch_sizes == [len(TEXTS)]


def test_batch_size_is_respected(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    emb.embed_passages(TEXTS, batch_size=2)
    assert tokenizer.batch_sizes == [2, 2, 1]


def test_default_batch_size_is_in_the_sanctioned_range():
    assert 16 <= DEFAULT_BATCH_SIZE <= 32


def test_a_nonsense_batch_size_raises(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    with pytest.raises(ValueError, match="batch_size"):
        emb.embed_passages(TEXTS, batch_size=0)


def test_padding_does_not_change_a_short_text_under_mean_pooling(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, pooling="mean")
    alone = emb.embed_passages([TEXTS[0]])
    padded = emb.embed_passages([TEXTS[0], "x" * 400])
    assert np.allclose(alone[0], padded[0], atol=1e-6)


# --- the fingerprint decides the maths (§3.1 rule 3) ----------------------------------


def test_prefixes_come_from_the_manifest_not_from_code(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, query_prefix="query: ", passage_prefix="doc: ")
    assert emb.spec.query_prefix == "query: "
    as_query = emb.embed_queries([TEXTS[0]])
    as_passage = emb.embed_passages([TEXTS[0]])
    assert not np.allclose(as_query, as_passage)
    assert np.allclose(as_query, emb.embed_texts(["query: " + TEXTS[0]]), atol=1e-6)


def test_no_prefix_means_no_prefix(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    assert np.allclose(emb.embed_queries([TEXTS[0]]), emb.embed_passages([TEXTS[0]]), atol=1e-6)


def test_pooling_comes_from_the_manifest(tmp_path, cfg, tokenizer):
    cls_vectors = make_embedder(tmp_path / "a", cfg, tokenizer, pooling="cls").embed_passages(TEXTS)
    mean_vectors = make_embedder(tmp_path / "b", cfg, tokenizer, pooling="mean").embed_passages(
        TEXTS
    )
    assert not np.allclose(cls_vectors, mean_vectors)


def test_an_unsupported_pooling_raises_rather_than_guessing(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, pooling="max")
    with pytest.raises(EmbedError, match="pooling"):
        emb.embed_passages(TEXTS)


def test_a_model_that_disagrees_with_its_manifest_dim_raises(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, dim=STUB_DIM * 2)
    with pytest.raises(EmbedError, match="dim"):
        emb.embed_passages(TEXTS)


def test_text_beyond_max_len_is_truncated_not_silently_wrong(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, pooling="mean", max_len=6)
    shared, diverged = "abcde" + "f" * 60, "abcde" + "z" * 60
    vectors = emb.embed_passages([shared, diverged])
    assert np.allclose(vectors[0], vectors[1], atol=1e-6)


def test_the_fingerprint_travels_with_the_embedder(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer, query_prefix="query: ")
    assert emb.fingerprint["query_prefix"] == "query: "
    assert emb.fingerprint["dim"] == STUB_DIM
    assert emb.fingerprint["ir_sha256"]


def test_a_missing_tokenizer_ir_names_the_setup_script(tmp_path, cfg):
    manifest = write_manifest(tmp_path)
    with pytest.raises(TokenizerUnavailable, match="scripts.setup"):
        Embedder.load(cfg, manifest_path=manifest)


# --- chunk-level API, row alignment, caching (§0 non-negotiable 1, §0.1) --------------


def test_embed_chunks_is_row_aligned_with_its_input(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    vectors = embed_chunks(CHUNKS, cfg, embedder=emb)
    assert vectors.shape == (len(CHUNKS), STUB_DIM)
    for i, chunk in enumerate(CHUNKS):
        assert np.allclose(vectors[i], emb.embed_passages([chunk.text])[0], atol=1e-6)


def test_embed_chunks_of_nothing_is_an_empty_matrix(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    assert embed_chunks([], cfg, embedder=emb).shape == (0, STUB_DIM)


def test_vectors_are_cached_as_npy_not_parquet(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    cache = StageCache(tmp_path / "cache")
    embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)

    written = list((tmp_path / "cache" / "embed").iterdir())
    assert [p.suffix for p in written] == [".npy"]
    assert not list((tmp_path / "cache" / "embed").glob("*.parquet"))


def test_cached_vectors_memmap_back_as_float32(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    cache = StageCache(tmp_path / "cache")
    vectors = embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)

    path = next((tmp_path / "cache" / "embed").glob("*.npy"))
    loaded = np.load(path, mmap_mode="r")
    assert loaded.shape == vectors.shape
    assert loaded.dtype == np.float32
    assert np.array_equal(np.asarray(loaded), vectors)


def test_a_second_run_hits_the_cache_instead_of_the_model(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    cache = StageCache(tmp_path / "cache")
    first = embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)
    calls_after_first = tokenizer.calls

    second = embed_chunks(CHUNKS, cfg, embedder=emb, cache=StageCache(tmp_path / "cache"))
    assert tokenizer.calls == calls_after_first
    assert np.array_equal(first, second)


def test_a_cache_hit_does_not_need_a_loaded_model(tmp_path, cfg, tokenizer):
    """§0.1: the key is the fingerprint, and the fingerprint is a hash of bytes on disk."""
    manifest = write_manifest(tmp_path)
    emb = Embedder.load(cfg, manifest_path=manifest, tokenizer=tokenizer)
    cache = StageCache(tmp_path / "cache")
    first = embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache, manifest_path=manifest)

    again = embed_chunks(CHUNKS, cfg, cache=cache, manifest_path=manifest)
    assert np.array_equal(first, again)


def test_changed_text_is_a_cache_miss(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    cache = StageCache(tmp_path / "cache")
    embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)
    before = tokenizer.calls

    edited = [*CHUNKS[:-1], make_chunk(99, "a different sentence entirely")]
    embed_chunks(edited, cfg, embedder=emb, cache=cache)
    assert tokenizer.calls > before


def test_a_different_embedder_is_a_different_cache_entry(tmp_path, cfg, tokenizer):
    a = Embedder.load(
        cfg, manifest_path=write_manifest(tmp_path / "a", seed=0), tokenizer=tokenizer
    )
    b = Embedder.load(
        cfg, manifest_path=write_manifest(tmp_path / "b", seed=1), tokenizer=tokenizer
    )
    assert a.fingerprint["ir_sha256"] != b.fingerprint["ir_sha256"]
    assert embed_config_hash(a.fingerprint) != embed_config_hash(b.fingerprint)

    key = {"stage": "embed", "input_hash": hash_texts(TEXTS), "stage_version": STAGE_VERSION}
    cache = VectorCache(tmp_path / "cache")
    path_a = cache.path(CacheKey(**key, config_hash=embed_config_hash(a.fingerprint)))
    path_b = cache.path(CacheKey(**key, config_hash=embed_config_hash(b.fingerprint)))
    assert path_a != path_b


def test_changing_only_a_prefix_invalidates_the_cache(tmp_path, cfg, tokenizer):
    """§3.1 rule 3 — a prefix that does not travel with the index degrades retrieval silently."""
    plain = Embedder.load(cfg, manifest_path=write_manifest(tmp_path / "a"), tokenizer=tokenizer)
    prefixed = Embedder.load(
        cfg,
        manifest_path=write_manifest(tmp_path / "b", passage_prefix="passage: "),
        tokenizer=tokenizer,
    )
    assert embed_config_hash(plain.fingerprint) != embed_config_hash(prefixed.fingerprint)


def test_a_corrupt_cache_entry_is_recomputed_not_returned(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    cache = StageCache(tmp_path / "cache")
    expected = embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)

    entry = next((tmp_path / "cache" / "embed").glob("*.npy"))
    entry.write_bytes(b"not an npy file at all")
    recovered = embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)
    assert np.allclose(recovered, expected, atol=1e-6)


def test_caching_can_be_switched_off(tmp_path, cfg, tokenizer):
    emb = make_embedder(tmp_path, cfg, tokenizer)
    cache = StageCache(tmp_path / "cache", enabled=False)
    embed_chunks(CHUNKS, cfg, embedder=emb, cache=cache)
    assert not (tmp_path / "cache" / "embed").exists()


def test_hash_texts_is_order_sensitive():
    assert hash_texts(TEXTS) != hash_texts(list(reversed(TEXTS)))
    assert hash_texts(TEXTS) == hash_texts(list(TEXTS))


# --- the committed BGE-M3 INT8 IR, when it is actually on disk -----------------------


@pytest.mark.skipif(
    os.environ.get("INTEL2026_REAL_MODELS") != "1",
    reason="set INTEL2026_REAL_MODELS=1 to run against the 544 MB BGE-M3 IR (§7.3)",
)
def test_the_real_embedder_produces_normalised_1024_dim_vectors(cfg):
    emb = Embedder.load(cfg)
    vectors = emb.embed_passages(TEXTS[:3], batch_size=DEFAULT_BATCH_SIZE)
    assert vectors.shape == (3, 1024)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert emb.fingerprint["hf_id"] == "BAAI/bge-m3"
