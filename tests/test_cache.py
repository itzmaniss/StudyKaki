from __future__ import annotations

import pytest

from core.cache import (
    CacheCorruptError,
    CacheKey,
    StageCache,
    hash_bytes,
    hash_rows,
    stage_timer,
)
from core.config import load_config
from core.schema import Block, Chunk, Document


def make_block(i: int = 0, **kw) -> Block:
    base = dict(
        block_id=f"b{i}",
        doc_id="doc-a",
        page=1,
        bbox=(0.1, 0.2, 0.8, 0.9),
        kind="paragraph",
        reading_order=i,
        script="latn",
        text=f"block {i}",
        ocr_confidence=None,
    )
    return Block(**{**base, **kw})


def make_chunk(**kw) -> Chunk:
    base = dict(
        chunk_id="c0",
        doc_id="doc-a",
        page_start=1,
        page_end=2,
        block_ids=["b0", "b1"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3", "3.2 Photosynthesis"],
        text="hello",
        token_count=1,
        lang="en",
        script="latn",
    )
    return Chunk(**{**base, **kw})


def make_document(**kw) -> Document:
    base = dict(
        doc_id="doc-a",
        filename="bio.pdf",
        mime="application/pdf",
        n_pages=3,
        has_text_layer=True,
        pipeline_version="v1",
    )
    return Document(**{**base, **kw})


def key(**kw) -> CacheKey:
    base = dict(stage="chunk", input_hash="in0", stage_version="chunk/1", config_hash="sha256:cfg0")
    return CacheKey(**{**base, **kw})


@pytest.fixture
def cache(tmp_path) -> StageCache:
    return StageCache(tmp_path / "cache")


@pytest.mark.parametrize(
    ("rows", "row_type"),
    [
        ([make_block(0), make_block(1, page=2, script="taml", ocr_confidence=0.42)], Block),
        ([make_chunk()], Chunk),
        ([make_document()], Document),
    ],
)
def test_round_trip_is_lossless(cache, rows, row_type):
    k = key(stage=row_type.__name__.lower())
    cache.store(k, rows, row_type)
    assert cache.load(k, row_type) == rows


def test_empty_output_round_trips(cache):
    """A stage that legitimately produces nothing must cache as a hit, not a miss."""
    k = key()
    cache.store(k, [], Block)
    assert cache.has(k)
    assert cache.load(k, Block) == []


def test_miss_returns_none(cache):
    assert cache.load(key(), Block) is None
    assert not cache.has(key())


def test_writes_where_the_architecture_says(cache):
    k = key(stage="normalize")
    path = cache.store(k, [make_block()], Block)
    assert path == cache.root / "normalize" / f"{k.content_hash}.parquet"
    assert path.suffix == ".parquet"


@pytest.mark.parametrize(
    "changed",
    [
        {"input_hash": "in1"},
        {"stage_version": "chunk/2"},
        {"config_hash": "sha256:cfg1"},
        {"stage": "normalize"},
    ],
)
def test_every_key_component_changes_the_entry(cache, changed):
    """(input_hash, stage_version, config_hash) — a stale hit here is a silent wrong answer."""
    assert cache.path(key()) != cache.path(key(**changed))


def test_get_or_compute_computes_once(cache):
    calls = []

    def compute():
        calls.append(1)
        return [make_block(7)]

    first = cache.get_or_compute(key(), Block, compute)
    second = cache.get_or_compute(key(), Block, compute)
    assert first == second == [make_block(7)]
    assert len(calls) == 1


def test_store_is_idempotent(cache):
    k = key()
    rows = [make_block(0), make_block(1)]
    first = cache.store(k, rows, Block).read_bytes()
    second = cache.store(k, rows, Block).read_bytes()
    assert first == second
    assert cache.load(k, Block) == rows


def test_no_temp_files_survive_a_write(cache):
    cache.store(key(), [make_block()], Block)
    assert [p.name for p in cache.root.rglob("*") if p.suffix == ".tmp"] == []


def test_disabled_cache_never_writes(tmp_path):
    cache = StageCache(tmp_path / "cache", enabled=False)
    calls = []

    def compute():
        calls.append(1)
        return [make_block()]

    cache.get_or_compute(key(), Block, compute)
    cache.get_or_compute(key(), Block, compute)
    assert len(calls) == 2
    assert not cache.root.exists()


def test_unreadable_entry_is_a_miss_and_self_heals(cache):
    k = key()
    path = cache.path(k)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"definitely not parquet")
    assert cache.load(k, Block) is None
    assert cache.get_or_compute(k, Block, lambda: [make_block(3)]) == [make_block(3)]
    assert cache.load(k, Block) == [make_block(3)]


def test_entry_of_the_wrong_row_type_raises(cache):
    """Better to fail loudly than to hand a half-built Chunk to the indexer."""
    k = key()
    cache.store(k, [make_block()], Block)
    with pytest.raises(CacheCorruptError, match="does not decode as Chunk"):
        cache.load(k, Chunk)


def test_store_rejects_mismatched_rows(cache):
    with pytest.raises(TypeError, match="expected all rows to be Block"):
        cache.store(key(), [make_chunk()], Block)


@pytest.mark.parametrize(
    "kw",
    [
        {"stage": "../escape"},
        {"stage": "Chunk"},
        {"stage": ""},
        {"input_hash": ""},
        {"stage_version": ""},
        {"config_hash": ""},
    ],
)
def test_malformed_keys_are_rejected(kw):
    with pytest.raises(ValueError):
        key(**kw)


def test_from_config_stays_under_the_configured_data_dir():
    cfg = load_config()
    cache = StageCache.from_config(cfg)
    assert cache.root == cfg.resolve(cfg.paths.data_dir) / "cache"


def test_hash_bytes_is_sha256_of_content():
    assert hash_bytes(b"abc") == hash_bytes(b"abc")
    assert hash_bytes(b"abc") != hash_bytes(b"abd")
    assert len(hash_bytes(b"")) == 64


def test_hash_rows_tracks_content_not_identity():
    assert hash_rows([make_block(0)]) == hash_rows([make_block(0)])
    assert hash_rows([make_block(0)]) != hash_rows([make_block(0, text="different")])
    assert hash_rows([make_block(0), make_block(1)]) != hash_rows([make_block(1), make_block(0)])


def test_stage_timer_reports_what_the_stage_recorded():
    with stage_timer("chunk", "in0") as span:
        span.n_out = 12
        span.cached = True
    assert (span.stage, span.n_out, span.cached) == ("chunk", 12, True)


def test_stage_timer_logs_even_when_the_stage_raises():
    with pytest.raises(ZeroDivisionError), stage_timer("chunk", "in0") as span:
        span.n_out = 3
        raise ZeroDivisionError
