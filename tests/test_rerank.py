"""Cross-encoder rerank — §10."""

from __future__ import annotations

import pytest

from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk, Retrieved
from retrieve.rerank import RerankingRetriever


def make_hit(rank: int, text: str = "t") -> Retrieved:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        doc_id="d",
        page_start=rank,
        page_end=rank,
        block_ids=[f"b{rank}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=[],
        text=text,
        token_count=1,
        lang="en",
        script="latn",
    )
    return Retrieved(chunk=chunk, score=1.0 / rank, rank=rank)


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={"paths": PathsConfig(data_dir=tmp_path / "d", ov_cache_dir=tmp_path / "o")}
    )


class FakeInner:
    def __init__(self, hits: list[Retrieved]) -> None:
        self.hits = hits
        self.asked_k: int | None = None

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        self.asked_k = k
        return self.hits[:k]


class FakePipe:
    """Returns (index, score) pairs in arbitrary order, like GenAI does."""

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, texts: list[str]):
        self.calls.append((query, len(texts)))
        pairs = [(i, self.scores.get(i, 0.0)) for i in range(len(texts))]
        return list(reversed(pairs))


class ExplodingPipe:
    def rerank(self, query: str, texts: list[str]):
        raise RuntimeError("primitive descriptor failed")


def test_reranking_reorders_by_cross_encoder_score(cfg):
    hits = [make_hit(i) for i in range(1, 6)]
    # Inner ranked c1 best; the cross-encoder says c4 is.
    pipe = FakePipe({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.9, 4: 0.05})

    out = RerankingRetriever(FakeInner(hits), cfg, pipe).retrieve("q", 5)

    assert [h.chunk.chunk_id for h in out][:2] == ["c4", "c3"]
    assert [h.rank for h in out] == [1, 2, 3, 4, 5]


def test_scores_are_restored_to_input_order(cfg):
    """GenAI returns pairs out of order; mapping them wrong silently inverts the ranking."""
    hits = [make_hit(i) for i in range(1, 4)]
    pipe = FakePipe({0: 0.9, 1: 0.1, 2: 0.5})

    out = RerankingRetriever(FakeInner(hits), cfg, pipe).retrieve("q", 3)

    assert [h.chunk.chunk_id for h in out] == ["c1", "c3", "c2"]
    assert out[0].score == pytest.approx(0.9)


def test_only_top_n_is_rescored_and_the_tail_stays_below(cfg):
    raw = cfg.model_dump()
    raw["retrieve"]["rerank"]["top_n"] = 2
    cfg = type(cfg)(**raw)
    hits = [make_hit(i) for i in range(1, 5)]
    pipe = FakePipe({0: 0.1, 1: 0.9})

    out = RerankingRetriever(FakeInner(hits), cfg, pipe).retrieve("q", 4)

    assert pipe.calls == [("q", 2)]
    assert [h.chunk.chunk_id for h in out] == ["c2", "c1", "c3", "c4"]
    assert [h.rank for h in out] == [1, 2, 3, 4]


def test_inner_is_asked_for_the_full_k_not_top_n(cfg):
    """Rerank cannot recover a chunk stage one never returned."""
    inner = FakeInner([make_hit(i) for i in range(1, 21)])
    RerankingRetriever(inner, cfg, FakePipe({})).retrieve("q", 20)
    assert inner.asked_k == 20


def test_a_failing_pipeline_degrades_to_the_inner_ranking(cfg):
    hits = [make_hit(i) for i in range(1, 4)]
    out = RerankingRetriever(FakeInner(hits), cfg, ExplodingPipe()).retrieve("q", 3)
    assert [h.chunk.chunk_id for h in out] == ["c1", "c2", "c3"]


def test_single_hit_skips_the_model(cfg):
    pipe = FakePipe({})
    out = RerankingRetriever(FakeInner([make_hit(1)]), cfg, pipe).retrieve("q", 5)
    assert len(out) == 1
    assert pipe.calls == []


def test_rejects_bad_k(cfg):
    with pytest.raises(ValueError, match="k must be"):
        RerankingRetriever(FakeInner([]), cfg, FakePipe({})).retrieve("q", 0)
