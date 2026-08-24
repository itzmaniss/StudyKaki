"""Conditional query rewrite — §10."""

from __future__ import annotations

import pytest

from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk, Retrieved
from retrieve.rewrite import RewritingRetriever, has_dangling_reference, should_rewrite


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={"paths": PathsConfig(data_dir=tmp_path / "d", ov_cache_dir=tmp_path / "o")}
    )


class SpyRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        self.queries.append(query)
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d",
            page_start=1,
            page_end=1,
            block_ids=["b"],
            bbox_union=(0.0, 0.0, 1.0, 1.0),
            heading_path=[],
            text="t",
            token_count=1,
            lang="en",
            script="latn",
        )
        return [Retrieved(chunk=chunk, score=0.9, rank=1)]


# --- the trigger (§10: only short or dangling) ----------------------------------------


def test_short_query_triggers():
    assert should_rewrite("what about it", trigger_max_tokens=5)


def test_long_standalone_question_does_not_trigger():
    q = "how does photosynthesis convert light energy into chemical energy in plants"
    assert not should_rewrite(q, trigger_max_tokens=5)


def test_dangling_pronoun_triggers_even_when_long():
    q = "can you explain how that works in the second chapter of the textbook please"
    assert has_dangling_reference(q)
    assert should_rewrite(q, trigger_max_tokens=5)


def test_tamil_standalone_question_does_not_trigger():
    """English pronoun list must not fire on a Tamil question that names its subject."""
    q = "ஜார்ஜ் பூல் என்பவர் யார் என்று விளக்கமாக கூறுங்கள் இப்போது"
    assert not has_dangling_reference(q)


# --- wiring ---------------------------------------------------------------------------


def test_untriggered_query_reaches_the_inner_retriever_unchanged(cfg):
    spy = SpyRetriever()
    r = RewritingRetriever(spy, cfg, lambda q: "REWRITTEN")

    r.retrieve("how does photosynthesis convert light into chemical energy", 5)

    assert spy.queries == ["how does photosynthesis convert light into chemical energy"]
    assert r.last_rewrite is None


def test_triggered_query_is_rewritten_before_retrieval(cfg):
    spy = SpyRetriever()
    r = RewritingRetriever(spy, cfg, lambda q: "photosynthesis light reactions")

    r.retrieve("what about it", 5)

    assert spy.queries == ["photosynthesis light reactions"]
    assert r.last_rewrite == "photosynthesis light reactions"


def test_no_rewriter_is_a_passthrough(cfg):
    spy = SpyRetriever()
    RewritingRetriever(spy, cfg, None).retrieve("what about it", 5)
    assert spy.queries == ["what about it"]


def test_empty_rewrite_falls_back_to_the_original(cfg):
    spy = SpyRetriever()
    r = RewritingRetriever(spy, cfg, lambda q: "   ")

    r.retrieve("what about it", 5)

    assert spy.queries == ["what about it"]
    assert r.last_rewrite is None


def test_runaway_rewrite_falls_back_to_the_original(cfg):
    """A model that ignores 'reply with the query only' must not poison retrieval."""
    spy = SpyRetriever()
    r = RewritingRetriever(spy, cfg, lambda q: "sure! here is a standalone query: " + "x " * 200)

    r.retrieve("what about it", 5)

    assert spy.queries == ["what about it"]


def test_preamble_and_quotes_are_stripped(cfg):
    spy = SpyRetriever()
    r = RewritingRetriever(spy, cfg, lambda q: '"boolean algebra"\nextra junk')

    r.retrieve("what about it", 5)

    assert spy.queries == ["boolean algebra"]
