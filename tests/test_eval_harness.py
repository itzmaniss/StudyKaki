"""The harness must be trustworthy before any real component plugs into it (§5)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from core.config import load_config
from core.schema import Chunk, Retrieved
from eval.metrics import GoldQuestion, first_relevant_rank, recall_at, reciprocal_rank
from eval.run import RandomRetriever, _pool_from_golden, evaluate, load_golden
from retrieve.retriever import Retriever, abstains


def chunk(doc_id="d1", page=42) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{page}",
        doc_id=doc_id,
        page_start=page,
        page_end=page,
        block_ids=["b"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=[],
        text="t",
        token_count=1,
        lang="en",
        script="latn",
    )


def hits(*specs) -> list[Retrieved]:
    return [
        Retrieved(chunk=chunk(d, p), score=s, rank=i) for i, (d, p, s) in enumerate(specs, start=1)
    ]


GOLD = GoldQuestion(q="?", lang="en", doc_id="d1", gold_pages=[42])


class TestRelevance:
    def test_wrong_doc_is_not_relevant(self):
        assert first_relevant_rank(hits(("d2", 42, 0.9)), GOLD) is None

    def test_wrong_page_is_not_relevant(self):
        assert first_relevant_rank(hits(("d1", 7, 0.9)), GOLD) is None

    def test_page_span_covers_gold_page(self):
        c = Chunk(
            chunk_id="x",
            doc_id="d1",
            page_start=41,
            page_end=43,
            block_ids=["b"],
            bbox_union=(0.0, 0.0, 1.0, 1.0),
            heading_path=[],
            text="t",
            token_count=1,
            lang="en",
            script="latn",
        )
        assert first_relevant_rank([Retrieved(chunk=c, score=0.9, rank=1)], GOLD) == 1

    def test_recall_respects_cutoff(self):
        h = hits(("d2", 1, 0.9), ("d2", 2, 0.8), ("d1", 42, 0.7))
        assert recall_at(h, GOLD, 1) == 0.0
        assert recall_at(h, GOLD, 5) == 1.0

    def test_reciprocal_rank_uses_first_hit(self):
        h = hits(("d2", 1, 0.9), ("d1", 42, 0.8), ("d1", 42, 0.7))
        assert reciprocal_rank(h, GOLD) == pytest.approx(0.5)

    def test_rr_zero_when_beyond_cutoff(self):
        h = [Retrieved(chunk=chunk("d1", 42), score=0.1, rank=11)]
        assert reciprocal_rank(h, GOLD, 10) == 0.0


class TestAbstain:
    def test_abstains_below_tau(self):
        assert abstains(hits(("d1", 42, 0.2)), tau=0.35)

    def test_answers_above_tau(self):
        assert not abstains(hits(("d1", 42, 0.9)), tau=0.35)

    def test_empty_result_abstains(self):
        assert abstains([], tau=0.35)


class TestGoldenFile:
    def test_placeholder_golden_parses(self):
        golden = load_golden()
        assert len(golden) >= 10
        assert any(g.unanswerable for g in golden), "§5 requires unanswerable questions"
        assert {g.lang for g in golden} >= {"en", "ta"}

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('# a comment\n\n{"q":"x","lang":"en","doc_id":"d1","gold_pages":[1]}\n')
        assert len(load_golden(p)) == 1

    def test_malformed_json_reports_line_number(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('{"q":"ok","lang":"en","doc_id":"d1"}\n{ not json\n')
        with pytest.raises(ValueError, match=r":2:"):
            load_golden(p)

    def test_missing_required_key_rejected(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('{"q":"x","lang":"en"}\n')
        with pytest.raises(ValueError, match="doc_id"):
            load_golden(p)

    def test_missing_file_is_explicit(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden(tmp_path / "absent.jsonl")


class TestHarness:
    def test_random_retriever_satisfies_protocol(self):
        r = RandomRetriever(_pool_from_golden(load_golden()))
        assert isinstance(r, Retriever)

    def test_random_retriever_is_deterministic_under_seed(self):
        pool = _pool_from_golden(load_golden())
        a = RandomRetriever(pool, seed=7).retrieve("q", 5)
        b = RandomRetriever(pool, seed=7).retrieve("q", 5)
        assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]

    def test_ranks_are_contiguous_and_scores_descend(self):
        h = RandomRetriever(_pool_from_golden(load_golden()), seed=1).retrieve("q", 10)
        assert [x.rank for x in h] == list(range(1, len(h) + 1))
        assert all(a.score >= b.score for a, b in pairwise(h))

    def test_empty_pool_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            RandomRetriever([])

    def test_evaluate_produces_bounded_metrics(self):
        cfg = load_config()
        golden = load_golden()
        result, df = evaluate(
            RandomRetriever(_pool_from_golden(golden), seed=0), golden, cfg, "random"
        )
        assert result.n_questions == len(golden)
        assert df.height == len(golden)
        for v in (result.recall_at_1, result.recall_at_5, result.mrr_at_10):
            assert 0.0 <= v <= 1.0
        assert result.recall_at_1 <= result.recall_at_5 <= result.recall_at_10
        assert result.groundedness is None, "groundedness needs answer/ — not yet built"

    def test_perfect_retriever_scores_one(self):
        """Guards against a harness that reports zero regardless of input."""
        cfg = load_config()
        golden = [g for g in load_golden() if not g.unanswerable]

        class Perfect:
            def retrieve(self, query: str, k: int) -> list[Retrieved]:
                g = next(x for x in golden if x.q == query)
                return [Retrieved(chunk=chunk(g.doc_id, g.gold_pages[0]), score=1.0, rank=1)]

        result, _ = evaluate(Perfect(), golden, cfg, "perfect")
        assert result.recall_at_1 == 1.0
        assert result.mrr_at_10 == 1.0

    def test_table_renders(self):
        cfg = load_config()
        golden = load_golden()
        result, _ = evaluate(RandomRetriever(_pool_from_golden(golden)), golden, cfg, "random")
        table = result.as_table()
        for col in ("recall@1", "recall@5", "recall@10", "MRR@10", "abstain_precision"):
            assert col in table


# --- groundedness (§5) ---------------------------------------------------------------


class ScriptedGenerator:
    """`StreamingGenerator` that answers every question with the same text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.name = "scripted"
        self.requested_device = "CPU"
        self.device = "CPU"
        self.last_usage = None

    def stream(self, prompt: str, settings):
        yield self.text


def _golden_one():
    return [GoldQuestion(q="what is a matrix?", lang="en", doc_id="d1", gold_pages=[42])]


class AlwaysHits:
    """Scores above any sane tau, so the generator is always reached."""

    def retrieve(self, query: str, k: int):
        return [Retrieved(chunk=chunk(page=42), score=0.99, rank=1)]


def test_groundedness_is_not_measured_without_a_generator():
    cfg = load_config()
    result, df = evaluate(AlwaysHits(), _golden_one(), cfg, label="t")
    assert result.groundedness is None
    assert "n/a" in result.as_table()


def test_an_invented_citation_lowers_groundedness():
    """The whole point of the column: `cite.verify` drops `[7]`, and the score records it.

    One real context block means `[1]` is grounded and `[7]` cannot be. Asserting on the
    ratio, not on any model wording.
    """
    cfg = load_config()
    both = evaluate(AlwaysHits(), _golden_one(), cfg, generator=ScriptedGenerator("a [1] b [7]"))[0]
    clean = evaluate(AlwaysHits(), _golden_one(), cfg, generator=ScriptedGenerator("a [1]"))[0]
    assert both.groundedness == pytest.approx(0.5)
    assert clean.groundedness == pytest.approx(1.0)


def test_an_uncited_answer_is_ungrounded():
    """§4 wants a citation on every claim, so asserting without one scores zero."""
    cfg = load_config()
    result = evaluate(
        AlwaysHits(), _golden_one(), cfg, generator=ScriptedGenerator("just trust me")
    )[0]
    assert result.groundedness == pytest.approx(0.0)
