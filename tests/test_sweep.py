"""§10 arm sweep — permutations, and refusing to report an arm that did not run."""

from __future__ import annotations

import pytest

from core.config import DEFAULT_CONFIG, load_config
from eval.sweep import ARMS, SweepRow, as_table, config_for

BASELINE = {"recall@5": 0.898, "mrr@10": 0.736}


class FakeResult:
    def __init__(self, r5=0.9, mrr=0.75):
        self.recall_at_5 = r5
        self.mrr_at_10 = mrr
        self.groundedness = None


@pytest.fixture
def cfg():
    return load_config(DEFAULT_CONFIG)


def test_every_arm_defaults_off_so_base_yaml_is_the_v1_baseline(cfg):
    for arm in ARMS:
        assert getattr(cfg.retrieve, arm).enabled is False


def test_config_for_enables_exactly_the_named_arms(cfg):
    only_hybrid = config_for(cfg, ("hybrid",))
    assert only_hybrid.retrieve.hybrid.enabled is True
    assert only_hybrid.retrieve.rerank.enabled is False
    assert only_hybrid.retrieve.rewrite.enabled is False


def test_toggling_an_arm_does_not_move_the_chunk_config_hash(cfg):
    """An arm must never invalidate the ingest cache."""
    assert config_for(cfg, ARMS).chunk_config_hash == cfg.chunk_config_hash


# --- the bug: a degraded arm must not be reported as a measurement -------------------


def test_a_degraded_row_withholds_its_metrics():
    rows = [SweepRow(arms=("rerank",), result=FakeResult(), seconds=1.0, degraded=54)]
    table = as_table(rows, BASELINE, 54)

    assert "DEGRADED" in table
    assert "0.900" not in table, "printed a number the arm never produced"


def test_a_clean_row_is_reported_normally():
    rows = [SweepRow(arms=("hybrid",), result=FakeResult(), seconds=1.0, degraded=0)]
    table = as_table(rows, BASELINE, 54)

    assert "DEGRADED" not in table
    assert "0.900" in table


def test_one_degraded_query_is_enough_to_withhold():
    """53 good queries and 1 silent fallback is still not a measurement of this arm."""
    rows = [SweepRow(arms=("rerank",), result=FakeResult(), seconds=1.0, degraded=1)]
    assert "DEGRADED" in as_table(rows, BASELINE, 54)


def test_trustworthy_requires_both_a_result_and_no_degradation():
    assert SweepRow(arms=(), result=FakeResult(), seconds=1.0).trustworthy is True
    assert SweepRow(arms=(), result=FakeResult(), seconds=1.0, degraded=1).trustworthy is False
    assert SweepRow(arms=(), result=None, seconds=0.0, skipped="x").trustworthy is False


def test_a_skipped_row_says_why():
    rows = [SweepRow(arms=("rerank",), result=None, seconds=0.0, skipped="no model")]
    assert "skipped: no model" in as_table(rows, BASELINE, 54)
