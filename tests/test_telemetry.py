"""Traces are only useful if they survive the parquet round trip and never lie about a
fallback, a tier, or an abstain. Timings are machine-dependent, so nothing here asserts on a
duration value — only on structure, provenance and error handling."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.schema import Answer, Chunk, Retrieved
from core.telemetry import (
    TIER_LOCAL_INDEX,
    TIER_PARAMETRIC,
    DeviceUse,
    QueryTrace,
    StageTiming,
    TraceRecorder,
    new_trace_id,
    read_traces,
    trace_path,
    traces_root,
    write_traces,
)


class FakeClock:
    """Monotonic, deterministic. Every read advances by `step` seconds."""

    def __init__(self, step: float = 0.5) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


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


def hit(score: float, rank: int = 1) -> Retrieved:
    return Retrieved(chunk=chunk(), score=score, rank=rank)


def a_trace(**over) -> QueryTrace:
    base = dict(
        trace_id="t1",
        started_at=datetime(2026, 8, 22, 14, 30, 5, 123456, tzinfo=UTC),
        query="எப்படி?",
        lang="ta",
        config_hash="sha256:abc",
        tier=TIER_LOCAL_INDEX,
        abstained=False,
        top_score=0.71,
        n_retrieved=20,
        n_citations=3,
        prompt_tokens=1200,
        completion_tokens=180,
        ttft_ms=940.5,
        total_ms=4210.25,
        stages=(StageTiming("embed", 12.5), StageTiming("retrieve", 8.0)),
        devices=(DeviceUse("embedder", "GPU", "CPU"),),
    )
    return QueryTrace(**{**base, **over})


class TestValidation:
    def test_unknown_tier_rejected(self):
        with pytest.raises(ValueError, match="tier"):
            a_trace(tier=9)

    def test_abstained_trace_cannot_carry_citations(self):
        with pytest.raises(ValueError, match="citations"):
            a_trace(abstained=True, n_citations=2)

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError, match="n_retrieved"):
            a_trace(n_retrieved=-1)

    def test_empty_trace_id_rejected(self):
        with pytest.raises(ValueError, match="trace_id"):
            a_trace(trace_id="")

    def test_negative_stage_duration_rejected(self):
        with pytest.raises(ValueError, match="duration_ms"):
            StageTiming("embed", -1.0)

    def test_unnamed_stage_rejected(self):
        with pytest.raises(ValueError, match="stage name"):
            StageTiming("", 1.0)

    def test_tier_label_is_available_for_the_ui(self):
        assert a_trace(tier=TIER_PARAMETRIC).tier_label == "general knowledge"
        assert a_trace().tier_label == "local index"


class TestDeviceUse:
    def test_fallback_is_visible(self):
        assert DeviceUse("embedder", "GPU", "CPU").fell_back

    def test_no_fallback_when_device_matches(self):
        assert not DeviceUse("generator", "CPU", "CPU").fell_back


class TestRoundTrip:
    def test_parquet_round_trip_is_lossless(self, tmp_path):
        original = a_trace()
        path = write_traces([original], tmp_path)
        (restored,) = read_traces(path)
        assert restored == original

    def test_trace_without_stages_or_devices_round_trips(self, tmp_path):
        """An empty list would otherwise infer as List(Null) and poison the day's file."""
        original = a_trace(trace_id="t2", stages=(), devices=())
        path = write_traces([original], tmp_path)
        (restored,) = read_traces(path)
        assert restored == original

    def test_null_optionals_round_trip_as_none(self, tmp_path):
        original = a_trace(top_score=None, ttft_ms=None, abstained=True, n_citations=0)
        path = write_traces([original], tmp_path)
        (restored,) = read_traces(path)
        assert restored.top_score is None
        assert restored.ttft_ms is None
        assert restored.abstained is True

    def test_appending_keeps_earlier_rows(self, tmp_path):
        when = a_trace().started_at
        write_traces([a_trace(trace_id="t1")], tmp_path, when)
        path = write_traces([a_trace(trace_id="t2")], tmp_path, when)
        assert [t.trace_id for t in read_traces(path)] == ["t1", "t2"]

    def test_one_file_per_utc_day(self, tmp_path):
        d1 = write_traces([a_trace()], tmp_path, datetime(2026, 8, 22, tzinfo=UTC))
        d2 = write_traces([a_trace()], tmp_path, datetime(2026, 8, 23, tzinfo=UTC))
        assert d1.name == "2026-08-22.parquet"
        assert d1 != d2

    def test_trace_path_lives_under_the_configured_data_dir(self, tmp_path):
        root = traces_root(tmp_path / "data")
        assert trace_path(root, datetime(2026, 8, 22, tzinfo=UTC)) == (
            tmp_path / "data" / "traces" / "2026-08-22.parquet"
        )

    def test_empty_write_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to write"):
            write_traces([], tmp_path)

    def test_missing_trace_file_is_explicit(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_traces(tmp_path / "2000-01-01.parquet")


class TestRecorder:
    def test_stages_are_recorded_in_order(self):
        rec = TraceRecorder("q", clock=FakeClock())
        with rec.stage("embed"):
            pass
        with rec.stage("retrieve"):
            pass
        trace = rec.finish()
        assert [s.stage for s in trace.stages] == ["embed", "retrieve"]
        assert all(s.duration_ms > 0 for s in trace.stages)

    def test_failing_stage_is_still_timed_and_reraises(self):
        rec = TraceRecorder("q", clock=FakeClock())
        with pytest.raises(RuntimeError, match="ocr blew up"), rec.stage("ocr"):
            raise RuntimeError("ocr blew up")
        trace = rec.finish()
        assert [s.stage for s in trace.stages] == ["ocr"]

    def test_retrieval_sets_top_score_and_count(self):
        rec = TraceRecorder("q", clock=FakeClock())
        rec.record_retrieval([hit(0.9, 1), hit(0.4, 2)])
        trace = rec.finish()
        assert trace.top_score == pytest.approx(0.9)
        assert trace.n_retrieved == 2

    def test_empty_retrieval_leaves_top_score_none(self):
        rec = TraceRecorder("q", clock=FakeClock())
        rec.record_retrieval([])
        trace = rec.finish()
        assert trace.top_score is None
        assert trace.n_retrieved == 0

    def test_tokens_accumulate_across_calls(self):
        rec = TraceRecorder("q", clock=FakeClock())
        rec.record_tokens(prompt=100)
        rec.record_tokens(completion=7)
        rec.record_tokens(completion=5)
        trace = rec.finish()
        assert (trace.prompt_tokens, trace.completion_tokens) == (100, 12)

    def test_first_token_is_stamped_once(self):
        rec = TraceRecorder("q", clock=FakeClock())
        rec.mark_first_token()
        first = rec._ttft_ms
        rec.mark_first_token()
        assert rec._ttft_ms == first

    def test_ttft_is_none_when_nothing_generated(self):
        assert TraceRecorder("q", clock=FakeClock()).finish().ttft_ms is None

    def test_device_fallback_is_captured(self):
        rec = TraceRecorder("q", clock=FakeClock())
        rec.record_device("embedder", requested="GPU", device="CPU")
        (use,) = rec.finish().devices
        assert (use.requested, use.device, use.fell_back) == ("GPU", "CPU", True)

    def test_answer_supplies_abstain_and_citations(self):
        rec = TraceRecorder("q", clock=FakeClock())
        answer = Answer(
            text="see [1]", citations=[hit(0.8)], abstained=False, trace_id=rec.trace_id
        )
        trace = rec.finish(answer)
        assert trace.n_citations == 1
        assert not trace.abstained

    def test_abstained_answer_records_no_citations(self):
        rec = TraceRecorder("q", clock=FakeClock())
        answer = Answer(
            text="not in your documents", citations=[], abstained=True, trace_id=rec.trace_id
        )
        trace = rec.finish(answer)
        assert trace.abstained and trace.n_citations == 0

    def test_mismatched_answer_trace_id_raises(self):
        rec = TraceRecorder("q", clock=FakeClock())
        answer = Answer(text="x", citations=[], abstained=True, trace_id="somebody-elses-id")
        with pytest.raises(ValueError, match="does not join|does not match"):
            rec.finish(answer)

    def test_finish_is_not_repeatable(self):
        rec = TraceRecorder("q", clock=FakeClock())
        rec.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            rec.finish()

    def test_tier_defaults_to_local_index_and_is_overridable(self):
        assert TraceRecorder("q", clock=FakeClock()).finish().tier == TIER_LOCAL_INDEX
        assert (
            TraceRecorder("q", clock=FakeClock()).finish(tier=TIER_PARAMETRIC).tier
            == TIER_PARAMETRIC
        )

    def test_trace_ids_are_unique(self):
        assert len({new_trace_id() for _ in range(200)}) == 200

    def test_recorder_output_survives_parquet(self, tmp_path):
        rec = TraceRecorder("q", lang="ta", config_hash="sha256:x", clock=FakeClock())
        with rec.stage("embed"):
            pass
        rec.record_device("embedder", requested="GPU", device="CPU")
        rec.record_retrieval([hit(0.55)])
        rec.mark_first_token()
        rec.record_tokens(prompt=10, completion=3)
        trace = rec.finish(tier=TIER_LOCAL_INDEX)
        (restored,) = read_traces(write_traces([trace], tmp_path))
        assert restored == trace
        assert restored.stage_ms.keys() == {"embed"}
