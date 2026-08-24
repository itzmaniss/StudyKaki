"""Answer generation — ARCHITECTURE.md §4, §9, §1, §0.6.

Every test here runs against a fake generator: the §4 contract is about *structure*
(abstain, provenance, tier, telemetry), not about what a model says, and CLAUDE.md forbids
asserting on model output text. The only literal strings checked are the two the
architecture fixes verbatim, `ABSTAIN_MESSAGE` and `TIER3_DISCLAIMER`.

`OpenVinoGenerator` is exercised against a fake `LLMPipeline` — an object with
`generate(prompt, config, streamer)` — so the streaming machinery is covered without the
2.5 GB INT4 IR being converted.
"""

from __future__ import annotations

import time

import pytest

from answer.cite import find_markers, has_citation_markers
from answer.generate import (
    STAGE_GENERATE,
    STAGE_GENERATE_TIER3,
    STAGE_RETRIEVE,
    AnswerStream,
    GenerationError,
    GenerationSettings,
    OpenVinoGenerator,
    StreamingGenerator,
    TokenUsage,
    generate_answer,
    stream_answer,
)
from answer.prompt import ABSTAIN_MESSAGE, TIER3_DISCLAIMER
from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk, Retrieved
from core.telemetry import TIER_LOCAL_INDEX, TIER_PARAMETRIC, TraceRecorder

# --- fixtures and fakes --------------------------------------------------------------


def make_hit(
    rank: int,
    *,
    score: float | None = None,
    text: str | None = None,
    lang: str = "en",
    script: str = "latn",
) -> Retrieved:
    body = f"body of block {rank}" if text is None else text
    chunk = Chunk(
        chunk_id=f"c{rank}",
        doc_id="sha256:deadbeefcafef00d",
        page_start=rank,
        page_end=rank,
        block_ids=[f"b{rank}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3", "3.2 Photosynthesis"],
        text=body,
        token_count=len(body.split()) or 1,
        lang=lang,
        script=script,
    )
    return Retrieved(chunk=chunk, score=0.9 if score is None else score, rank=rank)


@pytest.fixture
def cfg(tmp_path):
    """Base config with caches redirected — nothing in tests may write to `data/`."""
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=tmp_path / "ov_cache")
        }
    )


@pytest.fixture
def hits():
    """Three hits comfortably above the configured tau."""
    return [make_hit(i, score=0.9 - 0.1 * i) for i in range(1, 4)]


class FakeGenerator:
    """Scripted `StreamingGenerator`. One list of pieces per generation pass."""

    def __init__(self, *passes: list[str], usage: TokenUsage | None = None) -> None:
        self.passes = list(passes) or [["an answer [1]"]]
        self.name = "fake-generator"
        self.requested_device = "GPU"
        self.device = "CPU"
        self.prompts: list[str] = []
        self.emitted: list[str] = []
        self.last_usage = usage

    def stream(self, prompt: str, settings: GenerationSettings):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.passes) - 1)
        for piece in self.passes[index]:
            self.emitted.append(piece)
            yield piece


class ExplodingGenerator(FakeGenerator):
    def stream(self, prompt: str, settings: GenerationSettings):
        self.prompts.append(prompt)
        yield "partial "
        raise GenerationError("pipeline died mid-decode")


class CountingRecorder(TraceRecorder):
    """Counts `mark_first_token` calls so 'once per pass, not once per token' is testable."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.marks = 0

    def mark_first_token(self) -> None:
        self.marks += 1
        super().mark_first_token()


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int):
        self.calls.append((query, k))
        return self._hits[:k]


def drain(stream: AnswerStream) -> list[str]:
    return list(stream)


# --- abstain (§0.6, §4) --------------------------------------------------------------


def test_below_tau_abstains_without_running_the_model(cfg, hits):
    low = [make_hit(1, score=cfg.retrieve.tau - 0.01)]
    gen = FakeGenerator()

    result = generate_answer("what is photosynthesis?", generator=gen, cfg=cfg, hits=low)

    assert result.answer.abstained
    assert result.answer.citations == []
    assert result.answer.text == ABSTAIN_MESSAGE
    assert result.tier == TIER_LOCAL_INDEX
    assert gen.prompts == []


def test_no_hits_at_all_abstains(cfg):
    result = generate_answer("q", generator=FakeGenerator(), cfg=cfg, hits=[])
    assert result.answer.abstained
    assert result.answer.citations == []


def test_model_signalled_abstain_carries_no_citations(cfg, hits):
    """§4 rule 5: the model may decline. It must not then point at pages."""
    gen = FakeGenerator([f'"{ABSTAIN_MESSAGE}"'])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits)

    assert result.answer.abstained
    assert result.answer.citations == []
    assert result.answer.text == ABSTAIN_MESSAGE


def test_empty_generation_abstains(cfg, hits):
    result = generate_answer("q", generator=FakeGenerator(["", "   "]), cfg=cfg, hits=hits)
    assert result.answer.abstained
    assert result.answer.citations == []


def test_abstained_answer_is_streamed_so_the_ui_has_something_to_render(cfg):
    stream = stream_answer("q", generator=FakeGenerator(), cfg=cfg, hits=[])
    assert drain(stream) == [ABSTAIN_MESSAGE]


# --- grounded answers and citation verification (§4) ---------------------------------


def test_valid_markers_become_citations_pointing_at_the_cited_block(cfg, hits):
    gen = FakeGenerator(["carbon dioxide enters the leaf [2]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits)

    assert not result.answer.abstained
    assert [c.chunk.chunk_id for c in result.answer.citations] == [hits[1].chunk.chunk_id]


def test_invented_markers_never_reach_the_answer(cfg, hits):
    """The single most important guarantee in §4: strip any `[n]` the model invented."""
    gen = FakeGenerator(["grounded [1] and fabricated [9] and out of range [42]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits)

    assert find_markers(result.answer.text) == [1]
    assert [c.chunk.chunk_id for c in result.answer.citations] == [hits[0].chunk.chunk_id]


def test_citations_are_deduplicated_and_ordered_by_block(cfg, hits):
    gen = FakeGenerator(["a [3] b [1] c [3]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits)

    assert [c.rank for c in result.answer.citations] == [hits[0].rank, hits[2].rank]


def test_only_n_context_hits_are_offered_to_the_model(cfg):
    many = [make_hit(i, score=0.9) for i in range(1, 11)]
    gen = FakeGenerator(["cited [6]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=many)

    assert len(result.context) == cfg.retrieve.n_context
    assert f"[{cfg.retrieve.n_context}]" in gen.prompts[0]
    assert f"[{cfg.retrieve.n_context + 1}]" not in gen.prompts[0]
    # `[6]` indexes a block the model was never shown, so it is an invention like any other.
    assert result.answer.citations == []


# --- the prompt budget (BLOCKERS #11) ------------------------------------------------


def tamil_hits(n: int, *, chars: int = 2200) -> list[Retrieved]:
    """`n` Tamil chunks of realistic size — the shape that made every Tamil question fail."""
    return [
        make_hit(i, score=0.9, text="தமிழ் " * (chars // 6), lang="ta", script="taml")
        for i in range(1, n + 1)
    ]


class TokenizingGenerator(FakeGenerator):
    """A generator that can count its own tokens, like `OpenVinoGenerator` does."""

    def __init__(self, *passes: list[str], chars_per_token: float = 1.10) -> None:
        super().__init__(*passes)
        self.chars_per_token = chars_per_token
        self.counted: list[int] = []

    def count_tokens(self, text: str) -> int:
        n = int(len(text) / self.chars_per_token)
        self.counted.append(n)
        return n


def test_tamil_context_is_trimmed_instead_of_overflowing_the_model(cfg):
    """BLOCKERS #11 regression: five Tamil chunks used to reach the model and crash it.

    Asserts on the budget contract, not on a chunk count: the point is that whatever reaches
    the model fits, and that an answer comes back at all.
    """
    budget = cfg.generate.max_prompt_tokens
    gen = TokenizingGenerator(["ஜார்ஜ் பூல் [1]"])

    result = generate_answer("கேள்வி", generator=gen, cfg=cfg, hits=tamil_hits(5))

    assert gen.count_tokens(gen.prompts[0]) <= budget
    assert len(result.context) < cfg.retrieve.n_context
    assert result.answer.abstained is False


def test_english_context_of_the_same_shape_is_left_alone(cfg):
    """The trim is budget-driven, not language-driven — Latin at this size still fits."""
    english = [make_hit(i, score=0.9, text="photosynthesis " * 30) for i in range(1, 6)]
    gen = TokenizingGenerator(["an answer [1]"], chars_per_token=2.33)

    result = generate_answer("q", generator=gen, cfg=cfg, hits=english)

    assert len(result.context) == cfg.retrieve.n_context


def test_the_generators_own_tokenizer_is_preferred_over_the_estimate(cfg):
    gen = TokenizingGenerator(["an answer [1]"])

    generate_answer("q", generator=gen, cfg=cfg, hits=tamil_hits(5))

    assert gen.counted, "fit_context did not consult the generator's count_tokens"


def test_a_generator_without_a_tokenizer_still_gets_trimmed(cfg):
    """`count_tokens` is optional on the Protocol; the estimate must cover for it."""
    gen = FakeGenerator(["ஜார்ஜ் பூல் [1]"])
    assert not hasattr(gen, "count_tokens")

    result = generate_answer("கேள்வி", generator=gen, cfg=cfg, hits=tamil_hits(5))

    assert len(result.context) < cfg.retrieve.n_context


def test_citations_resolve_against_the_trimmed_context_not_the_requested_one(cfg):
    """The numbering contract must follow the blocks that survived the trim."""
    gen = TokenizingGenerator(["cited [5]"])

    result = generate_answer("கேள்வி", generator=gen, cfg=cfg, hits=tamil_hits(5))

    # `[5]` indexes a block that was trimmed away, so it is an invention like any other.
    assert len(result.context) < 5
    assert result.answer.citations == []
    assert result.markers_emitted == 1
    assert result.markers_grounded == 0


def test_a_single_oversized_chunk_is_attempted_rather_than_abstained(cfg):
    """One chunk over budget still goes to the model — #11 prefers a try to a refusal."""
    huge = [make_hit(1, score=0.9, text="தமிழ் " * 4000, lang="ta", script="taml")]
    gen = TokenizingGenerator(["ஒரு பதில் [1]"])

    result = generate_answer("கேள்வி", generator=gen, cfg=cfg, hits=huge)

    assert len(result.context) == 1
    assert gen.prompts, "the model was never called"
    assert result.answer.abstained is False


def test_provenance_travels_into_the_prompt(cfg, hits):
    gen = FakeGenerator()
    generate_answer("q", generator=gen, cfg=cfg, hits=hits, doc_names={hits[0].chunk.doc_id: "bio"})
    assert "bio / p.1" in gen.prompts[0]


# --- Tier 3, model parametric knowledge (§9) -----------------------------------------


def test_tier3_is_off_by_default(cfg):
    gen = FakeGenerator()
    result = generate_answer("q", generator=gen, cfg=cfg, hits=[])

    assert result.tier == TIER_LOCAL_INDEX
    assert TIER3_DISCLAIMER not in result.answer.text
    assert gen.prompts == []


def test_tier3_emits_no_citation_markers(cfg):
    """§9: never fabricate an `[n]` or a page number, whatever the model writes."""
    gen = FakeGenerator(["chlorophyll absorbs light [1] as shown on p.12 [2]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=[], tier3_enabled=True)

    assert not has_citation_markers(result.answer.text)
    assert result.answer.citations == []
    assert result.tier == TIER_PARAMETRIC


def test_tier3_declares_itself_and_says_tier1_abstained(cfg):
    gen = FakeGenerator(["general knowledge body"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=[], tier3_enabled=True)

    assert TIER3_DISCLAIMER in result.answer.text
    assert ABSTAIN_MESSAGE in result.answer.text
    assert result.answer.text.index(ABSTAIN_MESSAGE) < result.answer.text.index(TIER3_DISCLAIMER)
    assert not result.answer.abstained


def test_tier3_disclaimer_is_streamed_before_the_model_text(cfg):
    stream = stream_answer(
        "q", generator=FakeGenerator(["body"]), cfg=cfg, hits=[], tier3_enabled=True
    )
    deltas = drain(stream)

    assert ABSTAIN_MESSAGE in deltas[0]
    assert TIER3_DISCLAIMER in deltas[1]
    assert deltas[-1] == "body"


def test_tier3_never_runs_when_tier1_answered(cfg, hits):
    gen = FakeGenerator(["grounded [1]"], ["ungrounded"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits, tier3_enabled=True)

    assert result.tier == TIER_LOCAL_INDEX
    assert len(gen.prompts) == 1


def test_tier3_takes_over_when_the_model_abstains_on_context(cfg, hits):
    gen = FakeGenerator([ABSTAIN_MESSAGE], ["from general knowledge [1]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits, tier3_enabled=True)

    assert result.tier == TIER_PARAMETRIC
    assert not has_citation_markers(result.answer.text)
    assert result.answer.citations == []
    assert result.context == ()


# --- streaming (§1) ------------------------------------------------------------------


def test_stream_yields_incrementally_not_in_one_lump(cfg, hits):
    gen = FakeGenerator([["one ", "two ", "three [1]"]][0])
    stream = stream_answer("q", generator=gen, cfg=cfg, hits=hits)

    it = iter(stream)
    first = next(it)

    assert first == "one "
    assert gen.emitted == ["one "], "the whole answer was produced before the first delta"

    rest = list(it)
    assert rest == ["two ", "three [1]"]


def test_result_is_only_available_once_the_stream_finishes(cfg, hits):
    stream = stream_answer("q", generator=FakeGenerator(), cfg=cfg, hits=hits)
    with pytest.raises(RuntimeError, match="iterate the stream"):
        _ = stream.result

    drain(stream)
    assert stream.result.answer.text


def test_stream_is_single_use(cfg, hits):
    stream = stream_answer("q", generator=FakeGenerator(), cfg=cfg, hits=hits)
    drain(stream)
    with pytest.raises(RuntimeError, match="single-use"):
        iter(stream)


def test_generator_failure_propagates_with_the_model_named(cfg, hits):
    stream = stream_answer("q", generator=ExplodingGenerator(), cfg=cfg, hits=hits)
    with pytest.raises(GenerationError, match="mid-decode"):
        drain(stream)


def test_non_text_delta_is_rejected(cfg, hits):
    class BadGenerator(FakeGenerator):
        def stream(self, prompt, settings):
            yield 42

    with pytest.raises(GenerationError, match="expected str"):
        generate_answer("q", generator=BadGenerator(), cfg=cfg, hits=hits)


# --- telemetry wiring (§9, §5) -------------------------------------------------------


def test_first_token_is_marked_once_per_pass_not_once_per_token(cfg, hits):
    rec = CountingRecorder("q", config_hash=cfg.config_hash)
    gen = FakeGenerator(["", "a ", "b ", "c [1]"])

    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits, trace=rec)

    assert rec.marks == 1, "empty deltas must not count, and later tokens must not re-mark"
    assert result.trace.ttft_ms is not None
    assert result.trace.ttft_ms <= result.trace.total_ms


def test_abstain_records_no_ttft_because_nothing_was_generated(cfg):
    result = generate_answer("q", generator=FakeGenerator(), cfg=cfg, hits=[])
    assert result.trace.ttft_ms is None


def test_trace_records_the_tier_on_every_answer(cfg, hits):
    grounded = generate_answer("q", generator=FakeGenerator(["a [1]"]), cfg=cfg, hits=hits)
    ungrounded = generate_answer(
        "q", generator=FakeGenerator(["a"]), cfg=cfg, hits=[], tier3_enabled=True
    )

    assert grounded.trace.tier == TIER_LOCAL_INDEX
    assert ungrounded.trace.tier == TIER_PARAMETRIC
    assert ungrounded.tier_label


def test_trace_records_the_device_the_generator_actually_got(cfg, hits):
    result = generate_answer("q", generator=FakeGenerator(), cfg=cfg, hits=hits)

    device = result.trace.devices[0]
    assert (device.requested, device.device) == ("GPU", "CPU")
    assert device.fell_back


def test_device_is_recorded_once_across_both_tiers(cfg, hits):
    gen = FakeGenerator([ABSTAIN_MESSAGE], ["general"])
    result = generate_answer("q", generator=gen, cfg=cfg, hits=hits, tier3_enabled=True)

    assert len(result.trace.devices) == 1
    assert set(result.trace.stage_ms) == {STAGE_GENERATE, STAGE_GENERATE_TIER3}


def test_token_counts_come_from_the_generator_or_stay_zero(cfg, hits):
    counted = FakeGenerator(["a [1]"], usage=TokenUsage(prompt=311, completion=17))
    with_usage = generate_answer("q", generator=counted, cfg=cfg, hits=hits)
    without = generate_answer("q", generator=FakeGenerator(["a [1]"]), cfg=cfg, hits=hits)

    assert (with_usage.trace.prompt_tokens, with_usage.trace.completion_tokens) == (311, 17)
    assert (without.trace.prompt_tokens, without.trace.completion_tokens) == (0, 0)


def test_answer_and_trace_share_the_trace_id(cfg, hits):
    rec = TraceRecorder("q", config_hash=cfg.config_hash)
    result = generate_answer("q", generator=FakeGenerator(), cfg=cfg, hits=hits, trace=rec)

    assert result.answer.trace_id == rec.trace_id == result.trace.trace_id


def test_retrieval_runs_through_the_retriever_at_configured_k(cfg):
    retriever = FakeRetriever([make_hit(i, score=0.9) for i in range(1, 21)])
    gen = FakeGenerator(["a [1]"])

    result = generate_answer("what is photosynthesis?", generator=gen, cfg=cfg, retriever=retriever)

    assert retriever.calls == [("what is photosynthesis?", cfg.retrieve.k)]
    assert STAGE_RETRIEVE in result.trace.stage_ms
    assert result.trace.n_retrieved == cfg.retrieve.k
    assert result.trace.top_score == pytest.approx(0.9)


def test_query_language_is_detected_for_the_trace(cfg, hits):
    result = generate_answer("ஒளிச்சேர்க்கை என்றால் என்ன?", generator=FakeGenerator(), cfg=cfg, hits=hits)
    assert result.trace.lang == "ta"


# --- argument validation -------------------------------------------------------------


def test_exactly_one_source_of_hits_is_required(cfg, hits):
    with pytest.raises(ValueError, match="exactly one"):
        stream_answer("q", generator=FakeGenerator(), cfg=cfg)
    with pytest.raises(ValueError, match="exactly one"):
        stream_answer(
            "q", generator=FakeGenerator(), cfg=cfg, hits=hits, retriever=FakeRetriever(hits)
        )


def test_empty_question_is_rejected(cfg, hits):
    with pytest.raises(ValueError, match="non-empty"):
        stream_answer("   ", generator=FakeGenerator(), cfg=cfg, hits=hits)


def test_fake_generator_satisfies_the_protocol():
    assert isinstance(FakeGenerator(), StreamingGenerator)


# --- OpenVinoGenerator against a fake LLMPipeline (§1, §7.4) -------------------------


class FakePipe:
    """Anything with `generate(prompt, config, streamer)` is an `LLMPipeline` here."""

    def __init__(self, pieces, *, perf=None, error=None):
        self.pieces = pieces
        self.perf = perf
        self.error = error
        self.calls: list[tuple[str, object]] = []
        self.cancelled = False
        self.emitted = 0

    def generate(self, prompt, generation_config=None, streamer=None):
        self.calls.append((prompt, generation_config))
        if self.error is not None:
            raise self.error
        for piece in self.pieces:
            self.emitted += 1
            if streamer is not None and streamer(piece) != 0:
                self.cancelled = True
                break
        return _Result(self.perf)


class _Result:
    def __init__(self, perf):
        self.perf_metrics = perf
        self.texts = [""]


class FakePerf:
    def __init__(self, prompt_tokens, completion_tokens):
        self._p = prompt_tokens
        self._c = completion_tokens

    def get_num_input_tokens(self):
        return self._p

    def get_num_generated_tokens(self):
        return self._c


def endless_pieces(limit=2000):
    for i in range(limit):
        time.sleep(0.001)
        yield f"tok{i} "


@pytest.fixture
def settings():
    return GenerationSettings(max_new_tokens=32, temperature=0.2)


def test_openvino_generator_streams_the_callback_pieces_through(settings):
    pipe = FakePipe(["Photo", "synthesis", " is"], perf=FakePerf(311, 3))
    gen = OpenVinoGenerator(pipe, name="qwen3-4b-instruct", requested_device="CPU", device="CPU")

    deltas = list(gen.stream("prompt", settings))

    assert deltas == ["Photo", "synthesis", " is"]
    assert gen.last_usage == TokenUsage(prompt=311, completion=3)
    assert pipe.calls[0][0] == "prompt"


def test_openvino_generator_reports_no_usage_when_the_build_has_no_metrics(settings):
    gen = OpenVinoGenerator(FakePipe(["a"]), name="qwen3-4b-instruct")
    list(gen.stream("prompt", settings))
    assert gen.last_usage is None


def test_openvino_generator_wraps_pipeline_failures(settings):
    pipe = FakePipe([], error=RuntimeError("Exception from src/inference: out of memory"))
    gen = OpenVinoGenerator(pipe, name="qwen3-4b-instruct", device="GPU")

    with pytest.raises(GenerationError, match="qwen3-4b-instruct.*GPU"):
        list(gen.stream("prompt", settings))


def test_abandoning_the_stream_cancels_the_worker(settings):
    pipe = FakePipe(endless_pieces())
    gen = OpenVinoGenerator(FakePipe(endless_pieces()), name="qwen3-4b-instruct")
    gen.pipe = pipe

    it = gen.stream("prompt", settings)
    next(it)
    next(it)
    it.close()

    assert pipe.cancelled, "the streamer must return CANCEL once the consumer walks away"
    assert pipe.emitted < 2000


def test_generation_settings_map_onto_the_genai_config(settings):
    gen = OpenVinoGenerator(FakePipe([]), name="qwen3-4b-instruct")

    sampled = gen.make_config(settings)
    greedy = gen.make_config(GenerationSettings(max_new_tokens=8, temperature=0.0))

    assert sampled.max_new_tokens == 32
    assert sampled.do_sample and sampled.temperature == pytest.approx(0.2)
    assert not greedy.do_sample


def test_settings_come_from_config(cfg):
    settings = GenerationSettings.from_config(cfg)
    assert settings.max_new_tokens == cfg.generate.max_new_tokens
    assert settings.temperature == pytest.approx(cfg.generate.temperature)
    assert settings.max_prompt_tokens == cfg.generate.max_prompt_tokens


def test_settings_reject_nonsense():
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationSettings(max_new_tokens=0, temperature=0.2)
    with pytest.raises(ValueError, match="temperature"):
        GenerationSettings(max_new_tokens=8, temperature=-1.0)
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        GenerationSettings(max_new_tokens=8, temperature=0.2, max_prompt_tokens=0)
