"""Answer generation — ARCHITECTURE.md §4, §9, §1 (streaming), §0.6 (abstain).

The §4 flow, end to end:

    retrieve k -> if top score < tau ABSTAIN -> take n_context -> trim to the prompt budget
    -> prompt with numbered blocks -> stream -> cite.verify() strips invented [n] -> Answer

Two rules shape the code more than anything else:

**Abstain beats hallucinate (§0.6).** Below `tau` no model runs at all — there is nothing
to ground an answer in, and a fluent guess is the worst possible output for a study tool.
The model is also told to reply with the abstain message when the context does not cover
the question (`answer/prompt.py` rule 5); when it does, we take it at its word and drop the
citations rather than shipping an "answer" that says it found nothing but points at pages.

**Trimming beats failing (BLOCKERS #11).** `n_context` is a request, not a guarantee. OpenVINO's
INT4 CPU MatMul cannot build a primitive descriptor much above 7k prompt tokens, and Tamil hits
that first — it tokenizes at ~1.1 chars/token against English's 2.33, so five Tamil chunks are
~10.4k tokens where five English ones are ~4.7k. Context is therefore trimmed from the tail to
`generate.max_prompt_tokens` before the model is ever called. An answer from three chunks is
worth more than an exception from five.

**The pipeline is injectable.** `StreamingGenerator` is a Protocol, so every test here runs
against a fake and none need the 2.5 GB INT4 IR. `OpenVinoGenerator` takes its `LLMPipeline`
as a constructor argument for the same reason: the streaming machinery — worker thread,
queue, cancellation, token accounting — is testable without a converted model.

Tier 3 (§9) is reached only when Tier 1 abstained *and* the user explicitly opted in. Its
wording, disclaimer and marker-stripping live in `answer/prompt.py`; this module only
decides when it is allowed to run.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from answer.cite import find_markers, verify
from answer.prompt import (
    ABSTAIN_MESSAGE,
    TIER3_DISCLAIMER,
    FittedContext,
    abstain_answer,
    build_tier3_prompt,
    estimate_tokens,
    fit_context,
    render_tier3_answer,
)
from core.config import Config
from core.schema import Answer, Retrieved
from core.telemetry import (
    TIER_LABELS,
    TIER_LOCAL_INDEX,
    TIER_PARAMETRIC,
    QueryTrace,
    TraceRecorder,
)
from ingest.normalize import detect_script, lang_for_script
from retrieve.retriever import Retriever, abstains_for

log = structlog.get_logger("answer.generate")

CPU = "CPU"

STAGE_RETRIEVE = "retrieve"
STAGE_GENERATE = "generate"
STAGE_GENERATE_TIER3 = "generate_tier3"

#: How long to wait for the worker thread after the consumer abandons the stream. The
#: streamer callback checks the cancel flag between tokens, so one token of decode is the
#: worst case; the timeout only stops a wedged pipeline from hanging the UI.
_JOIN_TIMEOUT_S = 30.0

_DONE = object()


class GenerationError(RuntimeError):
    """Generation failed. Always names the model and the device it was running on."""


@dataclass(frozen=True)
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    def __post_init__(self) -> None:
        if self.prompt < 0 or self.completion < 0:
            raise ValueError(f"token counts must be >= 0, got {self}")


@dataclass(frozen=True)
class GenerationSettings:
    """§6 `generate:` — nothing about decoding is hardcoded outside `configs/base.yaml`."""

    max_new_tokens: int
    temperature: float
    #: Prompt-side ceiling (BLOCKERS #11). Context is trimmed to fit before generation.
    max_prompt_tokens: int = 6000

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_prompt_tokens <= 0:
            raise ValueError(f"max_prompt_tokens must be > 0, got {self.max_prompt_tokens}")

    @classmethod
    def from_config(cls, cfg: Config) -> GenerationSettings:
        return cls(
            max_new_tokens=cfg.generate.max_new_tokens,
            temperature=cfg.generate.temperature,
            max_prompt_tokens=cfg.generate.max_prompt_tokens,
        )


@runtime_checkable
class StreamingGenerator(Protocol):
    """Anything that can stream text. The seam that keeps this module testable.

    `requested_device` is what `configs/base.yaml` asked for and `device` is what the load
    actually got — both are reported so a silent CPU fallback (§7.4) shows up in telemetry.

    Optional: a `last_usage: TokenUsage | None` attribute, set when a stream completes. It
    is read with `getattr`, so a generator that cannot count tokens simply omits it and the
    trace records zero rather than an invented number (§0.5).

    Optional: a `count_tokens(text) -> int` method, used to hold the prompt under
    `max_prompt_tokens` (BLOCKERS #11). Also read with `getattr`; a generator that omits it
    falls back to `answer.prompt.estimate_tokens`.
    """

    name: str
    requested_device: str
    device: str

    def stream(self, prompt: str, settings: GenerationSettings) -> Iterator[str]:
        """Yield the answer in pieces, as they are decoded. Never buffer the whole thing."""
        ...


class OpenVinoGenerator:
    """OpenVINO GenAI `LLMPipeline`, streamed (§1, §4).

    `LLMPipeline.generate` is blocking and pushes tokens through a callback, so it runs on a
    worker thread and the callback feeds a queue this iterator drains. That is what makes
    generation *incremental* to the caller — the UI can paint the first token instead of
    waiting for 512 of them, which is the whole point of the TTFT number in §5.

    The pipeline object is injected rather than built in `__init__`: `load_generator` builds
    the real one, and tests pass anything exposing
    `generate(prompt, generation_config, streamer)`.
    """

    def __init__(
        self,
        pipe: Any,
        *,
        name: str,
        requested_device: str = CPU,
        device: str = CPU,
    ) -> None:
        self.pipe = pipe
        self.name = name
        self.requested_device = requested_device
        self.device = device
        self.last_usage: TokenUsage | None = None

    def count_tokens(self, text: str) -> int:
        """Exact prompt length, from the pipeline's own tokenizer.

        This is what makes the BLOCKERS #11 budget trustworthy: the character-rate estimate
        is calibrated on averages, while this is the number the MatMul will actually see. A
        GenAI build that exposes no tokenizer falls back to the estimate — an approximate
        budget still beats the failure it prevents.
        """
        try:
            ids = self.pipe.get_tokenizer().encode(text).input_ids
            return int(ids.get_shape()[-1])
        except (AttributeError, IndexError, RuntimeError, TypeError) as e:
            log.warning("generator.tokenizer_unavailable", model=self.name, error=str(e))
            return estimate_tokens(text)

    def make_config(self, settings: GenerationSettings) -> Any:
        genai = _genai()
        cfg = genai.GenerationConfig()
        cfg.max_new_tokens = settings.max_new_tokens
        # temperature 0 means "be reproducible", which is greedy decoding, not sampling with
        # a degenerate temperature — the latter is a division by zero in most kernels.
        if settings.temperature > 0.0:
            cfg.do_sample = True
            cfg.temperature = settings.temperature
        else:
            cfg.do_sample = False
        cfg.validate()
        return cfg

    def stream(self, prompt: str, settings: GenerationSettings) -> Iterator[str]:
        genai = _genai()
        gen_cfg = self.make_config(settings)
        running = int(genai.StreamingStatus.RUNNING)
        cancelled = int(genai.StreamingStatus.CANCEL)

        # Unbounded by design: it can hold at most `max_new_tokens` pieces, and a bounded
        # queue would let a slow consumer block the worker inside the callback, where the
        # cancel flag can no longer be checked.
        pieces: queue.Queue[Any] = queue.Queue()
        cancel = threading.Event()
        outcome: dict[str, Any] = {}

        def streamer(piece: str) -> int:
            if cancel.is_set():
                return cancelled
            pieces.put(piece)
            return running

        def work() -> None:
            try:
                # Keyword, not positional (BLOCKERS #16): LLMPipeline's only overload names
                # these two the same, so this is a no-op there — but VLMPipeline has 9
                # overloads and none of them accept 3 positional args for text-only input;
                # its pure-text overload is `generate(self, prompt, **kwargs)`.
                outcome["result"] = self.pipe.generate(
                    prompt, generation_config=gen_cfg, streamer=streamer
                )
            except Exception as e:
                outcome["error"] = e
            finally:
                pieces.put(_DONE)

        worker = threading.Thread(target=work, name=f"generate-{self.name}", daemon=True)
        worker.start()
        try:
            while True:
                item = pieces.get()
                if item is _DONE:
                    break
                yield item
        finally:
            cancel.set()
            worker.join(timeout=_JOIN_TIMEOUT_S)

        error = outcome.get("error")
        if error is not None:
            raise GenerationError(
                f"{self.name}: generation failed on {self.device}: {error}"
            ) from error
        self.last_usage = _usage_from(outcome.get("result"))


def load_generator(
    cfg: Config,
    *,
    manifest_path: str | Path | None = None,
) -> OpenVinoGenerator:
    """Build the configured generator, with the §7.4 device fallback and the §7.2 cache.

    `models/registry.py` compiles an `ov.CompiledModel`; `LLMPipeline` wants the IR
    directory and a device string instead, so the manifest lookup and fallback are done
    here against the same manifest, not duplicated policy.

    `entry.is_vlm` (BLOCKERS #16) picks `VLMPipeline` over `LLMPipeline` for a multi-part
    IR (e.g. `gemma-4-e2b-it`, any-to-any so the language tower does not stand alone). The
    two constructors are not call-compatible: `LLMPipeline` takes `config` as its third
    *positional* argument, but `VLMPipeline`'s text-only overload is
    `(models_path, device, **kwargs)` — a positional third argument does not match any of
    its nine overloads and raises `TypeError` before the model ever loads. Device
    properties are therefore passed as `**ov_config` for `VLMPipeline`.
    """
    from models.registry import ModelNotFound, load_manifest, ov_version, select_device, spec_for

    genai = _genai()
    _, spec = spec_for("generator", cfg)
    entry = load_manifest(manifest_path).by_name(spec.name)

    if entry.precision != spec.precision:
        raise GenerationError(
            f"{entry.name}: config asks for {spec.precision} but manifest holds "
            f"{entry.precision} — re-run models/convert.py or fix configs/base.yaml"
        )
    if not entry.is_converted:
        raise ModelNotFound(
            f"{entry.name}: IR missing at {entry.ir_dir} (weights are not committed, §7.3) — "
            f"run `uv run python -m scripts.setup`"
        )
    if entry.ov_version and entry.ov_version != ov_version():
        log.warning(
            "generator.ov_version_drift",
            model=entry.name,
            ir_built_with=entry.ov_version,
            runtime=ov_version(),
            hint="§7.1 — align openvino, openvino-tokenizers and openvino-genai first",
        )

    ov_config: dict[str, Any] = {}
    cache_dir = cfg.resolve(cfg.paths.ov_cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ov_config["CACHE_DIR"] = str(cache_dir)
    except OSError as e:
        log.warning("generator.cache_dir_unwritable", cache_dir=str(cache_dir), error=str(e))

    requested = select_device(spec.device)
    last_error: Exception | None = None
    for device in dict.fromkeys((requested, CPU)):
        try:
            if entry.is_vlm:
                pipe = genai.VLMPipeline(str(entry.ir_dir), device, **ov_config)
            else:
                pipe = genai.LLMPipeline(str(entry.ir_dir), device, ov_config)
        except (RuntimeError, OSError) as e:
            last_error = e
            log.warning("generator.load_failed", model=entry.name, device=device, error=str(e))
            continue
        log.info(
            "generator.loaded",
            model=entry.name,
            precision=entry.precision,
            requested_device=spec.device,
            device=device,
            fell_back=device != spec.device,
            pipeline="VLMPipeline" if entry.is_vlm else "LLMPipeline",
        )
        return OpenVinoGenerator(pipe, name=entry.name, requested_device=spec.device, device=device)

    raise GenerationError(
        f"{entry.name}: LLMPipeline would not load on {requested} or {CPU} — "
        f"last error: {last_error}"
    )


@dataclass(frozen=True)
class AnswerResult:
    """Everything the UI needs to render one answer without making a decision of its own.

    `tier` is mandatory rather than derived: §9 requires the answer tier to be visible on
    every answer, and `Answer` carries no tier field.
    """

    answer: Answer
    tier: int
    hits: tuple[Retrieved, ...]
    context: tuple[Retrieved, ...]
    trace: QueryTrace
    #: Citation markers the model wrote, and how many survived `cite.verify`. Counted here
    #: because `answer.text` is already cleaned — by the time anyone else sees it the invented
    #: markers are gone, and the difference between the two is the whole groundedness signal.
    markers_emitted: int = 0
    markers_grounded: int = 0

    @property
    def groundedness(self) -> float:
        """Fraction of the model's citation markers that pointed at real retrieved context.

        A non-abstaining answer that cited nothing scores 0.0: §4 requires every claim to
        carry a citation, so an uncited assertion is the exact failure this number exists to
        catch. Abstentions make no claim and should be excluded by the caller, not scored.
        """
        if not self.markers_emitted:
            return 0.0
        return self.markers_grounded / self.markers_emitted

    @property
    def tier_label(self) -> str:
        return TIER_LABELS[self.tier]

    @property
    def abstained(self) -> bool:
        return self.answer.abstained


class AnswerStream:
    """One query's answer, streamed. Iterate for deltas, then read `.result`.

    Deltas are **provisional**: they are raw model text, so an invented `[7]` can appear in
    a delta and be gone from the final answer once `answer/cite.py` has run. Render deltas
    live for responsiveness, then replace them with `result.answer.text`, which is the only
    verified form. Never assemble an `Answer` out of deltas.

    The stream owns the `TraceRecorder`: it records stages, device, tokens and TTFT, and
    calls `finish()` at the end. Pass your own recorder to set `lang`/`config_hash`/
    `trace_id` or to time earlier work — but do not finish it yourself.
    """

    def __init__(
        self,
        question: str,
        *,
        generator: StreamingGenerator,
        cfg: Config,
        hits: Sequence[Retrieved] | None = None,
        retriever: Retriever | None = None,
        tier3_enabled: bool = False,
        doc_names: Mapping[str, str] | None = None,
        trace: TraceRecorder | None = None,
    ) -> None:
        if (hits is None) == (retriever is None):
            raise ValueError("pass exactly one of hits= (already retrieved) or retriever=")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        self.question = question
        self._generator = generator
        self._cfg = cfg
        self._settings = GenerationSettings.from_config(cfg)
        self._hits = None if hits is None else list(hits)
        self._retriever = retriever
        self._tier3_enabled = tier3_enabled
        self._doc_names = doc_names
        self._recorder = trace or TraceRecorder(
            question,
            lang=lang_for_script(detect_script(question)),
            config_hash=cfg.config_hash,
        )
        self._result: AnswerResult | None = None
        self._started = False
        self._streamed = False
        self._device_recorded = False

    @property
    def trace_id(self) -> str:
        return self._recorder.trace_id

    @property
    def result(self) -> AnswerResult:
        if self._result is None:
            raise RuntimeError(
                "answer is not final — iterate the stream to completion before reading .result"
            )
        return self._result

    def __iter__(self) -> Iterator[str]:
        if self._started:
            raise RuntimeError("AnswerStream is single-use; build a new one per query")
        self._started = True
        return self._run()

    def _run(self) -> Iterator[str]:
        rec = self._recorder
        hits = self._hits
        retriever = self._retriever
        if hits is None and retriever is not None:
            with rec.stage(STAGE_RETRIEVE):
                hits = list(retriever.retrieve(self.question, self._cfg.retrieve.k))
        hits = hits or []
        rec.record_retrieval(hits)

        context: list[Retrieved] = []
        answer: Answer | None = None
        tier = TIER_LOCAL_INDEX
        emitted = grounded = 0

        if not abstains_for(hits, self._cfg.retrieve.tau, self._retriever):
            fitted = self._fit_context(hits)
            context = list(fitted.hits)
            raw = yield from self._generate(fitted.prompt, STAGE_GENERATE)
            text, citations = verify(raw, context)
            emitted, grounded = len(find_markers(raw)), len(find_markers(text))
            text = text.strip()
            if text and not _is_abstain_text(text):
                answer = Answer(
                    text=text, citations=citations, abstained=False, trace_id=rec.trace_id
                )

        if answer is None and self._tier3_enabled:
            # §9: Tier 1 abstained, so say so before offering ungrounded knowledge.
            if not self._streamed:
                yield f"{ABSTAIN_MESSAGE}\n\n"
            yield f"{TIER3_DISCLAIMER}\n\n"
            raw = yield from self._generate(
                build_tier3_prompt(self.question, enabled=True), STAGE_GENERATE_TIER3
            )
            answer = render_tier3_answer(raw, rec.trace_id, enabled=True, tier1_abstained=True)
            tier = TIER_PARAMETRIC
            context = []

        if answer is None:
            answer = abstain_answer(rec.trace_id)
            if not self._streamed:
                yield answer.text

        trace = rec.finish(answer, tier=tier, abstained=answer.abstained)
        self._result = AnswerResult(
            markers_emitted=emitted,
            markers_grounded=grounded,
            answer=answer,
            tier=tier,
            hits=tuple(hits),
            context=tuple(context),
            trace=trace,
        )

    def _fit_context(self, hits: Sequence[Retrieved]) -> FittedContext:
        """Take `n_context` blocks, then drop from the tail until the prompt fits (#11)."""
        requested = self._cfg.retrieve.n_context
        budget = self._settings.max_prompt_tokens
        fitted = fit_context(
            self.question,
            hits[:requested],
            max_prompt_tokens=budget,
            count_tokens=getattr(self._generator, "count_tokens", estimate_tokens),
            doc_names=self._doc_names,
        )
        if fitted.dropped:
            log.info(
                "generate.context_trimmed",
                requested=requested,
                kept=len(fitted.hits),
                dropped=fitted.dropped,
                prompt_tokens=fitted.tokens,
                budget=budget,
            )
        if fitted.over_budget:
            log.warning(
                "generate.prompt_over_budget",
                prompt_tokens=fitted.tokens,
                budget=budget,
                hint="a single chunk exceeds the budget; generation may fail (BLOCKERS #11)",
            )
        return fitted

    def _generate(self, prompt: str, stage: str) -> Iterator[str]:
        """Stream one generation pass, returning the raw (unverified) text it produced."""
        rec = self._recorder
        gen = self._generator
        if not self._device_recorded:
            rec.record_device(gen.name, gen.requested_device, gen.device)
            self._device_recorded = True

        parts: list[str] = []
        with rec.stage(stage):
            for delta in gen.stream(prompt, self._settings):
                if not isinstance(delta, str):
                    raise GenerationError(
                        f"{gen.name}: stream yielded {type(delta).__name__}, expected str"
                    )
                if not delta:
                    continue
                if not parts:
                    rec.mark_first_token()
                self._streamed = True
                parts.append(delta)
                yield delta

        usage = getattr(gen, "last_usage", None)
        if isinstance(usage, TokenUsage):
            rec.record_tokens(prompt=usage.prompt, completion=usage.completion)
        return "".join(parts)


def stream_answer(
    question: str,
    *,
    generator: StreamingGenerator,
    cfg: Config,
    hits: Sequence[Retrieved] | None = None,
    retriever: Retriever | None = None,
    tier3_enabled: bool = False,
    doc_names: Mapping[str, str] | None = None,
    trace: TraceRecorder | None = None,
) -> AnswerStream:
    """The §4 answer path. Pass `hits=` if you already retrieved, `retriever=` otherwise."""
    return AnswerStream(
        question,
        generator=generator,
        cfg=cfg,
        hits=hits,
        retriever=retriever,
        tier3_enabled=tier3_enabled,
        doc_names=doc_names,
        trace=trace,
    )


def generate_answer(
    question: str,
    *,
    generator: StreamingGenerator,
    cfg: Config,
    hits: Sequence[Retrieved] | None = None,
    retriever: Retriever | None = None,
    tier3_enabled: bool = False,
    doc_names: Mapping[str, str] | None = None,
    trace: TraceRecorder | None = None,
) -> AnswerResult:
    """`stream_answer` drained to completion, for callers that do not render token by token."""
    stream = stream_answer(
        question,
        generator=generator,
        cfg=cfg,
        hits=hits,
        retriever=retriever,
        tier3_enabled=tier3_enabled,
        doc_names=doc_names,
        trace=trace,
    )
    for _ in stream:
        pass
    return stream.result


def _is_abstain_text(text: str) -> bool:
    """Did the model use its §4 escape hatch?

    `answer/prompt.py` rule 5 tells the model to reply with exactly `ABSTAIN_MESSAGE` when
    the context does not answer the question. Models wrap it in quotes or drop the full
    stop, so compare loosely — but only against that one sentence, never against a general
    notion of uncertainty.
    """
    return _normalise(text) == _normalise(ABSTAIN_MESSAGE)


def _normalise(text: str) -> str:
    return text.strip().strip("\"'“”‘’").strip().rstrip(".").casefold()


def _usage_from(result: Any) -> TokenUsage | None:
    """Token counts from GenAI's own perf counters, or nothing at all.

    §0.5: a fabricated number is worse than a missing one, so a build that reports no
    metrics leaves the trace's token columns at zero rather than guessing from piece counts
    (a streamed piece is not reliably one token).
    """
    metrics = getattr(result, "perf_metrics", None)
    if metrics is None:
        return None
    try:
        return TokenUsage(
            prompt=int(metrics.get_num_input_tokens()),
            completion=int(metrics.get_num_generated_tokens()),
        )
    except (AttributeError, TypeError, ValueError) as e:
        log.warning("generator.usage_unavailable", error=str(e))
        return None


def _genai() -> Any:
    """Imported lazily so `answer.generate` (and its tests) load without OpenVINO GenAI."""
    import openvino_genai

    return openvino_genai
