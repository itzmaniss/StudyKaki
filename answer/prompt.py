"""Prompt construction and the answer-tier contract — ARCHITECTURE.md §4, §9, §0.6.

Tier 1 (local index) is the default and the only grounded path. Tier 3 (model parametric
knowledge) lives here as pure helpers because it is a *presentation* contract, not a model
concern: it must be opt-in, must carry a fixed disclaimer, and must never emit a citation.

**Numbering contract, shared with `answer/cite.py`:** context blocks are numbered `[1]..[n]`
by position in the list passed here. `answer/cite.py` resolves `[n]` back to
`context_hits[n - 1]`, so both must be given the same list in the same order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from answer.cite import strip_all_markers
from core.schema import Answer, Chunk, Retrieved
from ingest.normalize import detect_script, lang_for_script, script_histogram

# §0.6 — the literal abstain text. Do not paraphrase; the UI and eval both match on it.
ABSTAIN_MESSAGE = "I couldn't find this in your documents."

# §9 — the literal Tier 3 disclaimer. Also the UI's signal that an answer is ungrounded,
# since `Answer` carries no tier field.
TIER3_DISCLAIMER = "General knowledge — not from your materials. May not match your syllabus."

# BLOCKERS #21 - the reminder names the language, so a script code is not enough. Only
# the three corpus languages are named; anything else gets no reminder rather than a
# guess, since telling a model to "answer in und" is worse than saying nothing.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ta": "Tamil",
}

SYSTEM_INSTRUCTION = f"""\
You are a study assistant. Answer using ONLY the numbered context blocks provided.

Rules:
1. Write your answer in the same language as the question, regardless of the language of \
the context. Translate what you quote if you have to.
2. Cite inline with the block number, like [1] or [2][3], right after the claim it supports.
3. Only cite block numbers that appear in the context. Never invent a block number.
4. Never state a page number that is not shown in a block header.
5. If the context does not answer the question, reply exactly "{ABSTAIN_MESSAGE}" and stop. \
Do not guess and do not answer from your own knowledge.
6. Be concise. No preamble."""

TIER3_SYSTEM_INSTRUCTION = """\
Answer from your own general knowledge. The student's own documents did not cover this.

Rules:
1. Write your answer in the same language as the question.
2. Do NOT cite anything. Never write a bracketed number like [1], and never mention a page \
number, a chapter, or a document — you have no source to point at.
3. Say plainly when you are unsure.
4. Be concise. No preamble."""


# Characters per token by script, measured on this corpus against the Qwen3-4B INT4
# tokenizer (BLOCKERS #11). Tamil is why this table exists: it packs roughly twice as many
# tokens into the same character count as English, so five Tamil chunks are ~10.4k tokens
# where five English ones are ~4.7k. Unmeasured scripts inherit the rate of the measured one
# they most resemble, biased dense — Devanagari and Thai price as Tamil, not as Latin.
_CHARS_PER_TOKEN: dict[str, float] = {
    "taml": 1.10,
    "deva": 1.10,
    "thai": 1.10,
    "han": 1.62,
    "hans": 1.62,
    "hant": 1.62,
    "kana": 1.62,
    "hang": 1.62,
    "jpan": 1.62,
    "kore": 1.62,
    "latn": 2.33,
    "cyrl": 2.33,
    "arab": 2.33,
}

# Digits, punctuation and whitespace carry no script; price them as Latin.
_DEFAULT_CHARS_PER_TOKEN = 2.33

# The table holds averages, so an individual passage can run denser than its script's mean.
# This estimate is only reached when no real tokenizer is available, and undercounting
# reintroduces the exact failure the budget exists to prevent — so it deliberately runs high.
_ESTIMATE_SAFETY = 1.15


class Tier3DisabledError(RuntimeError):
    """§9: Tier 3 defaults to off and is opt-in only.

    Raised when the ungrounded path is invoked without an explicit `enabled=True`, so no
    code path can leak unsourced content into a study tool by omission.
    """


def format_context(hits: Sequence[Retrieved], *, doc_names: Mapping[str, str] | None = None) -> str:
    """Numbered context blocks, each headed with its `doc / p.N` provenance (§4)."""
    checked = _check_hits(hits)
    return "\n\n".join(
        _format_block(hit, number, doc_names) for number, hit in enumerate(checked, start=1)
    )


def language_reminder_for(question: str) -> str:
    """`(Answer in X.)` for a question whose language we can name, else "" (BLOCKERS #21).

    Placed between the context and `Answer:` by `build_prompt`. The position is the whole
    point: `SYSTEM_INSTRUCTION` rule 1 already says this thousands of tokens earlier, and
    that distance is exactly what a weaker instruction-follower loses it over.
    """
    name = LANGUAGE_NAMES.get(lang_for_script(detect_script(question)))
    return f"(Answer in {name}.)" if name else ""


def build_prompt(
    question: str,
    hits: Sequence[Retrieved],
    *,
    doc_names: Mapping[str, str] | None = None,
    system: str = SYSTEM_INSTRUCTION,
    language_reminder: bool = False,
) -> str:
    """Tier 1 prompt. `hits` must already be sliced to `cfg.retrieve.n_context`."""
    q = _check_question(question)
    context = format_context(hits, doc_names=doc_names)
    tail = language_reminder_for(q) if language_reminder else ""
    tail = f"{tail}\n" if tail else ""
    return f"{system}\n\nContext:\n{context}\n\nQuestion: {q}\n{tail}Answer:"


def estimate_tokens(text: str) -> int:
    """Script-weighted token estimate, for when no real tokenizer is at hand.

    Prefer the generator's own `count_tokens`; this is the documented fallback. Counting
    whitespace words (`ingest.chunk.count_tokens`) is *not* an alternative here — it
    undercounts Tamil subwords roughly sevenfold, which is precisely how BLOCKERS #11 stayed
    invisible until generation crashed.
    """
    if not text:
        return 0
    counts = script_histogram(text)
    tokens = (len(text) - sum(counts.values())) / _DEFAULT_CHARS_PER_TOKEN
    for bucket, n in counts.items():
        tokens += n / _CHARS_PER_TOKEN.get(bucket, _DEFAULT_CHARS_PER_TOKEN)
    return math.ceil(tokens * _ESTIMATE_SAFETY)


@dataclass(frozen=True)
class FittedContext:
    """The context blocks that actually fit the prompt budget, and the prompt they build."""

    hits: tuple[Retrieved, ...]
    prompt: str
    tokens: int
    dropped: int
    budget: int

    @property
    def over_budget(self) -> bool:
        """True when even one block busts the budget — generation is attempted regardless."""
        return self.tokens > self.budget


def fit_context(
    question: str,
    hits: Sequence[Retrieved],
    *,
    max_prompt_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
    doc_names: Mapping[str, str] | None = None,
    system: str = SYSTEM_INSTRUCTION,
    language_reminder: bool = False,
) -> FittedContext:
    """Longest prefix of `hits` whose Tier 1 prompt fits `max_prompt_tokens` (BLOCKERS #11).

    Blocks are dropped from the tail, so the highest-scoring context is what survives. At
    least one block is always kept, and a lone block over budget is returned anyway for the
    caller to log and attempt: an answer from three chunks beats an exception from five.

    Truncating a block's *text* to fit is deliberately not done. It would keep the block count
    up at the cost of the §4 provenance contract — a citation would point at a page whose text
    the model was never shown.
    """
    if max_prompt_tokens <= 0:
        raise ValueError(f"max_prompt_tokens must be > 0, got {max_prompt_tokens}")
    checked = _check_hits(hits)
    kept = checked
    while True:
        prompt = build_prompt(
            question,
            kept,
            doc_names=doc_names,
            system=system,
            language_reminder=language_reminder,
        )
        tokens = count_tokens(prompt)
        if tokens <= max_prompt_tokens or len(kept) == 1:
            return FittedContext(
                hits=tuple(kept),
                prompt=prompt,
                tokens=tokens,
                dropped=len(checked) - len(kept),
                budget=max_prompt_tokens,
            )
        kept = kept[:-1]


def abstain_answer(trace_id: str) -> Answer:
    """§0.6 — below `tau` we say so and cite nothing. Abstain beats hallucinate."""
    return Answer(text=ABSTAIN_MESSAGE, citations=[], abstained=True, trace_id=_check_id(trace_id))


def build_tier3_prompt(question: str, *, enabled: bool = False) -> str:
    """§9 Tier 3 prompt. `enabled` must come from an explicit user opt-in."""
    if enabled is not True:
        raise Tier3DisabledError(
            "Tier 3 (model parametric knowledge) is off by default; pass enabled=True only "
            "when the user has explicitly opted in (ARCHITECTURE.md §9)."
        )
    q = _check_question(question)
    return f"{TIER3_SYSTEM_INSTRUCTION}\n\nQuestion: {q}\nAnswer:"


def render_tier3_answer(
    model_text: str,
    trace_id: str,
    *,
    enabled: bool = False,
    tier1_abstained: bool = False,
) -> Answer:
    """Wrap ungrounded model output in the §9 contract.

    The disclaimer is prepended verbatim, every citation marker is stripped (the prompt asks
    the model not to emit them; this makes it true), and when Tier 1 abstained we say so
    first so the student is never left thinking this came from their materials.

    `abstained` is False — an answer *was* produced. The presence of `TIER3_DISCLAIMER` is
    what marks it ungrounded, since `Answer` has no tier field.
    """
    if enabled is not True:
        raise Tier3DisabledError(
            "Tier 3 (model parametric knowledge) is off by default; pass enabled=True only "
            "when the user has explicitly opted in (ARCHITECTURE.md §9)."
        )
    if not isinstance(model_text, str):
        raise TypeError(f"model_text must be str, got {type(model_text).__name__}")

    body = strip_all_markers(model_text).strip()
    lines = [ABSTAIN_MESSAGE] if tier1_abstained else []
    lines.append(TIER3_DISCLAIMER)
    if body:
        lines.append(body)
    return Answer(
        text="\n\n".join(lines), citations=[], abstained=False, trace_id=_check_id(trace_id)
    )


def _format_block(hit: Retrieved, number: int, doc_names: Mapping[str, str] | None) -> str:
    return f"[{number}] {_provenance(hit.chunk, doc_names)}\n{hit.chunk.text}"


def _provenance(chunk: Chunk, doc_names: Mapping[str, str] | None) -> str:
    name = (doc_names or {}).get(chunk.doc_id) or _short_doc_id(chunk.doc_id)
    if chunk.page_start == chunk.page_end:
        pages = f"p.{chunk.page_start}"
    else:
        pages = f"pp.{chunk.page_start}-{chunk.page_end}"
    heading = " > ".join(h for h in chunk.heading_path if h)
    return f"{name} / {pages}" + (f" | {heading}" if heading else "")


def _short_doc_id(doc_id: str) -> str:
    return doc_id.removeprefix("sha256:")[:8]


def _check_hits(hits: Sequence[Retrieved]) -> list[Retrieved]:
    try:
        checked = list(hits)
    except TypeError as exc:
        raise TypeError(f"hits must be a sequence of Retrieved, got {type(hits).__name__}") from exc
    if not checked:
        raise ValueError("cannot build context from zero hits — abstain instead (§0.6)")
    for i, hit in enumerate(checked):
        if not isinstance(hit, Retrieved):
            raise TypeError(f"hits[{i}] must be Retrieved, got {type(hit).__name__}")
    return checked


def _check_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError(f"question must be str, got {type(question).__name__}")
    q = question.strip()
    if not q:
        raise ValueError("question must not be empty")
    return q


def _check_id(trace_id: str) -> str:
    if not isinstance(trace_id, str):
        raise TypeError(f"trace_id must be str, got {type(trace_id).__name__}")
    if not trace_id:
        raise ValueError("trace_id must not be empty")
    return trace_id
