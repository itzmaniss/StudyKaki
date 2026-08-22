"""Prompt construction and the answer-tier contract — ARCHITECTURE.md §4, §9, §0.6.

Tier 1 (local index) is the default and the only grounded path. Tier 3 (model parametric
knowledge) lives here as pure helpers because it is a *presentation* contract, not a model
concern: it must be opt-in, must carry a fixed disclaimer, and must never emit a citation.

**Numbering contract, shared with `answer/cite.py`:** context blocks are numbered `[1]..[n]`
by position in the list passed here. `answer/cite.py` resolves `[n]` back to
`context_hits[n - 1]`, so both must be given the same list in the same order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from answer.cite import strip_all_markers
from core.schema import Answer, Chunk, Retrieved

# §0.6 — the literal abstain text. Do not paraphrase; the UI and eval both match on it.
ABSTAIN_MESSAGE = "I couldn't find this in your documents."

# §9 — the literal Tier 3 disclaimer. Also the UI's signal that an answer is ungrounded,
# since `Answer` carries no tier field.
TIER3_DISCLAIMER = "General knowledge — not from your materials. May not match your syllabus."

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


def build_prompt(
    question: str,
    hits: Sequence[Retrieved],
    *,
    doc_names: Mapping[str, str] | None = None,
    system: str = SYSTEM_INSTRUCTION,
) -> str:
    """Tier 1 prompt. `hits` must already be sliced to `cfg.retrieve.n_context`."""
    q = _check_question(question)
    context = format_context(hits, doc_names=doc_names)
    return f"{system}\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"


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
