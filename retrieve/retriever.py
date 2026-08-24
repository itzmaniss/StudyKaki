"""Retriever interface — ARCHITECTURE.md §4.

The eval harness talks to this Protocol only, so a random stub and the real dense
retriever are interchangeable and the harness itself can be trusted before any model exists.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.schema import Retrieved


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        """Return up to `k` hits, rank 1 = best, sorted by descending score."""
        ...


def abstains(hits: list[Retrieved], tau: float) -> bool:
    """§4: if the top score is below tau, abstain. Abstain beats hallucinate (§0.6)."""
    return not hits or hits[0].score < tau


def abstain_score(hits: list[Retrieved], retriever: object | None = None) -> float:
    """The score to compare against tau — always on the cosine scale (BLOCKERS #12).

    `tau` is calibrated against dense cosine similarity and its safe window is 0.01 wide
    (BLOCKERS #8). An arm that returns scores in other units — RRF, cross-encoder logits —
    must therefore hand back a cosine score for the abstain decision or the gate is
    meaningless. Arms do that by exposing `abstain_top_score`.
    """
    override = getattr(retriever, "abstain_top_score", None)
    if override is not None:
        return float(override)
    return hits[0].score if hits else 0.0


def abstains_for(
    hits: list[Retrieved],
    tau: float,
    retriever: object | None = None,
) -> bool:
    """`abstains`, but asks the retriever what score the gate should read."""
    if not hits:
        return True
    return abstain_score(hits, retriever) < tau


def degraded_calls(retriever: object | None) -> int:
    """Queries where an arm silently fell back instead of doing its job.

    Exists because a silent fallback is indistinguishable from a real measurement in a
    results table: `TextRerankPipeline` threw on all 54 questions, the rerank arm degraded to
    its inner ranking, and the sweep printed the baseline number as if rerank had been
    evaluated. Any arm that can degrade counts it; `eval/sweep.py` refuses to report a row
    whose count is non-zero.
    """
    own = int(getattr(retriever, "degraded_calls", 0) or 0)
    inner = getattr(retriever, "inner", None)
    return own + (degraded_calls(inner) if inner is not None else 0)
