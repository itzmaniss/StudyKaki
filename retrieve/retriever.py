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
