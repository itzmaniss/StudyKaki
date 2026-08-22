"""Retrieval metrics — ARCHITECTURE.md §5.

A hit is relevant when it comes from the gold document *and* overlaps a gold page.
Page-level rather than chunk-level, because gold labels are pages (`gold_pages`).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schema import Retrieved


@dataclass(frozen=True)
class GoldQuestion:
    q: str
    lang: str
    doc_id: str
    gold_pages: list[int]
    note: str = ""

    @property
    def unanswerable(self) -> bool:
        """No gold page means the system is expected to abstain (§5)."""
        return not self.gold_pages

    @classmethod
    def from_row(cls, row: dict) -> GoldQuestion:
        missing = {"q", "lang", "doc_id"} - row.keys()
        if missing:
            raise ValueError(f"golden row missing required keys {sorted(missing)}: {row}")
        return cls(
            q=row["q"],
            lang=row["lang"],
            doc_id=row["doc_id"],
            gold_pages=list(row.get("gold_pages") or []),
            note=row.get("note", ""),
        )


def is_relevant(hit: Retrieved, gold: GoldQuestion) -> bool:
    if hit.chunk.doc_id != gold.doc_id:
        return False
    span = range(hit.chunk.page_start, hit.chunk.page_end + 1)
    return any(p in span for p in gold.gold_pages)


def first_relevant_rank(hits: list[Retrieved], gold: GoldQuestion) -> int | None:
    for hit in hits:
        if is_relevant(hit, gold):
            return hit.rank
    return None


def recall_at(hits: list[Retrieved], gold: GoldQuestion, k: int) -> float:
    rank = first_relevant_rank([h for h in hits if h.rank <= k], gold)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(hits: list[Retrieved], gold: GoldQuestion, k: int = 10) -> float:
    rank = first_relevant_rank([h for h in hits if h.rank <= k], gold)
    return 1.0 / rank if rank else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
