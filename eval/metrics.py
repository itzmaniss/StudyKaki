"""Retrieval metrics — ARCHITECTURE.md §5.

A hit is relevant when it comes from the gold document *and* overlaps a gold page.
Page-level rather than chunk-level, because gold labels are pages (`gold_pages`).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schema import Retrieved


@dataclass(frozen=True)
class AltSource:
    """Another document carrying the same answer, and the pages it answers on *there*.

    Page numbers are per document on purpose. Parallel editions are not reliably
    page-aligned: `itu_wtdc22_{en,zh}` and `std12_cs_vol2_{en,ta}` line up at offset 0, but
    `std12_cs_vol1_ta` runs one page longer than its English twin and drifts up to three pages
    by the back of the book. Sharing one page list across both would score a correct hit as a
    miss and an unrelated hit as correct.
    """

    doc_id: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class GoldQuestion:
    q: str
    lang: str
    doc_id: str
    gold_pages: list[int]
    note: str = ""
    #: §5's shape is one `doc_id`, and that stays the primary. This is the extension for
    #: parallel translations, where the same answer genuinely lives in two documents and
    #: matching only one scores a retriever as wrong for being right in the other language.
    alt_sources: tuple[AltSource, ...] = ()

    def __post_init__(self) -> None:
        if self.alt_sources and self.unanswerable:
            raise ValueError(
                f"{self.q!r}: unanswerable questions cannot carry alt_doc_ids — empty "
                f"gold_pages means the system must abstain, so no document answers it"
            )
        seen = {self.doc_id}
        for alt in self.alt_sources:
            if alt.doc_id in seen:
                raise ValueError(f"{self.q!r}: duplicate doc_id in alt_doc_ids: {alt.doc_id}")
            seen.add(alt.doc_id)

    @property
    def unanswerable(self) -> bool:
        """No gold page means the system is expected to abstain (§5)."""
        return not self.gold_pages

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Every document that answers this question, primary first."""
        return (self.doc_id, *(a.doc_id for a in self.alt_sources))

    def pages_for(self, doc_id: str) -> tuple[int, ...] | None:
        """Gold pages within `doc_id`, or None if that document does not answer this."""
        if doc_id == self.doc_id:
            return tuple(self.gold_pages)
        for alt in self.alt_sources:
            if alt.doc_id == doc_id:
                return alt.pages
        return None

    @classmethod
    def from_row(cls, row: dict) -> GoldQuestion:
        missing = {"q", "lang", "doc_id"} - row.keys()
        if missing:
            raise ValueError(f"golden row missing required keys {sorted(missing)}: {row}")
        gold_pages = list(row.get("gold_pages") or [])
        return cls(
            q=row["q"],
            lang=row["lang"],
            doc_id=row["doc_id"],
            gold_pages=gold_pages,
            note=row.get("note", ""),
            alt_sources=_alt_sources(row.get("alt_doc_ids"), gold_pages, row),
        )


def _alt_sources(raw: object, gold_pages: list[int], row: dict) -> tuple[AltSource, ...]:
    """Parse `alt_doc_ids`, which takes either shape:

        "alt_doc_ids": ["<doc_id>"]              # same pages as gold_pages (aligned editions)
        "alt_doc_ids": {"<doc_id>": [68, 69]}    # its own pages (editions that drift)

    The list form is the terse common case; the object form is what a drifting translation
    needs. Anything else is rejected rather than guessed at — a silently misparsed label is
    an eval number that is wrong in a way nobody can see.
    """
    if raw is None:
        return ()
    if isinstance(raw, list):
        if not all(isinstance(d, str) for d in raw):
            raise ValueError(f"alt_doc_ids list must hold doc_id strings: {row}")
        return tuple(AltSource(doc_id=d, pages=tuple(gold_pages)) for d in raw)
    if isinstance(raw, dict):
        out = []
        for doc_id, pages in raw.items():
            if not isinstance(pages, list) or not all(isinstance(p, int) for p in pages):
                raise ValueError(f"alt_doc_ids[{doc_id!r}] must be a list of page ints: {row}")
            out.append(AltSource(doc_id=doc_id, pages=tuple(pages)))
        return tuple(out)
    raise ValueError(f"alt_doc_ids must be a list or an object, got {type(raw).__name__}: {row}")


def is_relevant(hit: Retrieved, gold: GoldQuestion) -> bool:
    pages = gold.pages_for(hit.chunk.doc_id)
    if pages is None:
        return False
    span = range(hit.chunk.page_start, hit.chunk.page_end + 1)
    return any(p in span for p in pages)


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


def groundedness(scores: list[float]) -> float:
    """Mean per-answer groundedness (§5).

    Each score is `AnswerResult.groundedness` — the fraction of the citation markers the model
    wrote that survived `answer/cite.py`'s check against the context it was actually given.
    Abstentions are excluded by the caller: refusing to answer is not an ungrounded claim, and
    scoring it as one would make abstaining look worse than inventing a citation.
    """
    return mean(scores)
