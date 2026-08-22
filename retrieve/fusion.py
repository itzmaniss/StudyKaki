"""Rank fusion — ARCHITECTURE.md §1 ("interface exists even if only dense is wired"), §10.

V1 wires **dense retrieval only**. This module exists so the seam is real rather than
imagined: `fuse()` is what `retrieve/dense.py` output goes through today, and it already
takes a list of ranked lists rather than one.

**What would plug in here (V2, §10 — not built, gated on tonight's eval numbers).**
`retrieve/lexical.py` produces a BM25 `RankedList` over the same chunk corpus and is passed
alongside the dense list. Nothing in this file changes when that happens. Per §10 the real
work there is per-script tokenization (jieba for zh, fugashi for ja, pythainlp for th, ICU
fallback; Latin/Tamil/Devanagari split on whitespace) — not the BM25 scoring itself. §11 and
CLAUDE.md forbid building it now, so this file deliberately contains no lexical retrieval.

**The score-units trap.** RRF scores are derived from ranks, not similarities: the best
possible RRF score for a single list is `1/(60+1)` ≈ 0.016, far below `cfg.retrieve.tau`
(0.35). Feeding RRF output to `retriever.abstains(hits, tau)` would therefore abstain on
every query. `FusionResult.scores_are_similarities` says which units you are holding, and
the abstain decision stays on the dense list (see `retrieve/dense.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from core.schema import Retrieved

#: §10 — reciprocal rank fusion constant. Damps the influence of any single list's top hit.
RRF_K = 60


@dataclass(frozen=True)
class RankedList:
    """One retriever's output, tagged so traces can say which arm found a chunk."""

    name: str
    hits: list[Retrieved] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a ranked list must be named — traces attribute hits by name")
        if self.weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")


@dataclass(frozen=True)
class FusionResult:
    """Fused hits plus the one thing a caller must not guess: what the scores mean."""

    hits: list[Retrieved]
    method: str
    scores_are_similarities: bool
    sources: tuple[str, ...] = ()


def reciprocal_rank_fusion(
    lists: Sequence[RankedList],
    *,
    rrf_k: int = RRF_K,
    k: int | None = None,
) -> list[Retrieved]:
    """Fuse ranked lists by `sum(weight / (rrf_k + rank))`, deduplicated on `chunk_id`.

    Ranks come from `Retrieved.rank`, not list position, so a sliced list keeps its true
    ranks. Returned scores are RRF scores — **not** similarities; see the module docstring.
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

    scores: dict[str, float] = {}
    chunks: dict[str, Retrieved] = {}
    order: dict[str, int] = {}

    for ranked in lists:
        for hit in ranked.hits:
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + ranked.weight / (rrf_k + hit.rank)
            if cid not in chunks:
                chunks[cid] = hit
                order[cid] = len(order)

    # Ties broken by first-seen order so the same inputs always give the same output.
    ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], order[cid]))
    if k is not None:
        ranked_ids = ranked_ids[:k]

    return [
        Retrieved(chunk=chunks[cid].chunk, score=scores[cid], rank=i)
        for i, cid in enumerate(ranked_ids, start=1)
    ]


def fuse(
    lists: Sequence[RankedList],
    *,
    k: int | None = None,
    rrf_k: int = RRF_K,
) -> FusionResult:
    """Pipeline entry point. One list passes through untouched; two or more get RRF.

    The passthrough is not a special case for its own sake — it is what keeps V1 honest.
    With only dense wired, scores stay cosine similarities and `cfg.retrieve.tau` still
    means what §4 says it means. The day a second arm appears, `scores_are_similarities`
    flips to False and the caller is forced to notice.
    """
    if not lists:
        raise ValueError("fuse() needs at least one ranked list — zero means nothing is wired")

    names = tuple(ranked.name for ranked in lists)
    if len(set(names)) != len(names):
        raise ValueError(f"ranked list names must be unique, got {names}")

    if len(lists) == 1:
        hits = lists[0].hits
        if k is not None:
            hits = hits[:k]
        return FusionResult(
            hits=list(hits),
            method="passthrough",
            scores_are_similarities=True,
            sources=names,
        )

    return FusionResult(
        hits=reciprocal_rank_fusion(lists, rrf_k=rrf_k, k=k),
        method="rrf",
        scores_are_similarities=False,
        sources=names,
    )
