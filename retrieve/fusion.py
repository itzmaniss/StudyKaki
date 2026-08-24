"""Rank fusion — ARCHITECTURE.md §1 ("interface exists even if only dense is wired"), §10.

V1 wires **dense retrieval only**. This module exists so the seam is real rather than
imagined: `fuse()` is what `retrieve/dense.py` output goes through today, and it already
takes a list of ranked lists rather than one.

**V2 (§10) plugged in as designed.** `retrieve/lexical.py` produces a BM25 `RankedList` over
the same chunk corpus and is passed alongside the dense list; nothing in the fusion arithmetic
below changed when it arrived. `HybridRetriever` at the bottom of this file is the composition,
off unless `retrieve.hybrid.enabled`.

**The score-units trap — and why `HybridRetriever` cannot yet answer.** RRF scores are derived
from ranks, not similarities: the best possible RRF score for a single list is `1/(60+1)`
≈ 0.016, far below `cfg.retrieve.tau` (0.45). Feeding RRF output to
`retriever.abstains(hits, tau)` would abstain on every query.
`FusionResult.scores_are_similarities` says which units you are holding.

This makes the hybrid arm immediately measurable on `recall@k` and `mrr@10`, which are pure
rank metrics and never look at `score` — and *not* yet safe to put behind `answer/generate.py`,
whose abstain gate is calibrated on cosine (BLOCKERS #8: tau's safe window is 0.01 wide). The
gap is recorded as BLOCKERS #12; `last_dense_top_score` exposes what an abstain decision would
need, but nothing consumes it yet.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from core.config import Config
from core.schema import Retrieved
from retrieve.retriever import Retriever

log = structlog.get_logger("retrieve.fusion")

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


class HybridRetriever:
    """Dense + BM25, fused by RRF — the §10 hybrid arm.

    Both arms are asked for `k` hits and fused down to `k`, rather than each being asked for
    `k/2`. A chunk that both arms rank highly should be able to beat one that only dense found,
    and that comparison cannot happen if the lists are truncated before fusion.

    Scores on the returned hits are **RRF scores, not similarities** — see the module
    docstring. `last_dense_top_score` carries the cosine score an abstain decision would need.
    """

    @classmethod
    def open(
        cls,
        cfg: Config,
        index_path: str | Path | None = None,
        *,
        models_manifest_path: str | Path | None = None,
    ) -> HybridRetriever:
        """Production wiring. Both arms open the same index directory, by construction."""
        from retrieve.dense import DenseRetriever, default_index_dir
        from retrieve.lexical import LexicalRetriever

        resolved = Path(index_path) if index_path is not None else default_index_dir(cfg)
        dense = DenseRetriever.open(cfg, resolved, models_manifest_path=models_manifest_path)
        # Reuse the already-loaded index rather than parsing chunks.parquet a second time.
        return cls(dense, LexicalRetriever(dense.index, cfg), cfg)

    def __init__(self, dense: Retriever, lexical: Retriever, cfg: Config) -> None:
        self.dense = dense
        self.lexical = lexical
        self.cfg = cfg
        self.last_dense_top_score: float | None = None

    @property
    def abstain_top_score(self) -> float | None:
        """Cosine score the abstain gate should read — never the RRF score (BLOCKERS #12).

        The lexical arm cannot contribute here: BM25 scores are unbounded and share no scale
        with tau, so a lexical-only hit cannot rescue a query dense wanted to abstain on. That
        is the deliberate cost of keeping tau's calibration (BLOCKERS #8).
        """
        return self.last_dense_top_score

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        started = time.perf_counter()
        cfg = self.cfg.retrieve.hybrid
        dense_hits = self.dense.retrieve(query, k)
        lexical_hits = self.lexical.retrieve(query, k)
        self.last_dense_top_score = dense_hits[0].score if dense_hits else None

        result = fuse(
            [
                RankedList(name="dense", hits=list(dense_hits), weight=cfg.dense_weight),
                RankedList(name="lexical", hits=list(lexical_hits), weight=cfg.lexical_weight),
            ],
            k=k,
            rrf_k=cfg.rrf_k,
        )
        log.info(
            "retrieve.hybrid",
            k=k,
            n_dense=len(dense_hits),
            n_lexical=len(lexical_hits),
            n_fused=len(result.hits),
            method=result.method,
            dense_top_score=self.last_dense_top_score,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return result.hits
