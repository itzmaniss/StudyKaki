"""Rank fusion — ARCHITECTURE.md §1, §10.

V1 wires dense only, but `fuse()` already takes a list of ranked lists, so the seam is real
rather than imagined. V2 plugged in as designed: `retrieve/lexical.py` passes a BM25
`RankedList` alongside, and none of the arithmetic changed. `HybridRetriever` is the
composition, off unless `retrieve.hybrid.enabled`.

The score-units trap: RRF scores come from ranks, not similarities. The best possible RRF
score for one list is `1/(60+1)` ≈ 0.016, far below tau (0.45), so feeding RRF to
`retriever.abstains` would abstain on every query. `FusionResult.scores_are_similarities`
says which units you hold.

So the hybrid arm is measurable on `recall@k` and `mrr@10` (pure rank metrics) but not yet
safe behind `answer/generate.py`, whose gate is calibrated on cosine — tau's safe window is
0.01 wide (BLOCKERS #8). Gap tracked as BLOCKERS #12; `last_dense_top_score` exposes what
an abstain decision would need, unconsumed for now.
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
    """Fuse ranked lists by `sum(weight / (rrf_k + rank))`, deduped on `chunk_id`.

    Ranks come from `Retrieved.rank`, not list position, so a sliced list keeps its true
    ranks. Returned scores are RRF, not similarities — see the module docstring.
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

    The passthrough keeps V1 honest: with only dense wired, scores stay cosine and tau still
    means what §4 says. A second arm flips `scores_are_similarities` and forces the caller
    to notice.
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

    Both arms are asked for `k` and fused down to `k`, not asked for `k/2` each: a chunk both
    arms rank highly should be able to beat one only dense found, and truncating first
    prevents that comparison.

    Returned scores are RRF, not similarities. `last_dense_top_score` carries the cosine an
    abstain decision would need.
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

        BM25 is unbounded and shares no scale with tau, so a lexical-only hit cannot rescue
        a query dense wanted to abstain on. The deliberate cost of tau's calibration (#8).
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
