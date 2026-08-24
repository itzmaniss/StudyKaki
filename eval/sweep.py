"""§10 arm sweep — every on/off combination, measured against the V1 baseline.

Two-phase by default. Retrieval is ~30ms/question and generation is ~90s, so all 8
permutations score on recall/MRR in seconds; groundedness costs ~80min per arm and is
opt-in (`--groundedness`) for the finalists only.

An arm whose module or model is missing is skipped with a reason, not faked.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import polars as pl

from core.config import Config, load_config
from eval.metrics import GoldQuestion
from eval.run import GOLDEN, EvalResult, evaluate, load_golden
from retrieve.retriever import Retriever

#: §10 order: rerank -> hybrid -> rewrite.
ARMS = ("rerank", "hybrid", "rewrite")

BASELINE = Path("eval/baselines/v1.json")


class ArmUnavailable(RuntimeError):
    """Arm is configured on but cannot be built. Skip the permutation, say why."""


@dataclass(frozen=True)
class SweepRow:
    arms: tuple[str, ...]
    result: EvalResult | None
    seconds: float
    skipped: str = ""

    @property
    def label(self) -> str:
        return "+".join(self.arms) if self.arms else "dense (V1)"


def config_for(base: Config, on: tuple[str, ...]) -> Config:
    """Base config with exactly `on` arms enabled."""
    raw = base.model_dump()
    for arm in ARMS:
        raw["retrieve"][arm]["enabled"] = arm in on
    return Config(**raw)


def build_retriever(
    cfg: Config,
    index_path: Path | None,
    *,
    models_manifest_path: Path | None = None,
) -> Retriever:
    """Compose the arms §10 asks for. Raises ArmUnavailable if one cannot be built."""
    r = cfg.retrieve

    if r.hybrid.enabled:
        try:
            from retrieve.fusion import HybridRetriever
        except ImportError as e:
            raise ArmUnavailable(f"hybrid: {e}") from e
        base: Retriever = HybridRetriever.open(
            cfg, index_path, models_manifest_path=models_manifest_path
        )
    else:
        from retrieve.dense import DenseRetriever

        base = DenseRetriever.open(cfg, index_path, models_manifest_path=models_manifest_path)

    if r.rerank.enabled:
        try:
            from retrieve.rerank import RerankingRetriever
        except ImportError as e:
            raise ArmUnavailable(f"rerank: {e}") from e
        base = RerankingRetriever.open(cfg, base, models_manifest_path=models_manifest_path)

    if r.rewrite.enabled:
        try:
            from retrieve.rewrite import RewritingRetriever
        except ImportError as e:
            raise ArmUnavailable(f"rewrite: {e}") from e
        base = RewritingRetriever.open(cfg, base)

    return base


def load_baseline(path: Path = BASELINE) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("metrics", {})


def run_permutation(
    on: tuple[str, ...],
    base_cfg: Config,
    golden: list[GoldQuestion],
    index_path: Path | None,
    *,
    generator: Any | None = None,
    models_manifest_path: Path | None = None,
) -> SweepRow:
    cfg = config_for(base_cfg, on)
    started = time.perf_counter()
    try:
        retriever = build_retriever(cfg, index_path, models_manifest_path=models_manifest_path)
    except (ArmUnavailable, FileNotFoundError, ValueError) as e:
        return SweepRow(arms=on, result=None, seconds=0.0, skipped=str(e))

    result, _ = evaluate(retriever, golden, cfg, label="+".join(on) or "dense", generator=generator)
    return SweepRow(arms=on, result=result, seconds=time.perf_counter() - started)


def as_table(rows: list[SweepRow], baseline: dict[str, Any], n_questions: int = 1) -> str:
    """One line per permutation, deltas against the V1 baseline."""
    head = f"{'arm':<24}{'recall@5':>10}{'Δ':>9}{'MRR@10':>9}{'Δ':>9}{'ground':>9}{'ms/q':>9}"
    out = [head, "-" * len(head)]
    b5 = baseline.get("recall@5")
    bmrr = baseline.get("mrr@10")
    n = max(n_questions, 1)

    for row in rows:
        if row.result is None:
            out.append(f"{row.label:<24}{'skipped: ' + row.skipped:>45}")
            continue
        r = row.result
        d5 = f"{r.recall_at_5 - b5:+.3f}" if b5 is not None else "n/a"
        dmrr = f"{r.mrr_at_10 - bmrr:+.3f}" if bmrr is not None else "n/a"
        g = "n/a" if r.groundedness is None else f"{r.groundedness:.3f}"
        out.append(
            f"{row.label:<24}{r.recall_at_5:>10.3f}{d5:>9}"
            f"{r.mrr_at_10:>9.3f}{dmrr:>9}{g:>9}{row.seconds / n * 1000:>9.0f}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§10 V2 arm sweep")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--index", type=Path, default=None, help="index dir; required if >1 exists")
    ap.add_argument("--out", type=Path, default=None, help="write per-permutation parquet here")
    ap.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated arms to sweep; others stay off",
    )
    ap.add_argument(
        "--groundedness",
        action="store_true",
        help="also generate answers (~80min per permutation) — finalists only",
    )
    args = ap.parse_args(argv)

    base_cfg = load_config(args.config)
    golden = load_golden(args.golden)
    baseline = load_baseline()
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; known: {list(ARMS)}")

    generator = None
    if args.groundedness:
        from answer.generate import load_generator

        generator = load_generator(base_cfg)

    rows: list[SweepRow] = []
    for flags in product((False, True), repeat=len(arms)):
        on = tuple(a for a, f in zip(arms, flags, strict=True) if f)
        rows.append(run_permutation(on, base_cfg, golden, args.index, generator=generator))
        print(f"  done: {rows[-1].label}", flush=True)

    print()
    print(as_table(rows, baseline, len(golden)))
    if not baseline:
        print("\n^ no eval/baselines/v1.json — deltas are n/a.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            [
                {
                    "arms": r.label,
                    "recall@5": None if r.result is None else r.result.recall_at_5,
                    "mrr@10": None if r.result is None else r.result.mrr_at_10,
                    "groundedness": None if r.result is None else r.result.groundedness,
                    "seconds": r.seconds,
                    "skipped": r.skipped,
                }
                for r in rows
            ]
        ).write_parquet(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
