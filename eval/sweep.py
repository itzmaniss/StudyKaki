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
from retrieve.retriever import Retriever, degraded_calls

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
    #: Queries where an arm silently fell back. Non-zero means the metrics below describe
    #: whatever ran instead of this arm, so they are not reported as a result.
    degraded: int = 0

    @property
    def trustworthy(self) -> bool:
        return self.result is not None and self.degraded == 0

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
    return SweepRow(
        arms=on,
        result=result,
        seconds=time.perf_counter() - started,
        degraded=degraded_calls(retriever),
    )


def as_table(rows: list[SweepRow], baseline: dict[str, Any], n_questions: int = 1) -> str:
    """One line per permutation, deltas against the V1 baseline."""
    head = (
        f"{'arm':<22}{'recall@5':>10}{'Δ':>8}{'MRR@10':>9}{'Δ':>8}{'ground':>8}{'Δ':>8}{'ms/q':>9}"
    )
    out = [head, "-" * len(head)]
    b5 = baseline.get("recall@5")
    bmrr = baseline.get("mrr@10")
    bg = baseline.get("groundedness")
    n = max(n_questions, 1)

    for row in rows:
        if row.result is None:
            out.append(f"{row.label:<22}{'skipped: ' + row.skipped:>45}")
            continue
        if row.degraded:
            # Never print a number an arm did not produce. A silent fallback reads exactly
            # like a real measurement otherwise — see retriever.degraded_calls.
            out.append(
                f"{row.label:<24}{f'DEGRADED on {row.degraded}/{n} queries - arm did not run':>55}"
            )
            continue
        r = row.result
        d5 = f"{r.recall_at_5 - b5:+.3f}" if b5 is not None else "n/a"
        dmrr = f"{r.mrr_at_10 - bmrr:+.3f}" if bmrr is not None else "n/a"
        g = "n/a" if r.groundedness is None else f"{r.groundedness:.3f}"
        dg = (
            f"{r.groundedness - bg:+.3f}"
            if (r.groundedness is not None and bg is not None)
            else "n/a"
        )
        out.append(
            f"{row.label:<22}{r.recall_at_5:>10.3f}{d5:>8}"
            f"{r.mrr_at_10:>9.3f}{dmrr:>8}{g:>8}{dg:>8}{row.seconds / n * 1000:>9.0f}"
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
        "--only",
        action="append",
        default=None,
        metavar="ARMS",
        help="run just this combination, e.g. --only rerank+hybrid (repeatable; "
        '"dense" means all arms off). Skips the other permutations, which matters when '
        "--groundedness makes each one cost ~80 min.",
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

    wanted = None
    if args.only:
        wanted = set()
        for spec in args.only:
            picked = tuple(a for a in ARMS if a in spec.split("+"))
            if spec.strip() not in ("dense", "") and not picked:
                raise SystemExit(f"--only {spec!r} names no known arm; known: {list(ARMS)}")
            wanted.add(picked)

    rows: list[SweepRow] = []
    for flags in product((False, True), repeat=len(arms)):
        on = tuple(a for a, f in zip(arms, flags, strict=True) if f)
        if wanted is not None and on not in wanted:
            continue
        rows.append(run_permutation(on, base_cfg, golden, args.index, generator=generator))
        print(f"  done: {rows[-1].label}", flush=True)

    print()
    print(as_table(rows, baseline, len(golden)))
    if not baseline:
        print("\n^ no eval/baselines/v1.json — deltas are n/a.")
    bad = [r.label for r in rows if r.degraded]
    if bad:
        print(f"\n^ {', '.join(bad)} degraded — those arms did not run; metrics withheld.")

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
                    "degraded": r.degraded,
                }
                for r in rows
            ]
        ).write_parquet(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
