"""Eval harness — ARCHITECTURE.md §5.

Built against a RANDOM retriever first, deliberately. If the table below is meaningless
but the harness runs, the harness is trustworthy; every real component is then measurable
against it. Nothing ships without a number from here (§0.5).

    uv run python -m eval.run --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from core.config import Config, load_config
from core.schema import Chunk, Retrieved
from eval.metrics import GoldQuestion, mean, recall_at, reciprocal_rank
from retrieve.retriever import Retriever, abstains

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"


class RandomRetriever:
    """Baseline floor. Samples from a fixed chunk pool; scores are noise.

    Any real retriever that cannot beat this is broken, not merely weak.
    """

    def __init__(self, pool: list[Chunk], seed: int = 0) -> None:
        if not pool:
            raise ValueError("RandomRetriever needs a non-empty chunk pool")
        self.pool = pool
        self.rng = random.Random(seed)

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        n = min(k, len(self.pool))
        picked = self.rng.sample(self.pool, n)
        scores = sorted((self.rng.random() for _ in range(n)), reverse=True)
        return [
            Retrieved(chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(zip(picked, scores, strict=True), start=1)
        ]


@dataclass(frozen=True)
class EvalResult:
    n_questions: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    abstain_precision: float
    groundedness: float | None
    retriever: str

    def as_table(self) -> str:
        cols = ["recall@1", "recall@5", "recall@10", "MRR@10", "abstain_precision", "groundedness"]
        vals = [
            f"{self.recall_at_1:.3f}",
            f"{self.recall_at_5:.3f}",
            f"{self.recall_at_10:.3f}",
            f"{self.mrr_at_10:.3f}",
            f"{self.abstain_precision:.3f}",
            "n/a" if self.groundedness is None else f"{self.groundedness:.3f}",
        ]
        widths = [max(len(c), len(v)) for c, v in zip(cols, vals, strict=True)]
        head = "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))
        body = "  ".join(v.ljust(w) for v, w in zip(vals, widths, strict=True))
        return (
            f"retriever={self.retriever}  n={self.n_questions}\n{head}\n{'-' * len(head)}\n{body}"
        )


def load_golden(path: Path = GOLDEN) -> list[GoldQuestion]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — §5 requires 40-60 golden questions before numbers mean anything"
        )
    out: list[GoldQuestion] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(GoldQuestion.from_row(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"{path}:{lineno}: {e}") from e
    if not out:
        raise ValueError(f"{path} contains no questions")
    return out


def evaluate(
    retriever: Retriever,
    golden: list[GoldQuestion],
    cfg: Config,
    label: str = "unknown",
) -> tuple[EvalResult, pl.DataFrame]:
    k = cfg.retrieve.k
    tau = cfg.retrieve.tau
    rows = []

    for gold in golden:
        hits = retriever.retrieve(gold.q, k)
        abstained = abstains(hits, tau)
        answerable = not gold.unanswerable
        rows.append(
            {
                "q": gold.q,
                "lang": gold.lang,
                "doc_id": gold.doc_id,
                "gold_pages": gold.gold_pages,
                "answerable": answerable,
                "abstained": abstained,
                "top_score": hits[0].score if hits else 0.0,
                # Retrieval quality is scored on what was retrieved, independent of the
                # abstain decision — otherwise tau tuning silently moves recall.
                "recall@1": recall_at(hits, gold, 1) if answerable else None,
                "recall@5": recall_at(hits, gold, 5) if answerable else None,
                "recall@10": recall_at(hits, gold, 10) if answerable else None,
                "rr@10": reciprocal_rank(hits, gold, 10) if answerable else None,
            }
        )

    df = pl.DataFrame(rows)
    ans = [r for r in rows if r["answerable"]]
    abstained_rows = [r for r in rows if r["abstained"]]

    # Of everything we declined to answer, how much *should* have been declined.
    abstain_precision = (
        mean([1.0 if not r["answerable"] else 0.0 for r in abstained_rows])
        if abstained_rows
        else 1.0
    )

    result = EvalResult(
        n_questions=len(golden),
        recall_at_1=mean([r["recall@1"] for r in ans]),
        recall_at_5=mean([r["recall@5"] for r in ans]),
        recall_at_10=mean([r["recall@10"] for r in ans]),
        mrr_at_10=mean([r["rr@10"] for r in ans]),
        abstain_precision=abstain_precision,
        groundedness=None,  # requires answer/ — wired in Block 6-8h
        retriever=label,
    )
    return result, df


def _pool_from_golden(golden: list[GoldQuestion], seed: int = 0) -> list[Chunk]:
    """Synthetic corpus so the harness is exercisable before any real index exists.

    Includes every gold page plus decoys, so a random retriever scores above zero but
    far below anything real.
    """
    rng = random.Random(seed)
    pool: list[Chunk] = []
    for gi, gold in enumerate(golden):
        pages = gold.gold_pages or [1]
        decoys = [p + rng.randint(1, 40) for p in pages]
        for pi, page in enumerate([*pages, *decoys]):
            pool.append(
                Chunk(
                    chunk_id=f"stub-{gi}-{pi}",
                    doc_id=gold.doc_id,
                    page_start=page,
                    page_end=page,
                    block_ids=[f"stub-block-{gi}-{pi}"],
                    bbox_union=(0.0, 0.0, 1.0, 1.0),
                    heading_path=[],
                    text=f"stub chunk for {gold.doc_id} p.{page}",
                    token_count=0,
                    lang=gold.lang,
                    script="latn",
                )
            )
    return pool


def _write_run(df: pl.DataFrame, cfg: Config, label: str, out_dir: Path | None) -> Path:
    target = out_dir or cfg.resolve(cfg.paths.data_dir) / "eval" / "runs"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"{stamp}_{label}.parquet"
    df.write_parquet(path, compression="zstd")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieval eval harness (ARCHITECTURE.md §5)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--retriever", choices=["random"], default="random")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    golden = load_golden(args.golden)

    retriever: Retriever = RandomRetriever(_pool_from_golden(golden, args.seed), args.seed)

    result, df = evaluate(retriever, golden, cfg, label=args.retriever)
    print(result.as_table())
    if args.retriever == "random":
        print("\n^ random baseline — these numbers are meaningless by design (§5).")

    try:
        path = _write_run(df, cfg, args.retriever, args.out_dir)
        print(f"wrote {path}")
    except OSError as e:
        print(f"could not write run parquet ({e}); metrics above are still valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
