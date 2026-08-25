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
from typing import Any

import polars as pl
import structlog

from core.config import Config, load_config
from core.schema import Chunk, Retrieved
from eval.bench import peak_rss_bytes
from eval.metrics import GoldQuestion, groundedness, mean, recall_at, reciprocal_rank
from ingest.normalize import detect_script
from retrieve.retriever import Retriever, abstains_for

log = structlog.get_logger("eval.run")

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
class GenerationOutcome:
    """What one generated answer tells us, beyond whether its citations were real.

    `groundedness` is `None` for an abstention, which is why the other two fields exist:
    a `None` on its own cannot distinguish "no generator was run" from "the model declined",
    and BLOCKERS #17 is precisely a case where those two were conflated and six Tamil
    failures became structurally invisible in the headline average.
    """

    groundedness: float | None
    model_abstained: bool
    #: `detect_script(answer) == detect_script(question)`. `None` when there is no answer to
    #: score (abstention) or nothing with letters in it to detect.
    answer_lang_match: bool | None


@dataclass(frozen=True)
class LangBreakdown:
    """One language's slice of the run (BLOCKERS #17).

    The pooled means hide a language that fails completely — 41 healthy en/zh rows drowned
    6 Tamil zeros and every headline number still looked fine. Nothing about that is
    specific to Tamil; it is what pooling does to any minority slice.
    """

    lang: str
    n: int
    recall_at_5: float
    mrr_at_10: float
    retrieval_abstained: int
    model_abstained: int
    n_scored: int
    groundedness: float | None
    answer_lang_match: float | None

    def as_row(self) -> list[str]:
        return [
            self.lang,
            str(self.n),
            f"{self.recall_at_5:.3f}",
            f"{self.mrr_at_10:.3f}",
            str(self.retrieval_abstained),
            str(self.model_abstained),
            str(self.n_scored),
            "n/a" if self.groundedness is None else f"{self.groundedness:.3f}",
            "n/a" if self.answer_lang_match is None else f"{self.answer_lang_match:.3f}",
        ]


LANG_COLS = (
    "lang",
    "n",
    "recall@5",
    "MRR@10",
    "abstain_ret",
    "abstain_model",
    "scored",
    "groundedness",
    "lang_match",
)


@dataclass(frozen=True)
class EvalResult:
    n_questions: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    abstain_precision: float | None
    groundedness: float | None
    retriever: str
    per_lang: tuple[LangBreakdown, ...] = ()

    def as_lang_table(self) -> str:
        """The pooled table above, split by language (BLOCKERS #17)."""
        rows = [b.as_row() for b in self.per_lang]
        widths = [
            max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
            for i, c in enumerate(LANG_COLS)
        ]
        head = "  ".join(c.ljust(w) for c, w in zip(LANG_COLS, widths, strict=True))
        body = "\n".join(
            "  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in rows
        )
        return f"{head}\n{'-' * len(head)}\n{body}"

    def as_table(self) -> str:
        cols = ["recall@1", "recall@5", "recall@10", "MRR@10", "abstain_precision", "groundedness"]
        vals = [
            f"{self.recall_at_1:.3f}",
            f"{self.recall_at_5:.3f}",
            f"{self.recall_at_10:.3f}",
            f"{self.mrr_at_10:.3f}",
            "n/a" if self.abstain_precision is None else f"{self.abstain_precision:.3f}",
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
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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
    generator: Any | None = None,
) -> tuple[EvalResult, pl.DataFrame]:
    """Retrieval metrics always; groundedness only when a `generator` is supplied.

    Generation is orders of magnitude slower than retrieval — a 4B model on CPU over sixty
    questions is minutes, not seconds — so the groundedness column stays `n/a` unless the
    caller has explicitly paid for it (`--groundedness`). Retrieval numbers are identical
    either way: the generator never touches `hits`.
    """
    k = cfg.retrieve.k
    tau = cfg.retrieve.tau
    rows = []

    for n, gold in enumerate(golden, start=1):
        # §5 tracks peak RSS, and a generative sweep is where it actually bites: a run that
        # dies without a traceback leaves this as the only evidence of why.
        if n % 5 == 0 or n == 1:
            log.info(
                "eval.progress",
                question=n,
                of=len(golden),
                lang=gold.lang,
                peak_rss_gb=round(peak_rss_bytes() / 1e9, 2),
            )
        hits = retriever.retrieve(gold.q, k)
        abstained = abstains_for(hits, tau, retriever)
        outcome = _generation_outcome(gold.q, hits, cfg, generator, retrieval_abstained=abstained)
        answerable = not gold.unanswerable
        rows.append(
            {
                "q": gold.q,
                "lang": gold.lang,
                "doc_id": gold.doc_id,
                "gold_pages": gold.gold_pages,
                "answerable": answerable,
                # `abstained` is the *retrieval* gate only — it is what tau controls, and it
                # is what abstain_precision is scored against. A model that refuses on
                # context retrieval accepted is `model_abstained`, a different failure with
                # a different fix (BLOCKERS #17).
                "abstained": abstained,
                "model_abstained": outcome.model_abstained if generator is not None else None,
                "answer_lang_match": outcome.answer_lang_match,
                "top_score": hits[0].score if hits else 0.0,
                # Retrieval quality is scored on what was retrieved, independent of the
                # abstain decision — otherwise tau tuning silently moves recall.
                "recall@1": recall_at(hits, gold, 1) if answerable else None,
                "recall@5": recall_at(hits, gold, 5) if answerable else None,
                "recall@10": recall_at(hits, gold, 10) if answerable else None,
                "rr@10": reciprocal_rank(hits, gold, 10) if answerable else None,
                "groundedness": outcome.groundedness,
            }
        )

    df = pl.DataFrame(rows)
    ans = [r for r in rows if r["answerable"]]
    abstained_rows = [r for r in rows if r["abstained"]]

    # Of everything we declined to answer, how much *should* have been declined. None, not
    # 1.0, when nothing was declined: a system that answers everything has not achieved
    # perfect abstain precision, it has abstained zero times, and printing 1.000 for that
    # reads as the strongest possible score for the weakest possible behaviour (§0.6).
    abstain_precision = (
        mean([1.0 if not r["answerable"] else 0.0 for r in abstained_rows])
        if abstained_rows
        else None
    )

    result = EvalResult(
        n_questions=len(golden),
        recall_at_1=mean([r["recall@1"] for r in ans]),
        recall_at_5=mean([r["recall@5"] for r in ans]),
        recall_at_10=mean([r["recall@10"] for r in ans]),
        mrr_at_10=mean([r["rr@10"] for r in ans]),
        abstain_precision=abstain_precision,
        groundedness=(
            groundedness(scored)
            if (scored := [r["groundedness"] for r in rows if r["groundedness"] is not None])
            else None
        ),
        retriever=label,
        per_lang=_per_lang(rows),
    )
    return result, df


def _per_lang(rows: list[dict[str, Any]]) -> tuple[LangBreakdown, ...]:
    out: list[LangBreakdown] = []
    for lang in sorted({r["lang"] for r in rows}):
        group = [r for r in rows if r["lang"] == lang]
        ans = [r for r in group if r["answerable"]]
        scored = [r["groundedness"] for r in group if r["groundedness"] is not None]
        matches = [r["answer_lang_match"] for r in group if r["answer_lang_match"] is not None]
        out.append(
            LangBreakdown(
                lang=lang,
                n=len(group),
                recall_at_5=mean([r["recall@5"] for r in ans]),
                mrr_at_10=mean([r["rr@10"] for r in ans]),
                retrieval_abstained=sum(1 for r in group if r["abstained"]),
                model_abstained=sum(1 for r in group if r["model_abstained"]),
                n_scored=len(scored),
                groundedness=groundedness(scored) if scored else None,
                answer_lang_match=mean([1.0 if m else 0.0 for m in matches]) if matches else None,
            )
        )
    return tuple(out)


def _generation_outcome(
    question: str,
    hits: list[Retrieved],
    cfg: Config,
    generator: Any | None,
    *,
    retrieval_abstained: bool,
) -> GenerationOutcome:
    """One generated answer, scored on its citations and on the language it came back in.

    `groundedness` is None for an abstention rather than 0.0 — declining to answer makes no
    claim, so it belongs out of the average entirely (see `eval.metrics.groundedness`). What
    BLOCKERS #17 showed is that dropping it there and recording nothing else makes the
    refusal itself unobservable, so the abstention is now *counted* on its way out.
    """
    if generator is None:
        return GenerationOutcome(groundedness=None, model_abstained=False, answer_lang_match=None)
    from answer.generate import generate_answer

    result = generate_answer(question, generator=generator, cfg=cfg, hits=hits)
    if result.abstained:
        # Retrieval-gated abstentions never reached the model, so they are not the model's
        # refusal. `retrieval_abstained` is the caller's own gate decision, reused rather
        # than recomputed: `abstain_score` reads the retriever's arm, and recomputing it
        # without that retriever would silently score a reranked run against dense scores.
        return GenerationOutcome(
            groundedness=None,
            model_abstained=not retrieval_abstained,
            answer_lang_match=None,
        )
    return GenerationOutcome(
        groundedness=result.groundedness,
        model_abstained=False,
        answer_lang_match=_lang_matches(question, result.answer.text),
    )


def _lang_matches(question: str, answer: str) -> bool | None:
    """§4's "answer in the language of the question", as a number.

    `ingest/normalize.py`'s detector is coarse — script, not language (§11 puts real language
    ID out of scope) — which is exactly enough here: Tamil answered in English is a script
    change, and that is the failure mode #17 found.
    """
    q_script, a_script = detect_script(question), detect_script(answer)
    if "unknown" in (q_script, a_script):
        return None
    return q_script == a_script


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


def build_retriever(
    name: str, cfg: Config, golden: list[GoldQuestion], seed: int, index: Path | None
) -> Retriever:
    """`random` needs no index and no model; `dense` needs both.

    Imported lazily so the random baseline keeps working on a machine with no IR on disk —
    that baseline is what proves the harness itself is sound (§5), and it must never depend
    on the thing it exists to measure.
    """
    if name == "random":
        return RandomRetriever(_pool_from_golden(golden, seed), seed)

    from retrieve.dense import DenseRetriever

    return DenseRetriever.open(cfg, index)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieval eval harness (ARCHITECTURE.md §5)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--retriever", choices=["random", "dense"], default="random")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--index", type=Path, default=None, help="index dir; defaults to the config's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--groundedness",
        action="store_true",
        help="also generate an answer per question and score its citations; needs the "
        "converted generator and takes minutes, so the column is n/a without it",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    golden = load_golden(args.golden)

    retriever: Retriever = build_retriever(args.retriever, cfg, golden, args.seed, args.index)

    generator = None
    if args.groundedness:
        from answer.generate import load_generator

        generator = load_generator(cfg)

    result, df = evaluate(retriever, golden, cfg, label=args.retriever, generator=generator)
    print(result.as_table())
    # Always, not only under --groundedness: the pooled row is what hid six dead Tamil
    # questions behind a healthy-looking 0.844 (BLOCKERS #17), and that is not a property of
    # the generation column — recall pools the same way.
    print(f"\nby language:\n{result.as_lang_table()}")
    if args.retriever == "random":
        print("\n^ random baseline — these numbers are meaningless by design (§5).")
    if any(q.note.startswith("PLACEHOLDER") for q in golden):
        print(
            "\n^ golden.jsonl still contains PLACEHOLDER questions — these numbers describe "
            "made-up labels, not retrieval quality (BLOCKERS.md #3)."
        )

    try:
        path = _write_run(df, cfg, args.retriever, args.out_dir)
        print(f"wrote {path}")
    except OSError as e:
        print(f"could not write run parquet ({e}); metrics above are still valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
