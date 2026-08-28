"""Slide charts from an eval run — SVG, no plotting dependency (§0.1 pins the deps).

    uv run python -m scripts.plot_eval
    uv run python -m scripts.plot_eval --run data/eval/runs/<id>_dense.parquet

Writes to `data/eval/charts/`. Every chart is stamped with the run id and the host, because
INT8 embeddings drift between CPU and GPU (BLOCKERS #18) and a chart from the wrong machine
will not match the slide numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import tempfile
import xml.sax.saxutils as xml
from dataclasses import dataclass
from pathlib import Path

import polars as pl

INK = "#12161c"
MUTED = "#6b7480"
GRID = "#dfe3e8"
BLUE = "#0068b5"
DEEP = "#00285a"
TEAL = "#00a3a1"
AMBER = "#d97706"
RED = "#c0392b"
GREEN = "#1e8e5a"
FONT = "Helvetica"

#: Generator sweep, 2026-08-27, Core Ultra 7 255H, all 54 questions, embedder on GPU.0.
#: Recorded in configs/base.yaml; that run's parquet is not on disk.
GENERATORS = (
    ("gemma-4-e2b-it int4", 0.796, 12.6, True),
    ("qwen3-4b-instruct int4", 0.935, 29.5, False),
)


@dataclass(frozen=True)
class Canvas:
    w: int
    h: int
    parts: list[str]

    def add(self, s: str) -> None:
        self.parts.append(s)

    def svg(self) -> str:
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>{body}</svg>'
        )


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 16,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{xml.escape(s)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, *, rx: float = 0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, *, width: float = 1) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def frame(c: Canvas, title: str, subtitle: str, stamp: str) -> None:
    c.add(text(56, 62, title, size=30, weight="600"))
    c.add(text(56, 92, subtitle, size=17, fill=MUTED))
    c.add(text(c.w - 56, c.h - 26, stamp, size=13, fill=MUTED, anchor="end"))


def confusion(df: pl.DataFrame, stamp: str) -> str:
    c = Canvas(1100, 760, [])
    counts = {
        (bool(r["answerable"]), bool(r["abstained"])): int(r["len"])
        for r in df.group_by(["answerable", "abstained"]).len().iter_rows(named=True)
    }
    tp = counts.get((False, True), 0)
    fn = counts.get((False, False), 0)
    fp = counts.get((True, True), 0)
    tn = counts.get((True, False), 0)
    n = tp + fn + fp + tn

    frame(
        c,
        "Abstain decision",
        f"Does the tau gate decline the questions the corpus cannot answer?  n={n}",
        stamp,
    )

    x0, y0, cw, ch = 300, 190, 330, 190
    cells = [
        (0, 0, tn, "answered", GREEN, "correct"),
        (1, 0, fp, "declined", AMBER, "lost a good answer"),
        (0, 1, fn, "answered", RED, "answered anyway"),
        (1, 1, tp, "declined", GREEN, "correct"),
    ]
    for col, row, v, label, colour, note in cells:
        x, y = x0 + col * cw, y0 + row * ch
        shade = "#f4f6f8" if v == 0 else colour + "1f"
        c.add(rect(x, y, cw - 8, ch - 8, shade, rx=8))
        c.add(rect(x, y, 6, ch - 8, colour if v else GRID, rx=3))
        c.add(
            text(
                x + cw / 2 - 4,
                y + 92,
                str(v),
                size=64,
                fill=INK if v else MUTED,
                anchor="middle",
                weight="600",
            )
        )
        c.add(text(x + cw / 2 - 4, y + 124, label, size=17, fill=MUTED, anchor="middle"))
        c.add(
            text(
                x + cw / 2 - 4, y + 150, note, size=14, fill=colour if v else MUTED, anchor="middle"
            )
        )

    c.add(text(x0 + cw / 2 - 4, y0 - 22, "system answered", size=16, fill=MUTED, anchor="middle"))
    c.add(
        text(x0 + cw + cw / 2 - 4, y0 - 22, "system declined", size=16, fill=MUTED, anchor="middle")
    )
    c.add(text(x0 - 26, y0 + 96, "answerable", size=16, fill=MUTED, anchor="end"))
    c.add(text(x0 - 26, y0 + 118, f"({tn + fp})", size=14, fill=MUTED, anchor="end"))
    c.add(text(x0 - 26, y0 + ch + 96, "unanswerable", size=16, fill=MUTED, anchor="end"))
    c.add(text(x0 - 26, y0 + ch + 118, f"({tp + fn})", size=14, fill=MUTED, anchor="end"))

    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    y = y0 + 2 * ch + 60
    for i, (k, v) in enumerate(
        [
            ("abstain precision", f"{prec:.3f}" if prec is not None else "n/a"),
            ("abstain recall", f"{rec:.3f}" if rec is not None else "n/a"),
            ("good answers lost", f"{fp}"),
        ]
    ):
        x = 300 + i * 220
        c.add(text(x, y, v, size=30, weight="600"))
        c.add(text(x, y + 24, k, size=14, fill=MUTED))
    c.add(
        text(
            56,
            c.h - 26,
            f"Only {tp + fn} unanswerable questions — the bottom row is thin evidence.",
            size=13,
            fill=AMBER,
        )
    )
    return c.svg()


def score_separation(df: pl.DataFrame, tau: float, stamp: str) -> str:
    c = Canvas(1240, 560, [])
    frame(
        c,
        "Why tau is a trade, not a threshold",
        "Top-1 cosine score per question. The two populations overlap almost completely.",
        stamp,
    )
    lo, hi = 0.40, 0.82
    x0, x1 = 150, c.w - 90
    rows = [
        ("answerable", df.filter(pl.col("answerable"))["top_score"].to_list(), BLUE, 230),
        ("unanswerable", df.filter(~pl.col("answerable"))["top_score"].to_list(), RED, 350),
    ]

    def sx(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    for g in range(5):
        v = lo + g * (hi - lo) / 4
        c.add(line(sx(v), 175, sx(v), 420, GRID))
        c.add(text(sx(v), 448, f"{v:.2f}", size=14, fill=MUTED, anchor="middle"))

    for label, vals, colour, y in rows:
        c.add(text(x0 - 20, y + 5, label, size=17, anchor="end"))
        c.add(text(x0 - 20, y + 26, f"n={len(vals)}", size=13, fill=MUTED, anchor="end"))
        if vals:
            c.add(line(sx(min(vals)), y, sx(max(vals)), y, colour + "55", width=3))
        for i, v in enumerate(sorted(vals)):
            c.add(
                f'<circle cx="{sx(v):.1f}" cy="{y + (-9 if i % 2 else 9):.1f}" r="6" '
                f'fill="{colour}" fill-opacity="0.75"/>'
            )

    c.add(line(sx(tau), 162, sx(tau), 425, INK, width=2))
    c.add(text(sx(tau), 152, f"tau = {tau}", size=16, weight="600", anchor="middle"))
    c.add(
        text(
            56,
            c.h - 58,
            "Left of the line is declined. Moving tau right loses real answers before it"
            " catches many more bad ones.",
            size=15,
            fill=MUTED,
        )
    )
    c.add(
        text(
            56,
            c.h - 26,
            "Cross-lingual questions score lowest — hardest won, closest to the noise.",
            size=15,
            fill=MUTED,
        )
    )
    return c.svg()


def recall_curve(df: pl.DataFrame, floor: dict[str, float], stamp: str) -> str:
    c = Canvas(1160, 660, [])
    frame(
        c,
        "Retrieval vs a random baseline",
        "Fraction of questions whose gold page is in the top k.",
        stamp,
    )
    ks = [1, 5, 10]
    series = (
        ([float(df[f"recall@{k}"].mean()) for k in ks], BLUE, "dense (BGE-M3 INT8)"),
        ([float(floor.get(f"recall@{k}", 0.0)) for k in ks], MUTED, "random floor"),
    )
    x0, y0, w, h = 150, 190, 700, 350

    def px(i: int) -> float:
        return x0 + i * (w / (len(ks) - 1))

    def py(v: float) -> float:
        return y0 + h - v * h

    for g in range(6):
        v = g / 5
        c.add(line(x0, py(v), x0 + w, py(v), GRID))
        c.add(text(x0 - 18, py(v) + 5, f"{v:.1f}", size=14, fill=MUTED, anchor="end"))
    for i, k in enumerate(ks):
        c.add(text(px(i), y0 + h + 34, f"k={k}", size=17, fill=MUTED, anchor="middle"))

    for j, (vals, colour, name) in enumerate(series):
        ly = 190 + j * 34
        c.add(rect(x0 + w + 60, ly - 10, 26, 6, colour, rx=3))
        c.add(text(x0 + w + 96, ly, name, size=15, fill=colour))
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        c.add(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="4"/>')
        for i, v in enumerate(vals):
            c.add(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="7" fill="{colour}"/>')
            c.add(
                text(
                    px(i),
                    py(v) - 20,
                    f"{v:.3f}",
                    size=17,
                    fill=colour,
                    anchor="middle",
                    weight="600",
                )
            )
    return c.svg()


def per_language(df: pl.DataFrame, stamp: str) -> str:
    c = Canvas(1100, 560, [])
    frame(
        c,
        "Per language",
        "Broken out because a pooled mean hides a weak slice (BLOCKERS #17).",
        stamp,
    )
    agg = (
        df.group_by("lang")
        .agg(n=pl.len(), r5=pl.col("recall@5").mean(), mrr=pl.col("rr@10").mean())
        .sort("lang")
    )
    names = {"en": "English", "ta": "Tamil", "zh": "Chinese"}
    x0, y0 = 190, 175
    for i, r in enumerate(agg.iter_rows(named=True)):
        y = y0 + i * 122
        c.add(
            text(
                x0 - 24,
                y + 34,
                names.get(r["lang"], r["lang"]),
                size=19,
                anchor="end",
                weight="600",
            )
        )
        c.add(text(x0 - 24, y + 58, f"n={r['n']}", size=14, fill=MUTED, anchor="end"))
        for j, (label, v, colour) in enumerate(
            (("recall@5", r["r5"], BLUE), ("MRR@10", r["mrr"], TEAL))
        ):
            by = y + j * 44
            c.add(rect(x0, by, 620, 32, "#eef1f4", rx=4))
            c.add(rect(x0, by, 620 * float(v), 32, colour, rx=4))
            c.add(text(x0 + 634, by + 23, f"{float(v):.3f}", size=17, weight="600"))
            c.add(text(x0 + 724, by + 23, label, size=14, fill=MUTED))
    return c.svg()


def generator_tradeoff(stamp: str) -> str:
    c = Canvas(1360, 560, [])
    frame(
        c,
        "Generator: speed against groundedness",
        "54 questions, Core Ultra 7 255H. Retrieval identical, so the generator is the only"
        " variable.",
        stamp,
    )
    x0, y0, bar = 400, 170, 540
    slowest = max(g[2] for g in GENERATORS)
    for i, (name, gnd, secs, is_default) in enumerate(GENERATORS):
        y = y0 + i * 150
        c.add(
            text(
                x0 - 30, y + 26, name, size=19, anchor="end", weight="600" if is_default else "400"
            )
        )
        if is_default:
            c.add(text(x0 - 30, y + 52, "ships by default", size=14, fill=BLUE, anchor="end"))
        for j, (frac, shown, label, colour) in enumerate(
            (
                (gnd, f"{gnd:.3f}", "groundedness", DEEP if is_default else BLUE),
                (secs / slowest, f"{secs:.1f} s", "median generate", AMBER),
            )
        ):
            by = y + j * 52
            c.add(rect(x0, by, bar, 36, "#eef1f4", rx=4))
            c.add(rect(x0, by, bar * frac, 36, colour, rx=4))
            c.add(text(x0 + bar + 22, by + 25, shown, size=19, weight="600"))
            c.add(text(x0 + bar + 130, by + 25, label, size=15, fill=MUTED))
    c.add(
        text(
            56,
            c.h - 26,
            "The default is 2.3x faster and 0.14 less grounded. Both rows are in"
            " configs/base.yaml.",
            size=15,
            fill=MUTED,
        )
    )
    return c.svg()


def to_png(svgs: list[Path], side: int = 2200) -> None:
    """Rasterise via QuickLook, which fits to a square and crops — so pad, render, crop back.

    macOS only. SVG is the portable output: PowerPoint and Keynote both import it directly.
    """
    if not (shutil.which("qlmanage") and shutil.which("sips")):
        print("qlmanage/sips not found — SVGs written; import those directly into slides")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        boxes: dict[str, tuple[int, int]] = {}
        for svg in svgs:
            body = svg.read_text(encoding="utf-8")
            w = int(re.search(r'width="(\d+)"', body).group(1))
            h = int(re.search(r'height="(\d+)"', body).group(1))
            box = max(w, h)
            inner = re.sub(r"^<svg[^>]*?>", "", body, count=1).rsplit("</svg>", 1)[0]
            (work / svg.name).write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{box}" height="{box}" '
                f'viewBox="0 0 {box} {box}"><rect width="{box}" height="{box}" fill="#ffffff"/>'
                f'<g transform="translate({(box - w) / 2},{(box - h) / 2})">{inner}</g></svg>',
                encoding="utf-8",
            )
            boxes[svg.stem] = (round(w * side / box), round(h * side / box))
        subprocess.run(
            [
                "qlmanage",
                "-t",
                "-s",
                str(side),
                "-o",
                str(work),
                *[str(work / s.name) for s in svgs],
            ],
            capture_output=True,
            check=False,
        )
        for svg in svgs:
            rendered = work / f"{svg.name}.png"
            if not rendered.exists():
                print(f"could not rasterise {svg.name}")
                continue
            dst = svg.with_suffix(".png")
            shutil.copyfile(rendered, dst)
            cw, ch = boxes[svg.stem]
            subprocess.run(
                ["sips", "-c", str(ch), str(cw), str(dst)], capture_output=True, check=True
            )
            print(f"wrote {dst}  {cw}x{ch}")


def latest_run(root: Path) -> Path:
    runs = sorted(root.glob("*_dense.parquet"))
    if not runs:
        raise SystemExit(
            f"no dense run in {root} — run `python -m eval.run --retriever dense` first"
        )
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Slide charts from an eval run")
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=Path("eval/baselines/v1.json"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/charts"))
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--png", action="store_true", help="also rasterise (macOS: qlmanage + sips)")
    args = ap.parse_args()

    run = args.run or latest_run(Path("data/eval/runs"))
    df = pl.read_parquet(run)
    floor = json.loads(args.baseline.read_text()).get("random_floor", {})
    host = f"{platform.system()} {platform.machine()}"
    stamp = f"{run.stem}  ·  {host}  ·  n={df.height}"

    args.out.mkdir(parents=True, exist_ok=True)
    charts = {
        "01_confusion_abstain": confusion(df, stamp),
        "02_score_separation": score_separation(df, args.tau, stamp),
        "03_recall_curve": recall_curve(df, floor, stamp),
        "04_per_language": per_language(df, stamp),
        "05_generator_tradeoff": generator_tradeoff(
            "2026-08-27 sweep  ·  Core Ultra 7 255H  ·  n=54"
        ),
    }
    written = []
    for name, svg in charts.items():
        path = args.out / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        print(f"wrote {path}")
        written.append(path)
    if args.png:
        to_png(written)


if __name__ == "__main__":
    main()
