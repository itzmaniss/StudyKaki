"""Download the evaluation corpus — ARCHITECTURE.md §0.3.

Network access lives in scripts, never in runtime code. This is the second such script
(`scripts/setup.py` fetches models); nothing under `ingest/`, `retrieve/` or `answer/`
may reach the network.

    uv run python -m scripts.fetch_corpus            # fetch everything missing
    uv run python -m scripts.fetch_corpus --report   # just say what is on disk
    uv run python -m scripts.fetch_corpus --lang ta

Idempotent: a file that already exists and starts with `%PDF` is skipped, so a partial
run can simply be repeated. Corpus lives under `data/`, which is gitignored — none of
this enters the repo.

Every entry reports whether it carries a native text layer, because that decides whether
`ingest/ocr.py` runs at all (§3: a text layer means OCR is skipped entirely). The corpus
needs genuine image-only scans or the OCR stage is never exercised on real data.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "data" / "corpus"

USER_AGENT = "intel-2026-corpus-fetcher/0.1 (offline study RAG; local eval corpus)"
PDF_MAGIC = b"%PDF"


@dataclass(frozen=True)
class Source:
    name: str
    lang: str
    url: str
    note: str

    @property
    def dest(self) -> Path:
        return CORPUS_ROOT / self.lang / f"{self.name}.pdf"


def _archive(identifier: str, filename: str) -> str:
    # Archive filenames routinely contain spaces and commas; urllib rejects them raw.
    return f"https://archive.org/download/{identifier}/{urllib.parse.quote(filename)}"


# Tamil/English Std12 Computer Science are the *same textbook in two languages*, which is
# what makes the §5 cross-lingual questions possible: a Tamil question whose answer lives
# in the English volume, and vice versa. BGE-M3 is built for exactly this.
SOURCES: tuple[Source, ...] = (
    Source(
        "std12_cs_vol1_ta",
        "ta",
        _archive("std12cs1tm2006tnbooks", "std12cs1tm2006tnbooks.pdf"),
        "Std12 Computer Science Vol 1, Tamil medium (~232pp)",
    ),
    Source(
        "std12_cs_vol2_ta",
        "ta",
        _archive("std12cs2tm2006tnbooks", "std12cs2tm2006tnbooks.pdf"),
        "Std12 Computer Science Vol 2, Tamil medium (~240pp)",
    ),
    Source(
        "digital_electronics_ta",
        "ta",
        _archive("book-23-digital-electronic-tamil", "Book 23, Digital Electronic_Tamil.pdf"),
        "Digital Electronics (Tamil) — dense technical scan, circuits and diagrams",
    ),
    Source(
        "std12_cs_vol1_en",
        "en",
        _archive("std12cs1em2006tnbooks", "std12cs1em2006tnbooks.pdf"),
        "Std12 Computer Science Vol 1, English medium — parallel to std12_cs_vol1_ta",
    ),
    Source(
        "std12_cs_vol2_en",
        "en",
        _archive("std12cs2em2006tnbooks", "std12cs2em2006tnbooks.pdf"),
        "Std12 Computer Science Vol 2, English medium — parallel to std12_cs_vol2_ta",
    ),
    Source(
        "cnnic_internet_report",
        "zh",
        "https://www.cnnic.net.cn/NMediaFile/2023/0807/MAIN1691372884990HDTP1QOST8.pdf",
        "CNNIC 中国互联网络发展状况统计报告 — dense IPv4/IPv6 and broadband tables",
    ),
    # ITU publishes the same document per UN language, suffixed -C (Chinese) and -E (English),
    # which makes them a second parallel pair for cross-lingual questions.
    Source(
        "itu_wtdc22_zh",
        "zh",
        "https://www.itu.int/dms_pub/itu-d/opb/tdc/D-TDC-WTDC-2022-PDF-C.pdf",
        "ITU WTDC-22 Final Report, Chinese — parallel to itu_wtdc22_en",
    ),
    Source(
        "itu_wtdc22_en",
        "en",
        "https://www.itu.int/dms_pub/itu-d/opb/tdc/D-TDC-WTDC-2022-PDF-E.pdf",
        "ITU WTDC-22 Final Report, English — parallel to itu_wtdc22_zh",
    ),
)


def is_pdf(path: Path) -> bool:
    """A served error page or a viewer shell is still a 200; check the magic bytes."""
    if not path.exists() or path.stat().st_size < len(PDF_MAGIC):
        return False
    with path.open("rb") as fh:
        return fh.read(len(PDF_MAGIC)) == PDF_MAGIC


def fetch(source: Source, *, force: bool = False) -> tuple[bool, str]:
    dest = source.dest
    if not force and is_pdf(dest):
        return True, f"skip (have {dest.stat().st_size / 1e6:.1f} MB)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pdf.part")
    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            tmp.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        return False, f"FAILED: {exc}"

    if not is_pdf(tmp):
        head = tmp.read_bytes()[:60]
        tmp.unlink(missing_ok=True)
        return False, f"FAILED: not a PDF (starts {head!r})"

    tmp.replace(dest)
    return True, f"ok ({dest.stat().st_size / 1e6:.1f} MB)"


def describe(path: Path) -> str:
    """Pages, and whether a native text layer exists — the OCR/no-OCR decision (§3)."""
    try:
        import pymupdf
    except ImportError:
        return "pymupdf unavailable"
    try:
        with pymupdf.open(path) as doc:
            pages = doc.page_count
            sampled = min(10, pages)
            chars = sum(len(doc[i].get_text().strip()) for i in range(sampled))
    except (RuntimeError, ValueError) as exc:
        return f"unreadable: {exc}"
    per_page = chars / sampled if sampled else 0
    layer = "TEXT LAYER (OCR skipped)" if per_page > 100 else "image-only (OCR REQUIRED)"
    return f"{pages}pp, {layer}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch the eval corpus (ARCHITECTURE.md §0.3)")
    ap.add_argument("--lang", action="append", help="restrict to these languages")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--report", action="store_true", help="describe what is on disk, fetch nothing")
    args = ap.parse_args(argv)

    selected = [s for s in SOURCES if not args.lang or s.lang in args.lang]
    failures = 0

    for source in selected:
        if args.report:
            state = describe(source.dest) if is_pdf(source.dest) else "MISSING"
            print(f"[{source.lang}] {source.name:24s} {state}")
            continue

        ok, message = fetch(source, force=args.force)
        detail = f"  {describe(source.dest)}" if ok else ""
        print(f"[{source.lang}] {source.name:24s} {message}{detail}")
        print(f"       {source.note}")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} source(s) failed. Re-run to retry; completed files are skipped.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
