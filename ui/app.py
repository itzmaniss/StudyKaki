"""Terminal UI — ARCHITECTURE.md §1 (thin, no logic here), §9 (tier visible on every answer).

Rendering only. Retrieval, the `tau` threshold, tier selection, citation verification and
telemetry all happen in `retrieve/` and `answer/`. This module turns one `AnswerResult` into
characters and decides nothing about an answer.

A terminal rather than a browser: §0.1 pins the dependency set and holds no GUI framework,
and §0.3 forbids the network at runtime. Citations are clickable the way a terminal makes
things clickable — an OSC 8 hyperlink to `file://…#page=N`, which opens the student's own
PDF at the cited page, offline, in every terminal that supports the escape (iTerm2, VS Code,
GNOME Terminal) and degrades to plain text everywhere else.

§9's Tier 3 contract is *enforced* in `answer/prompt.py`; what this file owes it is
*visibility*: a warning-framed block, the literal disclaimer carried through untouched, no
citation list, and Tier 1's abstention said out loud before the ungrounded answer.

Streaming deltas are provisional — `answer/cite.py` runs after the last token. The draft is
streamed for responsiveness and the verified answer is reprinted only when verification
actually changed it, so the common case shows one answer, not two.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import structlog

from answer.cite import find_markers
from answer.generate import AnswerResult, GenerationError, stream_answer

# `_provenance` is the renderer `answer/prompt.py` used for the block header the model saw.
# Reusing it means the `doc / p.N` printed under an answer is the string the model was
# citing, rather than a second format that can drift away from it.
from answer.prompt import _provenance as provenance
from core.config import DEFAULT_CONFIG, Config, load_config
from core.schema import Chunk
from core.telemetry import TIER_PARAMETRIC, QueryTrace
from ingest.load import doc_id_for
from retrieve.retriever import Retriever

log = structlog.get_logger("ui.app")

TIER3_OFFER = (
    "Tier 3 (the model's own general knowledge, ungrounded) is off. "
    "Turn it on with ':tier3 on', or start with --tier3."
)

HELP = """\
  <question>      ask; the answer's tier is always shown
  :tier3 on|off   §9 Tier 3 — ungrounded general knowledge, off by default
  :trace on|off   per-answer telemetry line
  :help  :quit"""

_CODES = {"dim": "2", "bold": "1", "cite": "36", "warn": "33", "ok": "32"}


@dataclass(frozen=True)
class Style:
    """ANSI wrapper. Disabled it returns its input, so every renderer is testable as text."""

    enabled: bool = False

    def __call__(self, name: str, text: str) -> str:
        if not self.enabled or not text:
            return text
        return f"\x1b[{_CODES[name]}m{text}\x1b[0m"

    def link(self, target: str | None, label: str) -> str:
        if not self.enabled or not target:
            return label
        return f"\x1b]8;;{target}\x1b\\{label}\x1b]8;;\x1b\\"


@dataclass(frozen=True)
class DocRegistry:
    """`doc_id` -> the file it came from, so a citation can name and open the real document.

    Built by hashing the source files with `ingest.load.doc_id_for` — the same sha256 of
    file bytes that became `Document.doc_id`, so nothing here has to be kept in sync.
    """

    names: dict[str, str] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def scan(cls, roots: Iterable[str | Path] | None) -> DocRegistry:
        names: dict[str, str] = {}
        paths: dict[str, Path] = {}
        for root in roots or ():
            source = Path(root)
            files = sorted(source.rglob("*.pdf")) if source.is_dir() else [source]
            for path in files:
                try:
                    doc_id = doc_id_for(path.read_bytes())
                except OSError as exc:
                    log.warning("ui.document_unreadable", path=str(path), error=str(exc))
                    continue
                names[doc_id] = path.name
                paths[doc_id] = path.resolve()
        log.info("ui.documents_registered", n=len(paths))
        return cls(names, paths)

    def link_for(self, chunk: Chunk) -> str | None:
        path = self.paths.get(chunk.doc_id)
        return None if path is None else f"{path.as_uri()}#page={chunk.page_start}"


def tier_line(result: AnswerResult, style: Style) -> str:
    """§9: the tier is on every answer, before anything else about it."""
    label = f"Tier {result.tier} · {result.tier_label}"
    if result.tier == TIER_PARAMETRIC:
        return style("warn", f"⚠ {label} — not from your materials")
    if result.abstained:
        return style("dim", f"{label} — abstained, nothing above the score threshold")
    n = len(result.answer.citations)
    return style("ok", f"{label} — {n} citation{'' if n == 1 else 's'}")


def citation_lines(result: AnswerResult, registry: DocRegistry, style: Style) -> list[str]:
    """`[n] doc / p.N`, clickable, numbered as the model saw the block (§4 numbering contract).

    Tier 3 never reaches here with citations: `answer/prompt.py` strips its markers and
    leaves the list empty, and §9 forbids inventing one.
    """
    numbers = {id(hit): i + 1 for i, hit in enumerate(result.context)}
    lines: list[str] = []
    for hit in result.answer.citations:
        number = numbers.get(id(hit))
        marker = f"[{number}]" if number else "[?]"
        label = provenance(hit.chunk, registry.names)
        link = style.link(registry.link_for(hit.chunk), label)
        lines.append(f"  {style('cite', marker)} {style('cite', link)}")
    return lines


def trace_line(trace: QueryTrace, style: Style) -> str:
    top = "—" if trace.top_score is None else f"{trace.top_score:.3f}"
    ttft = "—" if trace.ttft_ms is None else f"{trace.ttft_ms:.0f}ms"
    devices = " ".join(f"{d.model}:{d.device}" for d in trace.devices)
    return style(
        "dim",
        f"  {trace.trace_id[:8]} · top {top} · {trace.n_retrieved} retrieved · "
        f"ttft {ttft} · total {trace.total_ms:.0f}ms · {devices}",
    )


def render_result(
    result: AnswerResult,
    *,
    registry: DocRegistry | None = None,
    style: Style | None = None,
    tier3_enabled: bool = False,
    streamed: str | None = None,
    show_trace: bool = False,
) -> str:
    """Everything printed after the last token. `streamed` is the raw draft already on screen."""
    registry = registry or DocRegistry()
    style = style or Style()
    lines = [tier_line(result, style)]

    if result.tier == TIER_PARAMETRIC:
        lines += ["", *_framed(result.answer.text, style)]
    else:
        lines += _correction(result, streamed, style)

    if result.abstained and not tier3_enabled:
        lines += ["", style("dim", TIER3_OFFER)]

    cites = citation_lines(result, registry, style)
    if cites:
        lines += ["", style("dim", "Sources"), *cites]
    if show_trace:
        lines += ["", trace_line(result.trace, style)]
    return "\n".join(lines)


def _framed(text: str, style: Style) -> list[str]:
    """§9's 'visually distinct': a warning bar down the left of every line, icon on the first."""
    body = text.splitlines() or [""]
    return [f"  {style('warn', '▌')} {line}" for line in [f"⚠ {body[0]}", *body[1:]]]


def _correction(result: AnswerResult, streamed: str | None, style: Style) -> list[str]:
    """Reprint the answer only when `answer/cite.py` changed what was already on screen."""
    text = result.answer.text
    if streamed is None:
        return ["", text]
    if _flat(streamed) == _flat(text):
        return []
    dropped = len(find_markers(streamed)) - len(find_markers(text))
    note = (
        f"corrected — {dropped} invented citation marker{'' if dropped == 1 else 's'} removed"
        if dropped > 0
        else "corrected"
    )
    return ["", style("dim", note), text]


def _flat(text: str) -> str:
    return " ".join(text.split())


@dataclass
class Session:
    """One wired-up UI. Mutable only where the user can toggle it (§9 Tier 3 opt-in)."""

    cfg: Config
    generator: Any
    retriever: Retriever
    registry: DocRegistry = field(default_factory=DocRegistry)
    style: Style = field(default_factory=Style)
    tier3_enabled: bool = False
    show_trace: bool = False
    #: Echo tokens as they arrive. Off, only verified text is ever printed — a raw draft can
    #: carry a citation marker that `answer/cite.py` is about to remove, which is worth
    #: trading TTFT away for when the screen is being recorded.
    stream_draft: bool = True

    def ask(self, question: str, out: TextIO) -> AnswerResult:
        stream = stream_answer(
            question,
            generator=self.generator,
            cfg=self.cfg,
            retriever=self.retriever,
            tier3_enabled=self.tier3_enabled,
            doc_names=self.registry.names,
        )
        parts: list[str] = []
        for delta in stream:
            parts.append(delta)
            if self.stream_draft:
                out.write(delta)
                out.flush()
        if parts and self.stream_draft:
            out.write("\n")
        result = stream.result
        out.write(
            render_result(
                result,
                registry=self.registry,
                style=self.style,
                tier3_enabled=self.tier3_enabled,
                streamed="".join(parts) if self.stream_draft else None,
                show_trace=self.show_trace,
            )
            + "\n"
        )
        out.flush()
        return result


def handle_command(line: str, session: Session, out: TextIO) -> bool:
    """Returns False when the user asked to leave. Toggles only; no answer logic here."""
    parts = line[1:].split()
    name = parts[0].lower() if parts else ""
    value = parts[1].lower() if len(parts) > 1 else ""
    if name in {"quit", "q", "exit"}:
        return False
    if name == "help":
        out.write(HELP + "\n")
    elif name in {"tier3", "trace"} and value in {"on", "off"}:
        setattr(session, "tier3_enabled" if name == "tier3" else "show_trace", value == "on")
        out.write(session.style("dim", f"{name} {value}") + "\n")
    else:
        out.write(session.style("warn", f"unknown command {line!r}") + "\n")
        out.write(HELP + "\n")
    return True


def repl(session: Session, *, inp: TextIO | None = None, out: TextIO | None = None) -> int:
    inp = inp or sys.stdin
    out = out or sys.stdout
    out.write(
        session.style(
            "dim", f"offline study RAG · tier3 {'on' if session.tier3_enabled else 'off'}"
        )
        + "\n"
        + HELP
        + "\n"
    )
    while True:
        out.write("\n> ")
        out.flush()
        line = inp.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            if not handle_command(line, session, out):
                return 0
            continue
        try:
            session.ask(line, out)
        except (GenerationError, ValueError) as exc:
            out.write(session.style("warn", f"! {exc}") + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ui.app", description="Ask questions about your own documents, offline."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--index", default=None, help="index directory (default: the only one)")
    parser.add_argument(
        "--docs",
        action="append",
        default=[],
        metavar="PATH",
        help="source PDF or folder, so citations can name and open the file",
    )
    parser.add_argument("-q", "--question", default=None, help="ask one question and exit")
    parser.add_argument(
        "--tier3", action="store_true", help="§9 Tier 3: allow ungrounded general knowledge"
    )
    parser.add_argument("--trace", action="store_true", help="print a telemetry line per answer")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--no-stream", action="store_true", help="print only verified answers, never the draft"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="stage logs on stderr")
    return parser


def colour_enabled(no_color: bool, out: TextIO) -> bool:
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(out, "isatty", lambda: False)())


def configure_logging(verbose: bool = False) -> None:
    """Send `structlog` to stderr and quieten it — here, stdout is the answer.

    Called from `main()` and nowhere else: importing this module must not reconfigure
    logging for `eval/run.py` or the ingest pipeline, whose stdout is not a display.
    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO if verbose else logging.WARNING
        ),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
    )


def build_session(args: argparse.Namespace, *, out: TextIO) -> Session:
    """Wiring, and nothing else: config, index, model, document names."""
    from answer.generate import load_generator
    from retrieve.dense import DenseRetriever

    cfg = load_config(args.config)
    return Session(
        cfg=cfg,
        generator=load_generator(cfg),
        retriever=DenseRetriever.open(cfg, args.index),
        registry=DocRegistry.scan(args.docs),
        style=Style(enabled=colour_enabled(args.no_color, out)),
        tier3_enabled=args.tier3,
        show_trace=args.trace,
        stream_draft=not args.no_stream,
    )


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        session = build_session(args, out=out)
    except (OSError, RuntimeError, ValueError) as exc:
        out.write(f"{type(exc).__name__}: {exc}\n")
        return 2
    if args.question:
        session.ask(args.question, out)
        return 0
    try:
        return repl(session, out=out)
    except (KeyboardInterrupt, EOFError):
        out.write("\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
