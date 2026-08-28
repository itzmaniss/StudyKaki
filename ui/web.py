"""Browser UI — ARCHITECTURE.md §1 (thin), §9 (tier visible on every answer).

Rendering and transport only; nothing here decides anything about an answer.

Stdlib only — §0.1 carries no web framework, so this is `http.server` and one HTML file.
Localhost only — §0.3 forbids the network, so the server binds 127.0.0.1, the page loads no
CDN asset, and PDFs come from the local registry. It works with the wifi off, which is the
pitch.

`ui/app.py` remains the reference UI; this is the same `Session` wiring behind a page.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from collections.abc import Sequence
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import structlog

from answer.generate import AnswerResult, GenerationError, stream_answer
from answer.prompt import TIER3_DISCLAIMER
from core.config import DEFAULT_CONFIG, Config, load_config
from ui.app import DocRegistry, configure_logging

log = structlog.get_logger("ui.web")

PAGE = Path(__file__).resolve().parent / "index.html"

#: Demo chips, so nobody has to type first. Every one must clear `tau` on the shipped
#: config — an opening abstention is worse than no demo. Two of the previous three failed
#: that: "Who formalised boolean algebra?" tops out at 0.445 against tau 0.45 (BLOCKERS
#: #18), the short Tamil phrasing at 0.365. These three are verbatim from golden.jsonl and
#: score 0.647 / 0.608 / 0.781, one per language. Re-check after any embedder, chunking or
#: tau change.
EXAMPLES = (
    "In which year did the personal computer first appear?",
    "பூலியன் இயற்கணிதத்தை உருவாக்கியவர் யார்?",
    "我国IPv6地址数量是多少？",
)

#: Enough of the cited block to recognise the answer in it, short enough not to bury the page.
SNIPPET_CHARS = 220


def index_of(retriever: Any) -> Any:
    """The `DenseIndex` behind whichever retriever is wired.

    `HybridRetriever` holds its index on its dense arm, so reading `.index` alone would
    report an empty corpus on the configuration §10 ships.
    """
    index = getattr(retriever, "index", None)
    if index is None:
        index = getattr(getattr(retriever, "dense", None), "index", None)
    return index


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def stack_of(cfg: Config, generator: Any, index: Any) -> list[list[str]]:
    """What actually loaded, for the strip along the bottom of the page.

    §7.4 — the device the load got, not the one asked for, and a fallback says so. Claiming
    GPU while running on CPU is the lie the telemetry exists to prevent.
    """
    fingerprint = getattr(index, "embedder", {}) or {}
    device = str(getattr(generator, "device", "") or "—")
    requested = str(getattr(generator, "requested_device", "") or device)
    embedder = _short(str(fingerprint.get("hf_id", "")))
    precision = str(fingerprint.get("precision", ""))
    return [
        ["generator", _short(str(getattr(generator, "name", "") or "—"))],
        ["embedder", f"{embedder} {precision}".strip() or "—"],
        ["retrieval", "dense + BM25 · RRF" if cfg.retrieve.hybrid.enabled else "dense"],
        ["device", device + (f" (asked {requested})" if device != requested else "")],
    ]


class Backend:
    """Everything a request needs. Built once, shared across threads."""

    def __init__(self, cfg: Config, retriever: Any, generator: Any, registry: DocRegistry) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.generator = generator
        self.registry = registry
        # One InferRequest per model, so concurrent questions corrupt each other. A study
        # tool has one user; serialise rather than pretend otherwise.
        self.lock = threading.Lock()

    def meta(self) -> dict[str, Any]:
        index = index_of(self.retriever)
        return {
            "documents": len(self.registry.paths),
            "chunks": int(index.frame.height) if index is not None else 0,
            "examples": list(EXAMPLES),
            "stack": stack_of(self.cfg, self.generator, index),
        }


def citations_of(result: AnswerResult, registry: DocRegistry) -> list[dict[str, Any]]:
    """Verified citations only — `answer/cite.py` already dropped invented markers.

    Numbered by position in `result.context`, not in the citation list: §4 says `[n]` is the
    n-th block the model saw, so an answer citing only `[3]` must render as `[3]`.
    """
    numbers = {id(hit): i for i, hit in enumerate(result.context, start=1)}
    out = []
    for hit in result.answer.citations:
        n = numbers.get(id(hit))
        if n is None:
            continue
        chunk = hit.chunk
        pages = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp.{chunk.page_start}-{chunk.page_end}"
        )
        heading = " › ".join(h for h in chunk.heading_path if h)
        out.append(
            {
                "n": n,
                "doc": registry.names.get(chunk.doc_id, chunk.doc_id.removeprefix("sha256:")[:8]),
                "where": f"{pages}{' · ' + heading if heading else ''}",
                "snippet": snippet_of(chunk.text),
                "url": f"/doc/{chunk.doc_id}#page={chunk.page_start}"
                if chunk.doc_id in registry.paths
                else None,
            }
        )
    return out


def snippet_of(text: str) -> str:
    """The head of the cited block, so grounding is legible without opening the PDF.

    Sliced on characters, not words — Chinese has no spaces, so one word count would return
    a whole block for zh and a fragment for en.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS].rstrip() + "…"


def telemetry_of(result: AnswerResult) -> dict[str, str]:
    t = result.trace
    out: dict[str, str] = {}
    if getattr(t, "ttft_ms", None):
        out["ttft"] = f"{t.ttft_ms:.0f} ms"
    if getattr(t, "total_ms", None):
        out["total"] = f"{t.total_ms / 1000:.1f} s"
    # §7.4 — say which device we actually got, and flag it when that is not what was asked for.
    for use in getattr(t, "devices", ()):
        out["device"] = use.device + (f" (asked {use.requested})" if use.fell_back else "")
    if getattr(t, "completion_tokens", 0):
        out["tokens"] = str(t.completion_tokens)
    out["retrieved"] = str(len(result.hits))
    out["in context"] = str(len(result.context))
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backend: Backend

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("ui.request", request=fmt % args)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        route = urlparse(self.path)
        if route.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif route.path == "/meta":
            self._send(200, "application/json", json.dumps(self.backend.meta()).encode())
        elif route.path == "/ask":
            self._ask(parse_qs(route.query))
        elif route.path.startswith("/doc/"):
            self._doc(route.path[len("/doc/") :])
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _doc(self, doc_id: str) -> None:
        path = self.backend.registry.paths.get(doc_id)
        if path is None or not path.is_file():
            self._send(404, "text/plain", b"document not registered")
            return
        self._send(200, "application/pdf", path.read_bytes())

    def _event(self, name: str, payload: Any) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _ask(self, params: dict[str, list[str]]) -> None:
        question = (params.get("q") or [""])[0].strip()
        if not question:
            self._send(400, "text/plain", b"empty question")
            return
        tier3 = (params.get("tier3") or ["0"])[0] == "1"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        b = self.backend
        try:
            with b.lock:
                stream = stream_answer(
                    question,
                    generator=b.generator,
                    cfg=b.cfg,
                    retriever=b.retriever,
                    tier3_enabled=tier3,
                    doc_names=b.registry.names,
                )
                for delta in stream:
                    self._event("delta", delta)
                result = stream.result
        except (GenerationError, RuntimeError, ValueError, OSError) as exc:
            log.warning("ui.ask_failed", error=str(exc))
            self._event("failed", f"{type(exc).__name__}: {exc}")
            return
        except BrokenPipeError:
            return

        self._event(
            "done",
            {
                "text": result.answer.text,
                "tier": result.tier,
                "tier_label": result.tier_label,
                "abstained": result.abstained,
                # Two failures wearing one word (BLOCKERS #17): nothing above tau, versus
                # the model refusing pages that cleared it. The card says which.
                "model_abstained": result.model_abstained,
                # Markers that pointed at nothing. The draft on screen still shows them, so
                # the count is the difference between looking clean and being checked.
                "dropped": result.markers_emitted - result.markers_grounded,
                # §9: `Answer` has no tier field, so the disclaimer is the signal.
                "ungrounded": TIER3_DISCLAIMER in result.answer.text,
                "citations": citations_of(result, b.registry),
                "telemetry": telemetry_of(result),
            },
        )


def build_backend(args: argparse.Namespace) -> Backend:
    """Wiring, and nothing else."""
    from answer.generate import load_generator
    from retrieve.dense import DenseRetriever

    cfg = load_config(args.config)
    if cfg.retrieve.hybrid.enabled:
        from retrieve.fusion import HybridRetriever

        retriever: Any = HybridRetriever.open(cfg, args.index)
    else:
        retriever = DenseRetriever.open(cfg, args.index)
    return Backend(cfg, retriever, load_generator(cfg), DocRegistry.scan(args.docs))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Browser UI (ARCHITECTURE.md §1, §9)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--index", type=Path, default=None, help="index dir; required if >1 exists")
    ap.add_argument("--docs", action="append", default=None, help="PDF dir, for openable cites")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        backend = build_backend(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    handler = partial(Handler)
    handler.backend = backend  # type: ignore[attr-defined]
    Handler.backend = backend

    # 127.0.0.1, never 0.0.0.0: §0.3's promise is that nothing leaves the machine.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Study Assistant → {url}   (ctrl-c to stop)")
    log.info("ui.serving", url=url, hybrid=backend.cfg.retrieve.hybrid.enabled)
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
