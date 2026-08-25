"""Browser UI — §1 (thin), §9 (tier visible), §0.3 (no network at runtime)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from core.config import DEFAULT_CONFIG, load_config
from core.schema import Chunk, Retrieved
from ui.app import DocRegistry
from ui.web import (
    PAGE,
    SNIPPET_CHARS,
    Backend,
    Handler,
    citations_of,
    index_of,
    telemetry_of,
)

DOC = "sha256:deadbeefcafef00d"


def make_chunk(cid: str, page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=DOC,
        page_start=page,
        page_end=page,
        block_ids=[f"b{cid}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3", "3.2 Logic"],
        text=text,
        token_count=5,
        lang="en",
        script="latn",
    )


class FakeRetriever:
    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        return [
            Retrieved(
                chunk=make_chunk("c1", 62, "George Boole formalised boolean algebra."),
                score=0.9,
                rank=1,
            ),
            Retrieved(
                chunk=make_chunk("c2", 63, "Logic gates use HIGH and LOW."), score=0.7, rank=2
            ),
        ]


class LongChunkRetriever:
    """One hit whose text is longer than a citation card should ever show."""

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        text = "Boole wrote An Investigation of the Laws of Thought.\n" * 20
        return [Retrieved(chunk=make_chunk("c1", 62, text), score=0.9, rank=1)]


class FakeGen:
    name, requested_device, device = "fake", "GPU", "CPU"

    def __init__(self, pieces: tuple[str, ...] = ("George ", "Boole", " [1]")) -> None:
        self.pieces = pieces

    def stream(self, prompt: str, settings):
        yield from self.pieces


@pytest.fixture
def backend(tmp_path):
    cfg = load_config(DEFAULT_CONFIG)
    pdf = tmp_path / "logic.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    registry = DocRegistry({DOC: "logic.pdf"}, {DOC: pdf})
    return Backend(cfg, FakeRetriever(), FakeGen(), registry)


@pytest.fixture
def server(backend):
    Handler.backend = backend
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url: str) -> str:
    return urllib.request.urlopen(url, timeout=30).read().decode()


def sse_done(body: str) -> dict:
    """The final `done` payload out of the event stream."""
    for line in body.splitlines():
        if line.startswith("data: ") and '"tier_label"' in line:
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"no done event in stream: {body[:300]}")


# --- §0.3: nothing may reach the network -----------------------------------------------


def test_the_page_loads_no_remote_asset():
    """A CDN font or script would be a runtime network call, which §0.3 forbids."""
    html = PAGE.read_text(encoding="utf-8")
    for scheme in ("https://", "//fonts.", "cdn.", "unpkg", "jsdelivr"):
        assert scheme not in html, f"page references {scheme!r}"


def test_the_server_module_imports_no_web_framework():
    """§0.1 pins the dependency set and carries no web framework.

    Checks imports, not prose — the module docstring names the frameworks it declines to use.
    """
    import ast

    tree = ast.parse(Path("ui/web.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"streamlit", "fastapi", "flask", "gradio", "uvicorn", "django"}
    assert "http" in imported, "expected the stdlib server"


# --- routes -----------------------------------------------------------------------------


def test_root_serves_the_page(server):
    assert "Study Assistant" in get(server + "/")


def test_meta_reports_the_corpus(server):
    meta = json.loads(get(server + "/meta"))
    assert meta["documents"] == 1
    assert meta["examples"]


def test_meta_names_the_device_the_generator_actually_got(server):
    """§7.4 — the page's own footer must not claim the device the config asked for."""
    stack = dict(json.loads(get(server + "/meta"))["stack"])

    assert stack["generator"] == "fake"
    assert stack["device"] == "CPU (asked GPU)"
    assert stack["retrieval"] == "dense"


def test_meta_counts_chunks_behind_a_hybrid_retriever(backend):
    """`HybridRetriever` keeps the index on its dense arm; reading `.index` reports zero."""

    class Frame:
        height = 4210

    class Index:
        frame = Frame()
        embedder = {"hf_id": "BAAI/bge-m3", "precision": "int8"}

    class Hybrid:
        dense = type("Dense", (), {"index": Index()})()

    backend.retriever = Hybrid()
    meta = backend.meta()

    assert index_of(Hybrid()) is not None
    assert meta["chunks"] == 4210
    assert dict(meta["stack"])["embedder"] == "bge-m3 int8"


def test_unknown_route_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + "/nope")
    assert e.value.code == 404


def test_an_empty_question_is_refused(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + "/ask?q=%20%20")
    assert e.value.code == 400


def test_a_registered_document_is_served_for_citation_links(server):
    assert get(server + f"/doc/{DOC}").startswith("%PDF")


def test_an_unregistered_document_is_404_not_a_path_escape(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + "/doc/../../etc/passwd")
    assert e.value.code == 404


# --- answering ---------------------------------------------------------------------------


def test_answer_streams_then_reports_a_verified_result(server):
    body = get(server + "/ask?q=who%20made%20boolean%20algebra")

    assert "event: delta" in body, "no streaming deltas"
    done = sse_done(body)
    assert done["text"]
    assert done["abstained"] is False


def test_tier_is_always_reported(server):
    """§9: the tier used must be visible on every answer."""
    assert sse_done(get(server + "/ask?q=anything"))["tier_label"]


def test_invented_markers_never_reach_the_ui(backend):
    """`[9]` indexes a block the model was never shown; cite.verify drops it."""
    backend.generator = FakeGen(("Boole [1]", " and [9]"))
    Handler.backend = backend
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        done = sse_done(get(f"http://127.0.0.1:{srv.server_address[1]}/ask?q=q"))
    finally:
        srv.shutdown()

    assert "[9]" not in done["text"]
    assert all(c["n"] != 9 for c in done["citations"])
    # The draft on screen carried `[9]`; the page says the difference out loud rather than
    # silently swapping in a cleaner answer.
    assert done["dropped"] == 1


def test_a_clean_answer_reports_nothing_dropped(server):
    assert sse_done(get(server + "/ask?q=q"))["dropped"] == 0


# --- rendering helpers -------------------------------------------------------------------


def test_citations_carry_openable_provenance(backend):
    from answer.generate import generate_answer

    result = generate_answer(
        "q", generator=backend.generator, cfg=backend.cfg, retriever=backend.retriever
    )
    cites = citations_of(result, backend.registry)

    assert cites, "no citations rendered"
    assert cites[0]["doc"] == "logic.pdf"
    assert "p.62" in cites[0]["where"]
    assert cites[0]["url"] == f"/doc/{DOC}#page=62"


def test_citations_carry_the_cited_text_so_grounding_is_visible(backend):
    from answer.generate import generate_answer

    backend.retriever = LongChunkRetriever()
    result = generate_answer(
        "q", generator=backend.generator, cfg=backend.cfg, retriever=backend.retriever
    )
    snippet = citations_of(result, backend.registry)[0]["snippet"]

    assert snippet.startswith("Boole")
    assert "\n" not in snippet, "snippet should be flowed, not raw layout"
    assert len(snippet) <= SNIPPET_CHARS + 1, "long chunks must be trimmed, not pasted whole"
    assert snippet.endswith("…")


def test_citation_url_is_absent_when_the_pdf_is_not_registered(backend):
    from answer.generate import generate_answer

    backend.registry = DocRegistry({DOC: "logic.pdf"}, {})
    result = generate_answer(
        "q", generator=backend.generator, cfg=backend.cfg, retriever=backend.retriever
    )
    assert citations_of(result, backend.registry)[0]["url"] is None


def test_telemetry_names_the_device_actually_used(backend):
    """§7.4 — a silent CPU fallback must be visible, not hidden behind the config's wish."""
    from answer.generate import generate_answer

    result = generate_answer(
        "q", generator=backend.generator, cfg=backend.cfg, retriever=backend.retriever
    )
    tele = telemetry_of(result)

    assert "CPU" in tele["device"]
    assert "asked GPU" in tele["device"]
    assert tele["retrieved"] == "2"
