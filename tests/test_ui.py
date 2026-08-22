"""Terminal UI — ARCHITECTURE.md §1 (thin, no logic), §9 (answer tier visible, Tier 3 contract).

Every test drives the real answer path with a fake generator, so what is asserted is what a
student would actually see. The literal strings checked are the two the architecture fixes
verbatim (`ABSTAIN_MESSAGE`, `TIER3_DISCLAIMER`) and the `doc / p.N` provenance — never
model output (CLAUDE.md).
"""

from __future__ import annotations

import io

import pymupdf
import pytest

from answer.cite import find_markers
from answer.generate import generate_answer
from answer.prompt import ABSTAIN_MESSAGE, TIER3_DISCLAIMER
from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Chunk, Retrieved
from core.telemetry import TIER_LOCAL_INDEX, TIER_PARAMETRIC
from ingest.load import doc_id_for
from ui.app import (
    TIER3_OFFER,
    DocRegistry,
    Session,
    Style,
    build_parser,
    citation_lines,
    handle_command,
    main,
    render_result,
    repl,
)

DOC_ID = "sha256:deadbeefcafef00d"


def make_hit(rank: int, *, score: float = 0.9, page: int = 12) -> Retrieved:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        doc_id=DOC_ID,
        page_start=page,
        page_end=page,
        block_ids=[f"b{rank}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3", "3.2 Photosynthesis"],
        text=f"body of block {rank}",
        token_count=4,
        lang="en",
        script="latn",
    )
    return Retrieved(chunk=chunk, score=score, rank=rank)


class FakeGenerator:
    """Scripted `StreamingGenerator`; one list of pieces per generation pass."""

    def __init__(self, *passes: list[str]) -> None:
        self.passes = list(passes) or [["an answer [1]"]]
        self.name = "fake-generator"
        self.requested_device = "GPU"
        self.device = "CPU"
        self.prompts: list[str] = []
        self.last_usage = None

    def stream(self, prompt: str, settings):
        self.prompts.append(prompt)
        yield from self.passes[min(len(self.prompts) - 1, len(self.passes) - 1)]


class FakeRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        return self._hits[:k]


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=tmp_path / "ov_cache")
        }
    )


@pytest.fixture
def hits():
    return [make_hit(i, score=0.9 - 0.1 * i, page=10 + i) for i in range(1, 4)]


def grounded(cfg, hits, *passes):
    return generate_answer("q", generator=FakeGenerator(*passes), cfg=cfg, hits=hits)


def ungrounded(cfg, *passes, tier3: bool = True):
    low = [make_hit(1, score=cfg.retrieve.tau - 0.01)]
    return generate_answer(
        "q", generator=FakeGenerator(*passes), cfg=cfg, hits=low, tier3_enabled=tier3
    )


def session_for(cfg, hits, *passes, **kw) -> tuple[Session, io.StringIO]:
    out = io.StringIO()
    session = Session(
        cfg=cfg,
        generator=FakeGenerator(*passes),
        retriever=FakeRetriever(hits),
        registry=DocRegistry(names={DOC_ID: "bio.pdf"}),
        **kw,
    )
    return session, out


# --- §9: the tier is visible on every answer ------------------------------------------------


def test_tier_1_grounded_answer_shows_its_tier(cfg, hits):
    rendered = render_result(grounded(cfg, hits, ["an answer [1]"]))
    assert "Tier 1" in rendered
    assert "local index" in rendered
    assert "Tier 3" not in rendered


def test_tier_1_abstention_shows_its_tier_and_says_it_abstained(cfg):
    result = generate_answer("q", generator=FakeGenerator(), cfg=cfg, hits=[])
    rendered = render_result(result)
    assert "Tier 1" in rendered
    assert "abstained" in rendered


def test_tier_3_answer_shows_its_tier_and_that_it_is_not_from_your_materials(cfg):
    rendered = render_result(ungrounded(cfg, ["general knowledge body"]))
    assert "Tier 3" in rendered
    assert "not from your materials" in rendered.splitlines()[0]


def test_the_ui_does_not_choose_the_tier_itself(cfg, hits):
    """Tier 3 enabled but Tier 1 answered: §9's order is decided in `answer/`, not here."""
    result = generate_answer(
        "q", generator=FakeGenerator(["grounded [1]"]), cfg=cfg, hits=hits, tier3_enabled=True
    )
    assert result.tier == TIER_LOCAL_INDEX
    assert "Tier 1" in render_result(result, tier3_enabled=True)


# --- §9: the Tier 3 contract ----------------------------------------------------------------


def test_tier_3_is_visually_distinct_and_carries_the_literal_disclaimer(cfg):
    rendered = render_result(ungrounded(cfg, ["general knowledge body"]))
    body = [line for line in rendered.splitlines() if line.startswith("  ▌")]
    assert TIER3_DISCLAIMER in rendered
    assert "⚠" in body[0]
    assert len(body) == len(ungrounded(cfg, ["general knowledge body"]).answer.text.splitlines())


def test_tier_3_emits_no_citation_markers_and_lists_no_sources(cfg):
    """§9: never fabricate a `[n]` or a page number — even when the model wrote some."""
    result = ungrounded(cfg, ["invented [1] and [2] markers"])
    rendered = render_result(result)
    assert result.tier == TIER_PARAMETRIC
    assert find_markers(rendered) == []
    assert result.answer.citations == []
    assert "Sources" not in rendered


def test_tier_1_abstention_is_stated_before_tier_3_is_offered(cfg):
    session, out = session_for(cfg, [], ["ungrounded body"], tier3_enabled=True)
    session.ask("q", out)
    text = out.getvalue()
    assert text.index(ABSTAIN_MESSAGE) < text.index(TIER3_DISCLAIMER)


def test_an_abstention_offers_tier_3_only_while_it_is_off(cfg):
    session, out = session_for(cfg, [], tier3_enabled=False)
    session.ask("q", out)
    assert ABSTAIN_MESSAGE in out.getvalue()
    assert TIER3_OFFER in out.getvalue()
    assert TIER3_OFFER not in render_result(ungrounded(cfg), tier3_enabled=True)


# --- citations ------------------------------------------------------------------------------


def test_citations_render_as_doc_slash_page(cfg, hits):
    registry = DocRegistry(names={DOC_ID: "bio.pdf"})
    (line,) = citation_lines(grounded(cfg, hits, ["an answer [1]"]), registry, Style())
    assert line.strip().startswith("[1] bio.pdf / p.11")


def test_citation_numbers_are_the_block_numbers_the_model_cited(cfg, hits):
    """§4 numbering: `[n]` is context position n, not the order citations were collected."""
    lines = citation_lines(
        grounded(cfg, hits, ["only the third block [3]"]), DocRegistry(), Style()
    )
    assert len(lines) == 1
    assert lines[0].strip().startswith("[3] ")
    assert "p.13" in lines[0]


def test_a_citation_is_a_clickable_link_to_the_cited_page(tmp_path, cfg, hits):
    pdf = tmp_path / "bio.pdf"
    doc = pymupdf.open()
    doc.new_page(width=200, height=200)
    pdf.write_bytes(doc.tobytes())
    registry = DocRegistry.scan([tmp_path])
    hit = make_hit(1, page=42)
    hit = Retrieved(
        chunk=Chunk(**{**hit.chunk.to_row(), "doc_id": doc_id_for(pdf.read_bytes())}),
        score=0.9,
        rank=1,
    )
    (line,) = citation_lines(grounded(cfg, [hit], ["an answer [1]"]), registry, Style(enabled=True))
    assert pdf.as_uri() + "#page=42" in line
    assert "bio.pdf / p.42" in line


def test_an_unknown_document_still_cites_a_page_without_a_link(cfg, hits):
    (line,) = citation_lines(grounded(cfg, hits, ["an answer [1]"]), DocRegistry(), Style(True))
    assert "file://" not in line
    assert "/ p.11" in line


def test_a_grounded_answer_lists_its_sources(cfg, hits):
    rendered = render_result(
        grounded(cfg, hits, ["two blocks [1][2]"]), registry=DocRegistry(names={DOC_ID: "bio.pdf"})
    )
    assert "Sources" in rendered
    assert rendered.count("bio.pdf / p.") == 2


# --- the streamed draft vs the verified answer ----------------------------------------------


def test_a_clean_answer_is_not_reprinted_after_streaming(cfg, hits):
    session, out = session_for(cfg, hits, ["grounded answer [1]"])
    session.ask("q", out)
    assert out.getvalue().count("grounded answer") == 1


def test_an_invented_marker_is_removed_and_the_correction_is_shown(cfg, hits):
    session, out = session_for(cfg, hits, ["answer [1] and [99]"])
    result = session.ask("q", out)
    text = out.getvalue()
    assert "invented citation marker" in text
    assert find_markers(result.answer.text) == [1]
    assert text.rstrip().endswith(citation_lines(result, session.registry, Style())[-1])


def test_a_caller_that_did_not_stream_gets_the_answer_text(cfg, hits):
    rendered = render_result(grounded(cfg, hits, ["an answer [1]"]), streamed=None)
    assert "an answer [1]" in rendered


# --- documents ------------------------------------------------------------------------------


def test_documents_are_registered_by_the_hash_that_became_their_doc_id(tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=200, height=200)
    (tmp_path / "notes.pdf").write_bytes(doc.tobytes())
    registry = DocRegistry.scan([tmp_path])
    doc_id = doc_id_for((tmp_path / "notes.pdf").read_bytes())
    assert registry.names == {doc_id: "notes.pdf"}
    assert registry.paths[doc_id] == (tmp_path / "notes.pdf").resolve()


def test_an_unreadable_document_is_skipped_not_fatal(tmp_path):
    assert DocRegistry.scan([tmp_path / "missing.pdf"]).names == {}
    assert DocRegistry.scan(None).paths == {}


# --- presentation ---------------------------------------------------------------------------


def test_disabled_style_emits_no_escape_sequences(cfg, hits):
    rendered = render_result(grounded(cfg, hits, ["an answer [1]"]), style=Style(enabled=False))
    assert "\x1b" not in rendered


def test_enabled_style_colours_the_tier_line(cfg, hits):
    rendered = render_result(grounded(cfg, hits, ["an answer [1]"]), style=Style(enabled=True))
    assert rendered.splitlines()[0].startswith("\x1b[")


def test_the_trace_line_is_opt_in(cfg, hits):
    result = grounded(cfg, hits, ["an answer [1]"])
    assert result.trace.trace_id[:8] not in render_result(result)
    assert result.trace.trace_id[:8] in render_result(result, show_trace=True)


# --- commands and the loop ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "attribute", "expected"),
    [
        (":tier3 on", "tier3_enabled", True),
        (":tier3 off", "tier3_enabled", False),
        (":trace on", "show_trace", True),
    ],
)
def test_commands_toggle_only_what_they_name(cfg, command, attribute, expected):
    session, out = session_for(cfg, [], tier3_enabled=True, show_trace=False)
    session.tier3_enabled = attribute != "tier3_enabled"
    assert handle_command(command, session, out) is True
    assert getattr(session, attribute) is expected


def test_an_unknown_command_prints_help_and_keeps_going(cfg):
    session, out = session_for(cfg, [])
    assert handle_command(":sudo", session, out) is True
    assert "unknown command" in out.getvalue()
    assert ":tier3 on|off" in out.getvalue()


def test_quit_stops_the_loop(cfg):
    session, out = session_for(cfg, [])
    assert handle_command(":quit", session, out) is False


def test_the_loop_answers_questions_and_ends_at_end_of_input(cfg, hits):
    session, out = session_for(cfg, hits, ["an answer [1]"])
    assert repl(session, inp=io.StringIO("what is photosynthesis?\n\n:quit\n"), out=out) == 0
    assert "Tier 1" in out.getvalue()


def test_a_failed_answer_does_not_kill_the_session(cfg, hits):
    session, out = session_for(cfg, hits, ["an answer [1]"])

    def explode(question, out_stream):
        raise ValueError("index went away")

    session.ask = explode
    assert repl(session, inp=io.StringIO("q\n"), out=out) == 0
    assert "index went away" in out.getvalue()


# --- entry point ----------------------------------------------------------------------------


def test_tier3_defaults_to_off_on_the_command_line():
    assert build_parser().parse_args([]).tier3 is False
    assert build_parser().parse_args(["--tier3"]).tier3 is True


def test_wiring_failure_is_reported_without_a_traceback(tmp_path):
    out = io.StringIO()
    assert main(["--config", str(tmp_path / "nope.yaml")], out=out) == 2
    assert "nope.yaml" in out.getvalue()
