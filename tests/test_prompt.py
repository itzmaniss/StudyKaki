"""Prompt construction, abstain, and the tier contract — ARCHITECTURE.md §4, §9, §0.6.

Assertions cover structure, provenance and error handling. Nothing here asserts on model
output text; the only literal strings checked are the two the architecture fixes verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from answer.cite import find_markers
from answer.prompt import (
    ABSTAIN_MESSAGE,
    SYSTEM_INSTRUCTION,
    TIER3_DISCLAIMER,
    TIER3_SYSTEM_INSTRUCTION,
    Tier3DisabledError,
    abstain_answer,
    build_prompt,
    build_tier3_prompt,
    estimate_tokens,
    fit_context,
    format_context,
    language_reminder_for,
    render_tier3_answer,
)
from answer.sources.online import DisabledOnlineSource, OnlineSource
from core.schema import Chunk, Retrieved


def make_hit(rank: int, **kw) -> Retrieved:
    base = dict(
        chunk_id=f"c{rank}",
        doc_id="sha256:deadbeefcafef00d",
        page_start=rank,
        page_end=rank,
        block_ids=[f"b{rank}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3", "3.2 Photosynthesis"],
        text=f"body of block {rank}",
        token_count=4,
        lang="en",
        script="latn",
    )
    return Retrieved(chunk=Chunk(**{**base, **kw}), score=1.0 / rank, rank=rank)


@pytest.fixture
def hits() -> list[Retrieved]:
    return [make_hit(i) for i in range(1, 4)]


# --- numbered context blocks (§4) ----------------------------------------------------


def test_blocks_are_numbered_from_one_in_list_order(hits):
    """The numbering contract cite.verify relies on: `[n]` is the n-th hit passed in."""
    context = format_context(hits)
    assert context.index("[1]") < context.index("[2]") < context.index("[3]")
    for number, hit in enumerate(hits, start=1):
        assert f"[{number}]" in context
        assert hit.chunk.text in context


def test_numbering_follows_list_position_not_retrieved_rank():
    reordered = [make_hit(9), make_hit(4)]
    context = format_context(reordered)
    assert context.startswith("[1] ")
    assert "[2] " in context
    assert "[9]" not in context


def test_each_block_carries_doc_and_page_provenance(hits):
    context = format_context(hits)
    # sha256: prefix stripped, short id kept — Chunk has no filename, Document does.
    assert "deadbeef / p.1" in context
    assert "deadbeef / p.2" in context


def test_page_range_is_rendered_when_a_chunk_spans_pages():
    context = format_context([make_hit(1, page_start=7, page_end=9)])
    assert "pp.7-9" in context


def test_heading_path_is_included_and_omitted_when_absent():
    with_heading = format_context([make_hit(1)])
    assert "Chapter 3 > 3.2 Photosynthesis" in with_heading

    without = format_context([make_hit(1, heading_path=[])])
    assert ">" not in without
    assert "| " not in without


def test_doc_names_override_the_short_id(hits):
    context = format_context(hits, doc_names={"sha256:deadbeefcafef00d": "biology.pdf"})
    assert "biology.pdf / p.1" in context
    assert "deadbeef /" not in context


def test_unknown_doc_id_falls_back_to_short_id(hits):
    context = format_context(hits, doc_names={"other-doc": "wrong.pdf"})
    assert "wrong.pdf" not in context
    assert "deadbeef / p.1" in context


# --- full prompt ---------------------------------------------------------------------


def test_prompt_contains_instructions_question_and_every_block(hits):
    prompt = build_prompt("ஒளிச்சேர்க்கை என்றால் என்ன?", hits)
    assert prompt.startswith(SYSTEM_INSTRUCTION)
    assert "ஒளிச்சேர்க்கை என்றால் என்ன?" in prompt
    assert format_context(hits) in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_instructs_language_matching_citation_and_abstain(hits):
    """§4: answer in the question's language, cite block numbers, say so if context is thin."""
    prompt = build_prompt("q", hits)
    assert "same language as the question" in prompt
    assert "Never invent a block number." in prompt
    assert ABSTAIN_MESSAGE in prompt


def test_prompt_accepts_a_custom_system_instruction(hits):
    prompt = build_prompt("q", hits, system="CUSTOM")
    assert prompt.startswith("CUSTOM")


@pytest.mark.parametrize("bad", [None, 42, b"q", ["q"]])
def test_prompt_rejects_non_string_question(bad, hits):
    with pytest.raises(TypeError, match="question"):
        build_prompt(bad, hits)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_prompt_rejects_blank_question(blank, hits):
    with pytest.raises(ValueError, match="question"):
        build_prompt(blank, hits)


def test_prompt_rejects_empty_context_because_that_is_an_abstain():
    with pytest.raises(ValueError, match="abstain"):
        build_prompt("q", [])


@pytest.mark.parametrize("bad", [None, 42, object()])
def test_prompt_rejects_non_sequence_hits(bad):
    with pytest.raises(TypeError, match="hits"):
        build_prompt("q", bad)


def test_prompt_rejects_hits_that_are_not_retrieved(hits):
    with pytest.raises(TypeError, match=r"hits\[1\]"):
        build_prompt("q", [hits[0], "not a hit"])


# --- abstain (§0.6) ------------------------------------------------------------------


def test_abstain_message_is_the_literal_the_architecture_fixes():
    assert ABSTAIN_MESSAGE == "I couldn't find this in your documents."


def test_abstain_answer_carries_no_citations():
    answer = abstain_answer("trace-1")
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.abstained is True
    assert answer.citations == []
    assert answer.trace_id == "trace-1"


@pytest.mark.parametrize("bad", [None, 42])
def test_abstain_answer_rejects_non_string_trace_id(bad):
    with pytest.raises(TypeError, match="trace_id"):
        abstain_answer(bad)


def test_abstain_answer_rejects_empty_trace_id():
    with pytest.raises(ValueError, match="trace_id"):
        abstain_answer("")


# --- Tier 3: model parametric knowledge (§9) -----------------------------------------


def test_tier3_disclaimer_is_the_literal_the_architecture_fixes():
    assert TIER3_DISCLAIMER == (
        "General knowledge — not from your materials. May not match your syllabus."
    )


def test_tier3_prompt_is_off_by_default():
    with pytest.raises(Tier3DisabledError):
        build_tier3_prompt("what is photosynthesis?")


@pytest.mark.parametrize("truthy", [1, "yes", [1]])
def test_tier3_requires_a_real_boolean_opt_in(truthy):
    """A stray truthy value must not open the ungrounded path."""
    with pytest.raises(Tier3DisabledError):
        build_tier3_prompt("q", enabled=truthy)


def test_tier3_prompt_when_enabled_forbids_citations_and_pages():
    prompt = build_tier3_prompt("what is photosynthesis?", enabled=True)
    assert prompt.startswith(TIER3_SYSTEM_INSTRUCTION)
    assert "Do NOT cite anything." in prompt
    assert "what is photosynthesis?" in prompt
    assert TIER3_SYSTEM_INSTRUCTION != SYSTEM_INSTRUCTION


def test_tier3_render_is_off_by_default():
    with pytest.raises(Tier3DisabledError):
        render_tier3_answer("some general knowledge", "trace-1")


def test_tier3_answer_carries_the_disclaimer_and_no_citations():
    answer = render_tier3_answer("Photosynthesis converts light.", "trace-1", enabled=True)
    assert TIER3_DISCLAIMER in answer.text
    assert answer.citations == []
    assert answer.abstained is False
    assert answer.trace_id == "trace-1"


@pytest.mark.parametrize(
    "model_text",
    [
        "Grounded-looking [1] claim [2, 3].",
        "[1]",
        "See [0] and [99] and [1].",
        "Mixed [1] with `arr[2]` code.",
        "no markers at all",
        "",
    ],
)
def test_tier3_answer_never_emits_a_citation_marker(model_text):
    """§9: emits no citation markers, ever. The model cannot override this."""
    answer = render_tier3_answer(model_text, "trace-1", enabled=True)
    assert find_markers(answer.text) == []
    assert answer.citations == []


def test_tier3_says_tier1_abstained_before_offering_general_knowledge():
    answer = render_tier3_answer(
        "Photosynthesis converts light.", "trace-1", enabled=True, tier1_abstained=True
    )
    assert answer.text.startswith(ABSTAIN_MESSAGE)
    assert answer.text.index(ABSTAIN_MESSAGE) < answer.text.index(TIER3_DISCLAIMER)
    assert answer.abstained is False


def test_tier3_omits_the_abstain_notice_when_tier1_did_not_abstain():
    answer = render_tier3_answer("Photosynthesis converts light.", "trace-1", enabled=True)
    assert not answer.text.startswith(ABSTAIN_MESSAGE)
    assert answer.text.startswith(TIER3_DISCLAIMER)


def test_tier3_handles_empty_model_output():
    answer = render_tier3_answer("   ", "trace-1", enabled=True)
    assert answer.text == TIER3_DISCLAIMER


@pytest.mark.parametrize("bad", [None, 42, b"text"])
def test_tier3_render_rejects_non_string_model_text(bad):
    with pytest.raises(TypeError, match="model_text"):
        render_tier3_answer(bad, "trace-1", enabled=True)


def test_tier3_render_rejects_bad_trace_id():
    with pytest.raises(ValueError, match="trace_id"):
        render_tier3_answer("text", "", enabled=True)


# --- prompt budget and context fitting (BLOCKERS #11) --------------------------------


def test_tamil_is_priced_denser_than_latin_for_the_same_character_count():
    """The whole of #11 in one assertion: equal characters, unequal token cost.

    `ingest.chunk.count_tokens` counts whitespace words and would rate these two the same,
    which is exactly why it cannot be used as a prompt budget.
    """
    latin = "a" * 300
    tamil = "த" * 300
    assert estimate_tokens(tamil) > estimate_tokens(latin) * 1.5


def test_the_estimate_never_undercounts_the_measured_tamil_rate():
    """Tamil measured at 1.10 chars/token; an estimate below that reintroduces the crash."""
    tamil = "த" * 1000
    assert estimate_tokens(tamil) >= 1000 / 1.10


def test_empty_text_costs_nothing():
    assert estimate_tokens("") == 0


def test_context_is_trimmed_from_the_tail_so_the_best_hits_survive(hits):
    """Dropping is tail-first: rank 1 is the last thing to go."""
    fitted = fit_context("q", hits, max_prompt_tokens=10_000, count_tokens=lambda text: len(text))
    assert fitted.dropped == 0
    assert [h.rank for h in fitted.hits] == [1, 2, 3]

    tight = fit_context("q", hits, max_prompt_tokens=10_000, count_tokens=lambda _: 20_000)
    assert [h.rank for h in tight.hits] == [1]
    assert tight.dropped == 2


def test_a_fitting_prefix_is_kept_whole(hits):
    """Counting only the blocks, two fit and the third does not.

    Counts block *bodies*, not `[n]` markers: SYSTEM_INSTRUCTION quotes "[1] or [2][3]" as
    its citation example, so marker-matching would score every prompt identically.
    """
    per_block = 100

    def count(text: str) -> int:
        return per_block * text.count("body of block ")

    fitted = fit_context("q", hits, max_prompt_tokens=250, count_tokens=count)
    assert [h.rank for h in fitted.hits] == [1, 2]
    assert fitted.dropped == 1
    assert fitted.tokens == 200
    assert fitted.over_budget is False


def test_one_oversized_block_is_returned_rather_than_dropped(hits):
    """#11's rule: trimming beats failing, but so does trying beats abstaining."""
    fitted = fit_context("q", hits, max_prompt_tokens=10, count_tokens=lambda _: 9_999)
    assert len(fitted.hits) == 1
    assert fitted.over_budget is True
    assert fitted.tokens == 9_999


def test_the_returned_prompt_is_the_one_built_from_the_kept_hits(hits):
    """The caller must generate from `fitted.prompt`, not rebuild it and drift."""
    fitted = fit_context("q", hits, max_prompt_tokens=250, count_tokens=lambda _: 100)
    assert fitted.prompt == build_prompt("q", list(fitted.hits))


def test_a_non_positive_budget_is_refused(hits):
    with pytest.raises(ValueError, match="max_prompt_tokens must be > 0"):
        fit_context("q", hits, max_prompt_tokens=0)


def test_fitting_zero_hits_is_refused(hits):
    """Empty context is an abstain decision, not a trimming one (§0.6)."""
    with pytest.raises(ValueError, match="zero hits"):
        fit_context("q", [], max_prompt_tokens=6000)


# --- Tier 2: interface only (§9, §11) ------------------------------------------------


def test_tier2_is_disabled_and_unbuilt():
    assert DisabledOnlineSource.enabled is False
    with pytest.raises(NotImplementedError, match="interface-only"):
        DisabledOnlineSource().fetch("query", 5)


def test_tier2_stub_satisfies_the_online_source_interface():
    assert isinstance(DisabledOnlineSource(), OnlineSource)


def test_tier2_module_imports_nothing_that_can_reach_the_network():
    """§0.3: nothing hits the network at runtime. The Tier 2 stub must not even import a client."""
    source = Path(__file__).resolve().parent.parent / "answer" / "sources" / "online.py"
    body = source.read_text()
    for forbidden in ("requests", "urllib", "httpx", "aiohttp", "socket", "http.client"):
        assert f"import {forbidden}" not in body


# --- language reminder (BLOCKERS #21) -------------------------------------------------


def test_no_reminder_by_default(hits):
    """Off unless asked: it re-baselines every eval number, so it is opt-in per model."""
    assert "(Answer in" not in build_prompt("what is a matrix?", hits)


def test_reminder_sits_between_the_context_and_the_answer_cue(hits):
    """Position is the point — rule 1 says this thousands of tokens earlier and is lost."""
    p = build_prompt("what is a matrix?", hits, language_reminder=True)
    assert p.endswith("(Answer in English.)\nAnswer:")
    assert p.index("Context:") < p.index("(Answer in English.)")


def test_the_reminder_names_the_question_language_not_the_context_language(hits):
    """The context blocks here are Latin-script; the reminder must follow the question."""
    tamil = build_prompt("அணி என்றால் என்ன?", hits, language_reminder=True)
    chinese = build_prompt("矩阵是什么？", hits, language_reminder=True)
    assert "(Answer in Tamil.)" in tamil
    assert "(Answer in Chinese.)" in chinese


def test_an_unnameable_language_gets_no_reminder(hits):
    """Better silent than telling a model to answer in \"und\"."""
    assert language_reminder_for("12345 -- ...") == ""
    assert "(Answer in" not in build_prompt("12345 -- ...", hits, language_reminder=True)


def test_the_reminder_survives_context_trimming(hits):
    """`fit_context` rebuilds the prompt per candidate slice; it must not drop it."""
    fitted = fit_context(
        "what is a matrix?", hits, max_prompt_tokens=100000, language_reminder=True
    )
    assert "(Answer in English.)" in fitted.prompt
