"""Citation verification — ARCHITECTURE.md §4.

The model invents citation markers. Everything here is about making sure an invented `[n]`
never reaches the student and never lands in `Answer.citations`. Assertions are on structure
and provenance only; no assertion depends on model-generated wording.
"""

from __future__ import annotations

import pytest

from answer.cite import find_markers, has_citation_markers, strip_all_markers, verify
from core.schema import Chunk, Retrieved


def make_hit(rank: int, *, page: int = 1, doc_id: str = "d1", text: str = "body") -> Retrieved:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        doc_id=doc_id,
        page_start=page,
        page_end=page,
        block_ids=[f"b{rank}"],
        bbox_union=(0.0, 0.0, 1.0, 1.0),
        heading_path=["Chapter 3"],
        text=text,
        token_count=len(text.split()),
        lang="en",
        script="latn",
    )
    return Retrieved(chunk=chunk, score=1.0 / rank, rank=rank)


@pytest.fixture
def hits() -> list[Retrieved]:
    return [make_hit(i, page=10 * i) for i in range(1, 6)]


# --- valid markers survive untouched -------------------------------------------------


def test_valid_marker_is_kept_and_reported(hits):
    clean, used = verify("Photosynthesis needs light [2].", hits)
    assert clean == "Photosynthesis needs light [2]."
    assert [u.chunk.chunk_id for u in used] == ["c2"]


def test_clean_text_is_byte_identical_when_nothing_is_invented(hits):
    # Markdown hard line breaks are two trailing spaces — repair must not eat them.
    text = "First [1].  \nSecond [3][4].\n\n- bullet [5]\n"
    clean, used = verify(text, hits)
    assert clean == text
    assert [u.chunk.chunk_id for u in used] == ["c1", "c3", "c4", "c5"]


def test_only_referenced_hits_are_returned(hits):
    _, used = verify("Only this one [4].", hits)
    assert len(used) == 1
    assert used[0].chunk.page_start == 40


def test_citations_are_deduplicated_and_ordered_by_block_number(hits):
    clean, used = verify("Later [3]. Earlier [1]. Again [3].", hits)
    assert clean == "Later [3]. Earlier [1]. Again [3]."
    assert [u.chunk.chunk_id for u in used] == ["c1", "c3"]


def test_returned_citations_keep_full_provenance(hits):
    _, used = verify("See [5].", hits)
    chunk = used[0].chunk
    assert (chunk.doc_id, chunk.page_start, chunk.bbox_union) == ("d1", 50, (0.0, 0.0, 1.0, 1.0))


# --- invented markers are dropped ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # [0] — blocks are 1-indexed, so zero is always a fabrication.
        ("Light drives it [0].", "Light drives it."),
        # Out of range against 5 context blocks.
        ("Light drives it [99].", "Light drives it."),
        ("Light drives it [6].", "Light drives it."),
        # Absurdly large numbers must not overflow or survive.
        ("Light drives it [999999999999999999999].", "Light drives it."),
        # Marker at line start: no leading space left behind.
        ("[99] Light drives it.", "Light drives it."),
        # Marker between words: exactly one space survives.
        ("See [99] and read on.", "See and read on."),
        # Two adjacent fabrications.
        ("See [98][99] and read on.", "See and read on."),
        # Hugging a closing bracket / CJK punctuation.
        ("光合成が必要です [42]。", "光合成が必要です。"),
        # End of text with no trailing punctuation.
        ("Light drives it [99]", "Light drives it"),
    ],
)
def test_invented_markers_are_stripped_cleanly(text, expected, hits):
    clean, used = verify(text, hits)
    assert clean == expected
    assert used == []


def test_mixed_group_keeps_only_real_blocks(hits):
    clean, used = verify("Both matter [1, 99].", hits)
    assert clean == "Both matter [1]."
    assert [u.chunk.chunk_id for u in used] == ["c1"]


def test_group_of_only_invented_markers_is_removed_entirely(hits):
    clean, used = verify("Both matter [98, 99].", hits)
    assert clean == "Both matter."
    assert used == []


def test_comma_group_is_canonicalised_to_single_number_brackets(hits):
    """A renderer must be able to trust a bare `[n]` regex on clean_text."""
    clean, used = verify("Sources [1, 2,3].", hits)
    assert clean == "Sources [1][2][3]."
    assert [u.chunk.chunk_id for u in used] == ["c1", "c2", "c3"]


def test_leading_zero_marker_is_normalised(hits):
    clean, used = verify("Padded [01].", hits)
    assert clean == "Padded [1]."
    assert [u.chunk.chunk_id for u in used] == ["c1"]


def test_duplicate_number_inside_one_group_is_emitted_once(hits):
    clean, _ = verify("Repeated [2, 2].", hits)
    assert clean == "Repeated [2]."


def test_empty_context_strips_every_marker():
    clean, used = verify("Everything is invented [1][2] here [3].", [])
    assert clean == "Everything is invented here."
    assert used == []


# --- things that are not citation markers --------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Not a marker [abc].",
        "Not a marker [].",
        "Not a marker [-1].",
        "Not a marker [1.5].",
        "Footnote style [^1].",
        "Range style [1-3].",
        "A markdown [link](http://example.invalid) stays.",
    ],
)
def test_non_numeric_brackets_are_left_alone(text, hits):
    """Only `[digits]` is a citation marker. Mangling other brackets would corrupt the answer."""
    clean, used = verify(text, hits)
    assert clean == text
    assert used == []


# --- code is never rewritten ---------------------------------------------------------


def test_markers_inside_inline_code_are_untouched_and_uncited(hits):
    text = "Index it with `arr[1]` and `arr[99]`, then see [2]."
    clean, used = verify(text, hits)
    assert clean == text
    assert [u.chunk.chunk_id for u in used] == ["c2"]


def test_markers_inside_fenced_code_are_untouched_and_uncited(hits):
    text = "Example:\n\n```python\nx = arr[1]\ny = arr[99]\n```\n\nExplained in [3].\n"
    clean, used = verify(text, hits)
    assert clean == text
    assert [u.chunk.chunk_id for u in used] == ["c3"]


def test_unclosed_fence_protects_to_end_of_text(hits):
    text = "Broken:\n\n```\nx = arr[99]\n"
    clean, used = verify(text, hits)
    assert clean == text
    assert used == []


def test_unmatched_backtick_does_not_protect_a_fabrication(hits):
    """Fail open: if a code span never closes, the invented marker is still stripped."""
    clean, used = verify("a ` b [99].", hits)
    assert clean == "a ` b."
    assert used == []


def test_multi_backtick_span_needs_a_matching_run(hits):
    text = "Literal ``arr[99]`` stays."
    clean, used = verify(text, hits)
    assert clean == text
    assert used == []


def test_find_markers_ignores_code(hits):
    assert find_markers("`arr[99]` then [2] and [7].") == [2, 7]
    assert has_citation_markers("no markers here `arr[1]`") is False


# --- postcondition -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "[0][1][2][3][4][5][6][99]",
        "mixed [1, 99] and [98] and [abc] and `[7]`",
        "```\n[99]\n```\n[4] tail [0]",
        "[99]",
        "",
        "no markers at all",
        "[1]" * 50,
    ],
)
def test_every_surviving_marker_indexes_a_real_block(text, hits):
    clean, used = verify(text, hits)
    survivors = find_markers(clean)
    assert all(1 <= n <= len(hits) for n in survivors)
    assert {u.rank for u in used} == set(survivors)


def test_verify_is_idempotent(hits):
    once, _ = verify("Mixed [1, 99] and [0] and [3].", hits)
    twice, _ = verify(once, hits)
    assert once == twice


# --- helpers -------------------------------------------------------------------------


def test_strip_all_markers_removes_every_marker():
    assert strip_all_markers("General knowledge [1] with [2, 3].") == "General knowledge with."
    assert has_citation_markers(strip_all_markers("a [1] b [99] c")) is False


def test_sentinel_character_in_model_output_cannot_corrupt_repair(hits):
    clean, used = verify("Odd\x00 output [1].", hits)
    assert "\x00" not in clean
    assert [u.chunk.chunk_id for u in used] == ["c1"]


# --- malformed input -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["list"]])
def test_verify_rejects_non_string_answer_text(bad, hits):
    with pytest.raises(TypeError, match="answer_text"):
        verify(bad, hits)


@pytest.mark.parametrize("bad", [None, 42, object()])
def test_verify_rejects_non_sequence_context(bad):
    with pytest.raises(TypeError, match="context_hits"):
        verify("text", bad)


@pytest.mark.parametrize("bad", ["not a hit", None, 7])
def test_verify_rejects_context_entries_that_are_not_retrieved(bad, hits):
    with pytest.raises(TypeError, match=r"context_hits\[1\]"):
        verify("text", [hits[0], bad])


def test_find_markers_rejects_non_string():
    with pytest.raises(TypeError, match="text"):
        find_markers(None)
