from __future__ import annotations

import pytest

from core.cache import CacheKey, StageCache, hash_rows
from core.config import ChunkConfig, load_config
from core.schema import Block
from ingest.chunk import (
    STAGE_VERSION,
    MixedDocumentError,
    chunk_blocks,
    count_tokens,
    heading_level,
)

DEFAULT = ChunkConfig(target_tokens=400, overlap=60, min_tokens=80)
TINY = ChunkConfig(target_tokens=20, overlap=6, min_tokens=1)


def block(
    order: int,
    text: str,
    *,
    kind: str = "paragraph",
    page: int = 1,
    script: str = "latn",
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.2),
    doc_id: str = "doc-a",
) -> Block:
    return Block(
        block_id=f"b{order}",
        doc_id=doc_id,
        page=page,
        bbox=bbox,
        kind=kind,
        reading_order=order,
        script=script,
        text=text,
        ocr_confidence=None,
    )


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


@pytest.mark.parametrize(
    ("text", "script", "expected"),
    [
        ("one two three", "latn", 3),
        ("ஒளி சேர்க்கை", "taml", 2),
        ("光合作用是植物", "hans", 7),
        ("光合 作用", "hant", 4),
        ("こうごうせい", "jpan", 6),
        ("การสังเคราะห์", "thai", 13),
        ("", "latn", 0),
        ("", "hans", 0),
    ],
)
def test_cjk_and_thai_count_characters_everything_else_counts_words(text, script, expected):
    assert count_tokens(text, script) == expected


def test_a_chinese_paragraph_is_not_mistaken_for_two_tokens():
    """`str.split()` on unspaced Chinese returns 1 — a 400-'token' chunk would be a chapter."""
    text = "光合作用是植物利用光能把二氧化碳和水转化为葡萄糖的过程"
    assert count_tokens(text, "latn") == 1
    assert count_tokens(text, "hans") == len(text)


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Chapter 3", 1),
        ("Introduction", 1),
        ("3.2 Photosynthesis", 2),
        ("3.2.1 The Calvin Cycle", 3),
        ("4. Results", 1),
        ("12.4.7.1 Deeply nested", 4),
    ],
)
def test_heading_level_follows_the_numbering(text, level):
    assert heading_level(text) == level


def test_no_blocks_no_chunks():
    assert chunk_blocks([], DEFAULT) == []


def test_blocks_from_two_documents_are_rejected():
    """A bbox_union spanning two documents would be a citation pointing nowhere."""
    blocks = [block(0, "one"), block(1, "two", doc_id="doc-b")]
    with pytest.raises(MixedDocumentError, match="one document at a time"):
        chunk_blocks(blocks, DEFAULT)


def test_a_block_is_never_split_even_when_it_exceeds_the_target():
    huge = block(0, words(1000))
    chunks = chunk_blocks([huge], DEFAULT)
    assert len(chunks) == 1
    assert chunks[0].block_ids == ["b0"]
    assert chunks[0].text == huge.text
    assert chunks[0].token_count == 1000


def test_every_block_reaches_at_least_one_chunk():
    blocks = [block(i, words(30, f"s{i}")) for i in range(20)]
    chunks = chunk_blocks(blocks, DEFAULT)
    covered = {bid for c in chunks for bid in c.block_ids}
    assert covered == {b.block_id for b in blocks}


def test_chunks_fill_towards_the_target_without_overshooting_on_the_first_block():
    blocks = [block(i, words(50, f"s{i}")) for i in range(20)]
    chunks = chunk_blocks(blocks, DEFAULT)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.token_count <= DEFAULT.target_tokens + 50


def test_consecutive_chunks_overlap_within_the_budget():
    blocks = [block(i, words(30, f"s{i}")) for i in range(30)]
    chunks = chunk_blocks(blocks, DEFAULT)
    assert len(chunks) >= 2
    shared = set(chunks[0].block_ids) & set(chunks[1].block_ids)
    assert shared, "consecutive chunks in one section must overlap"
    assert len(shared) * 30 <= DEFAULT.overlap


def test_a_chunk_is_never_a_subset_of_its_neighbour():
    blocks = [block(i, words(30, f"s{i}")) for i in range(30)]
    chunks = chunk_blocks(blocks, DEFAULT)
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert not set(earlier.block_ids) <= set(later.block_ids)
        assert not set(later.block_ids) <= set(earlier.block_ids)


def test_chunks_never_straddle_a_heading():
    blocks = [
        block(0, "3.1 Light Reactions", kind="heading"),
        block(1, words(200, "a")),
        block(2, "3.2 Calvin Cycle", kind="heading"),
        block(3, words(200, "b")),
    ]
    chunks = chunk_blocks(blocks, DEFAULT)
    assert len(chunks) == 2
    assert set(chunks[0].block_ids) == {"b0", "b1"}
    assert set(chunks[1].block_ids) == {"b2", "b3"}


def test_overlap_does_not_leak_across_a_heading():
    blocks = [
        block(0, "3.1 Light Reactions", kind="heading"),
        *[block(i, words(30, f"a{i}")) for i in range(1, 30)],
        block(30, "3.2 Calvin Cycle", kind="heading"),
        *[block(i, words(30, f"b{i}")) for i in range(31, 40)],
    ]
    chunks = chunk_blocks(blocks, DEFAULT)
    first_section = {c.chunk_id for c in chunks if c.heading_path[-1] == "3.1 Light Reactions"}
    for chunk in chunks:
        if chunk.chunk_id in first_section:
            continue
        assert all(int(bid[1:]) >= 30 for bid in chunk.block_ids)


def test_heading_path_nests_by_numbering():
    blocks = [
        block(0, "Chapter 3", kind="heading"),
        block(1, "3.1 Light Reactions", kind="heading"),
        block(2, words(100, "a")),
        block(3, "3.1.1 Photosystem II", kind="heading"),
        block(4, words(100, "b")),
        block(5, "3.2 Calvin Cycle", kind="heading"),
        block(6, words(100, "c")),
    ]
    paths = [c.heading_path for c in chunk_blocks(blocks, DEFAULT)]
    assert paths == [
        ["Chapter 3", "3.1 Light Reactions"],
        ["Chapter 3", "3.1 Light Reactions", "3.1.1 Photosystem II"],
        ["Chapter 3", "3.2 Calvin Cycle"],
    ]


def test_a_bare_parent_heading_survives_in_the_path_not_as_a_chunk():
    """ "Chapter 3" alone would index as a two-token chunk that can only be a false positive."""
    blocks = [
        block(0, "Chapter 3", kind="heading"),
        block(1, "3.1 Light Reactions", kind="heading"),
        block(2, words(100, "a")),
    ]
    chunks = chunk_blocks(blocks, DEFAULT)
    assert len(chunks) == 1
    assert "b0" not in chunks[0].block_ids
    assert chunks[0].heading_path[0] == "Chapter 3"


def test_a_trailing_bare_heading_is_kept():
    """Nothing inherits it, so dropping it would lose the text outright."""
    blocks = [
        block(0, "3.1 Light Reactions", kind="heading"),
        block(1, words(100, "a")),
        block(2, "Appendix", kind="heading"),
    ]
    chunks = chunk_blocks(blocks, DEFAULT)
    assert chunks[-1].block_ids == ["b2"]
    assert chunks[-1].text == "Appendix"


def test_blocks_before_the_first_heading_get_an_empty_path():
    chunks = chunk_blocks([block(0, words(50, "a"))], DEFAULT)
    assert chunks[0].heading_path == []


def test_bbox_union_contains_every_member_block():
    blocks = [
        block(0, words(30, "a"), bbox=(0.10, 0.20, 0.40, 0.30)),
        block(1, words(30, "b"), bbox=(0.55, 0.05, 0.95, 0.15)),
        block(2, words(30, "c"), bbox=(0.20, 0.60, 0.50, 0.85)),
    ]
    by_id = {b.block_id: b for b in blocks}
    for chunk in chunk_blocks(blocks, DEFAULT):
        ux0, uy0, ux1, uy1 = chunk.bbox_union
        for bid in chunk.block_ids:
            x0, y0, x1, y1 = by_id[bid].bbox
            assert ux0 <= x0 and uy0 <= y0
            assert ux1 >= x1 and uy1 >= y1
        assert (ux0, uy0, ux1, uy1) == (0.10, 0.05, 0.95, 0.85)


def test_bbox_union_is_tight_not_the_whole_page():
    """A union padded to (0,0,1,1) would highlight the entire page on every citation."""
    blocks = [block(i, words(30, f"a{i}"), bbox=(0.2, 0.3, 0.6, 0.4)) for i in range(3)]
    assert chunk_blocks(blocks, DEFAULT)[0].bbox_union == (0.2, 0.3, 0.6, 0.4)


def test_pages_span_the_member_blocks():
    blocks = [block(i, words(80, f"a{i}"), page=1 + i // 2) for i in range(6)]
    for chunk in chunk_blocks(blocks, DEFAULT):
        member_pages = [int(bid[1:]) // 2 + 1 for bid in chunk.block_ids]
        assert chunk.page_start == min(member_pages)
        assert chunk.page_end == max(member_pages)
        assert chunk.page_start <= chunk.page_end


def test_undersized_tail_is_folded_into_its_predecessor():
    blocks = [block(0, words(390, "a")), block(1, words(20, "b"))]
    chunks = chunk_blocks(blocks, ChunkConfig(target_tokens=400, overlap=0, min_tokens=80))
    assert len(chunks) == 1
    assert chunks[0].block_ids == ["b0", "b1"]
    assert chunks[0].token_count == 410


def test_merging_never_duplicates_an_overlapped_block():
    blocks = [block(i, words(30, f"a{i}")) for i in range(15)]
    for chunk in chunk_blocks(blocks, DEFAULT):
        assert len(chunk.block_ids) == len(set(chunk.block_ids))


def test_script_and_lang_follow_the_dominant_member():
    blocks = [
        block(0, "ATP and NADPH", script="latn"),
        block(1, "光合作用是植物利用光能把二氧化碳和水转化为葡萄糖的过程", script="hans"),
    ]
    chunk = chunk_blocks(blocks, DEFAULT)[0]
    assert chunk.script == "hans"
    assert chunk.lang == "zh"


def test_a_chinese_section_splits_on_character_count():
    text = "光" * 150
    blocks = [block(i, text, script="hans") for i in range(6)]
    chunks = chunk_blocks(blocks, DEFAULT)
    assert len(chunks) >= 3
    assert all(c.script == "hans" and c.lang == "zh" for c in chunks)
    assert all(c.token_count <= DEFAULT.target_tokens + 150 for c in chunks)


def test_token_count_matches_what_the_chunk_actually_stores():
    blocks = [block(i, words(50, f"a{i}")) for i in range(6)]
    for chunk in chunk_blocks(blocks, DEFAULT):
        assert chunk.token_count == sum(len(part.split()) for part in chunk.text.split("\n\n"))
        assert len(chunk.text.split("\n\n")) == len(chunk.block_ids)


def test_chunk_ids_are_deterministic_and_unique():
    blocks = [block(i, words(60, f"a{i}")) for i in range(12)]
    first = chunk_blocks(blocks, DEFAULT)
    second = chunk_blocks(blocks, DEFAULT)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_reading_order_drives_grouping_not_input_order():
    shuffled = [block(2, words(40, "c")), block(0, words(40, "a")), block(1, words(40, "b"))]
    chunk = chunk_blocks(shuffled, DEFAULT)[0]
    assert chunk.block_ids == ["b0", "b1", "b2"]


def test_tuning_the_target_changes_the_chunking():
    blocks = [block(i, words(30, f"a{i}")) for i in range(10)]
    assert len(chunk_blocks(blocks, TINY)) > len(chunk_blocks(blocks, DEFAULT))


def test_second_call_is_served_from_cache(tmp_path):
    cache = StageCache(tmp_path / "cache")
    blocks = [block(i, words(60, f"a{i}")) for i in range(8)]
    first = chunk_blocks(blocks, DEFAULT, cache=cache)
    key = CacheKey(
        stage="chunk",
        input_hash=hash_rows(blocks),
        stage_version=STAGE_VERSION,
        config_hash=load_config().chunk_config_hash,
    )
    assert cache.has(key), "default config_hash must match Config.chunk_config_hash"
    assert chunk_blocks(blocks, DEFAULT, cache=cache) == first


def test_retuning_the_chunker_does_not_reuse_the_old_entry(tmp_path):
    """The whole point of the cache key: a new target_tokens is a different stage output."""
    cache = StageCache(tmp_path / "cache")
    blocks = [block(i, words(30, f"a{i}")) for i in range(10)]
    wide = chunk_blocks(blocks, DEFAULT, cache=cache)
    narrow = chunk_blocks(blocks, TINY, cache=cache)
    assert len(narrow) > len(wide)
    assert chunk_blocks(blocks, DEFAULT, cache=cache) == wide


def test_cached_result_is_identical_to_the_uncached_one(tmp_path):
    blocks = [
        block(0, "3.1 Light Reactions", kind="heading"),
        block(1, words(200, "a"), page=1, bbox=(0.1, 0.2, 0.5, 0.4)),
        block(2, "光" * 100, script="hans", page=2, bbox=(0.3, 0.1, 0.9, 0.7)),
    ]
    assert chunk_blocks(blocks, DEFAULT, cache=StageCache(tmp_path / "c")) == chunk_blocks(
        blocks, DEFAULT
    )
