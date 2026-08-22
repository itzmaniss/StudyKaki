from __future__ import annotations

import unicodedata

import pytest

from core.cache import CacheKey, StageCache, hash_rows
from core.schema import SCRIPTS, Block
from ingest.normalize import (
    STAGE_VERSION,
    detect_script,
    lang_for_script,
    normalize_block,
    normalize_blocks,
    normalize_text,
)

ZWNJ = "\u200c"
ZWJ = "\u200d"
ZWSP = "\u200b"
SOFT_HYPHEN = "\u00ad"
BOM = "\ufeff"


def make_block(text: str, **kw) -> Block:
    base = dict(
        block_id="b0",
        doc_id="doc-a",
        page=1,
        bbox=(0.1, 0.2, 0.8, 0.9),
        kind="paragraph",
        reading_order=0,
        script="unknown",
        text=text,
        ocr_confidence=None,
    )
    return Block(**{**base, **kw})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Photosynthesis in chloroplasts", "latn"),
        ("ஒளிச்சேர்க்கை என்பது", "taml"),
        ("การสังเคราะห์ด้วยแสง", "thai"),
        ("प्रकाश संश्लेषण", "deva"),
        ("光合作用是植物的过程", "hans"),
        ("這個過程與學說有關", "hant"),
        ("こうごうせいは しょくぶつ", "jpan"),
        ("광합성은 식물의 과정입니다", "kore"),
        ("фотосинтез растений", "cyrl"),
        ("التمثيل الضوئي", "arab"),
        ("123 456 — (7.8) %", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected
    assert detect_script(text) in SCRIPTS


def test_kanji_with_kana_reads_as_japanese_not_chinese():
    """Kana is the only reliable signal; ideographs alone cannot separate the two."""
    assert detect_script("光合成はクロロフィルによる") == "jpan"


def test_the_majority_script_wins_a_mixed_block():
    assert detect_script("ATP ஒளிச்சேர்க்கை ஒளிச்சேர்க்கை ஒளிச்சேர்க்கை") == "taml"
    assert detect_script("photosynthesis in chloroplasts 光合") == "latn"


@pytest.mark.parametrize(
    ("script", "lang"),
    [("taml", "ta"), ("hans", "zh"), ("hant", "zh"), ("jpan", "ja"), ("unknown", "und")],
)
def test_lang_for_script(script, lang):
    assert lang_for_script(script) == lang


def test_every_valid_script_has_a_language_tag():
    assert all(lang_for_script(s) != "und" for s in SCRIPTS - {"unknown"})


def test_text_is_composed_to_nfc():
    decomposed = "cafe\u0301 du chloroplaste"
    assert not unicodedata.is_normalized("NFC", decomposed)
    result = normalize_text(decomposed, "latn")
    assert unicodedata.is_normalized("NFC", result)
    assert result.startswith("caf\u00e9")


def test_hyphen_broken_word_is_rejoined():
    assert normalize_text("photo-\nsynthesis happens", "latn") == "photosynthesis happens"


def test_a_hyphenated_proper_noun_keeps_its_hyphen():
    """Anglo-/Saxon is a compound split by the layout, not a word split by hyphenation."""
    assert normalize_text("Anglo-\nSaxon rule", "latn") == "Anglo-Saxon rule"


def test_ordinary_line_breaks_become_spaces():
    assert normalize_text("light energy\ndrives the reaction", "latn") == (
        "light energy drives the reaction"
    )


@pytest.mark.parametrize(
    ("text", "script", "expected"),
    [
        ("光合作用是植物\n利用光能的过程", "hans", "光合作用是植物利用光能的过程"),
        ("การสังเคราะห์\nด้วยแสง", "thai", "การสังเคราะห์ด้วยแสง"),
    ],
)
def test_cjk_and_thai_lines_join_without_a_space(text, script, expected):
    assert normalize_text(text, script) == expected


def test_whitespace_runs_collapse():
    assert normalize_text("  ATP   and \t\t NADPH \n\n  ", "latn") == "ATP and NADPH"


def test_soft_hyphen_and_bom_are_removed():
    assert normalize_text(f"{BOM}co{SOFT_HYPHEN}operate", "latn") == "cooperate"


@pytest.mark.parametrize("joiner", [ZWJ, ZWNJ])
def test_semantic_joiners_survive(joiner):
    """ZWJ/ZWNJ change what a Devanagari or Tamil cluster renders as — stripping corrupts text."""
    assert joiner in normalize_text(f"क{joiner}ष", "deva")


def test_thai_word_boundaries_survive():
    """ZWSP is Thai's word separator; a later lexical retriever needs it."""
    assert ZWSP in normalize_text(f"การ{ZWSP}แสง", "thai")


def test_normalize_block_fills_in_the_script_and_keeps_provenance():
    block = make_block("photo-\nsynthesis", page=4, reading_order=17, kind="caption")
    result = normalize_block(block)
    assert result.script == "latn"
    assert result.text == "photosynthesis"
    assert (result.block_id, result.doc_id, result.page, result.bbox, result.kind) == (
        block.block_id,
        block.doc_id,
        block.page,
        block.bbox,
        block.kind,
    )
    assert result.reading_order == block.reading_order
    assert result.ocr_confidence is block.ocr_confidence


def test_ocr_confidence_is_carried_through():
    assert normalize_block(make_block("ஒளிச்சேர்க்கை", ocr_confidence=0.42)).ocr_confidence == 0.42


def test_blocks_that_normalise_to_nothing_are_dropped():
    blocks = [
        make_block("real content here", block_id="b0"),
        make_block("   \n \t  ", block_id="b1", reading_order=1),
        make_block(f"{BOM}{SOFT_HYPHEN}", block_id="b2", reading_order=2),
    ]
    assert [b.block_id for b in normalize_blocks(blocks)] == ["b0"]


def test_normalize_blocks_preserves_order():
    blocks = [make_block(f"paragraph {i}", block_id=f"b{i}", reading_order=i) for i in range(5)]
    assert [b.block_id for b in normalize_blocks(blocks)] == [f"b{i}" for i in range(5)]


@pytest.mark.parametrize("text", ["", "   ", "\n\n", SOFT_HYPHEN, BOM, "\u00a0"])
def test_malformed_text_normalises_to_empty_without_raising(text):
    assert normalize_text(text, detect_script(text)) == ""


def test_normalize_blocks_on_an_empty_input():
    assert normalize_blocks([]) == []


def test_second_call_is_served_from_cache(tmp_path):
    cache = StageCache(tmp_path / "cache")
    blocks = [make_block("photo-\nsynthesis", block_id="b0")]
    first = normalize_blocks(blocks, cache=cache)
    key = CacheKey(
        stage="normalize",
        input_hash=hash_rows(blocks),
        stage_version=STAGE_VERSION,
        config_hash="none",
    )
    assert cache.has(key)
    assert normalize_blocks(blocks, cache=cache) == first


def test_cached_result_is_identical_to_the_uncached_one(tmp_path):
    blocks = [
        make_block("光合作用是植物\n利用光能的过程", block_id="b0"),
        make_block("photo-\nsynthesis", block_id="b1", reading_order=1),
    ]
    assert normalize_blocks(blocks, cache=StageCache(tmp_path / "cache")) == normalize_blocks(
        blocks
    )
