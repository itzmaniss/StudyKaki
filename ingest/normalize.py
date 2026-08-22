"""Normalise blocks — ARCHITECTURE.md §3 `normalize`.

Unicode NFC, per-block script detection, join hyphen-broken lines, collapse whitespace.

Script detection is not cosmetic: it decides how `ingest/chunk.py` counts tokens (characters
for CJK/Thai, whitespace tokens elsewhere) and how lines are rejoined (no space between CJK
lines). Getting it wrong silently halves or doubles every chunk in a document.

Deliberately *not* stripped: ZWJ/ZWNJ (U+200C/U+200D) are semantically load-bearing in
Devanagari and Tamil, and ZWSP (U+200B) carries word boundaries in Thai. Only the BOM and
the soft hyphen — which is by definition a line-breaking artefact — are removed.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import replace

import structlog

from core.cache import CacheKey, StageCache, hash_rows, stage_timer
from core.schema import Block

log = structlog.get_logger(__name__)

STAGE_VERSION = "normalize/1"

_HYPHENS = frozenset("-\u2010\u2011")
_SOFT_HYPHEN = "\u00ad"
_BOM = "\ufeff"
_STRIP_CHARS = str.maketrans("", "", _SOFT_HYPHEN + _BOM)

# CJK and Thai are not space-delimited, so rejoining their lines with a space injects
# separators that were never in the source.
NO_SPACE_SCRIPTS = frozenset({"hans", "hant", "jpan", "thai"})

# Only these scripts hyphenate across a line break in practice.
HYPHENATING_SCRIPTS = frozenset({"latn", "cyrl"})

_SCRIPT_LANG: dict[str, str] = {
    "latn": "en",
    "cyrl": "ru",
    "arab": "ar",
    "deva": "hi",
    "taml": "ta",
    "thai": "th",
    "hans": "zh",
    "hant": "zh",
    "jpan": "ja",
    "kore": "ko",
    "unknown": "und",
}

# (start, end_inclusive, bucket). `han`/`kana`/`hang` are resolved into jpan/kore/hans/hant
# by `_resolve`, because a codepoint alone cannot tell those four apart.
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "latn"),
    (0x0061, 0x007A, "latn"),
    (0x00C0, 0x024F, "latn"),
    (0x0400, 0x052F, "cyrl"),
    (0x0600, 0x06FF, "arab"),
    (0x0750, 0x077F, "arab"),
    (0x08A0, 0x08FF, "arab"),
    (0x0900, 0x097F, "deva"),
    (0x0B80, 0x0BFF, "taml"),
    (0x0E00, 0x0E7F, "thai"),
    (0x1100, 0x11FF, "hang"),
    (0x1E00, 0x1EFF, "latn"),
    (0x3040, 0x30FF, "kana"),
    (0x3130, 0x318F, "hang"),
    (0x31F0, 0x31FF, "kana"),
    (0x3400, 0x4DBF, "han"),
    (0x4E00, 0x9FFF, "han"),
    (0xA960, 0xA97F, "hang"),
    (0xAC00, 0xD7FF, "hang"),
    (0xF900, 0xFAFF, "han"),
    (0xFB50, 0xFDFF, "arab"),
    (0xFE70, 0xFEFF, "arab"),
    (0xFF66, 0xFF9D, "kana"),
    (0x20000, 0x2A6DF, "han"),
)
_RANGE_STARTS: tuple[int, ...] = tuple(lo for lo, _, _ in _RANGES)

# Characters that exist in only one of the two Chinese orthographies. A handful of hits is
# enough to decide; with no hits at all, simplified is the safer default by corpus volume.
_SIMPLIFIED_ONLY = frozenset("国见说会学时这个们为对关后万与书电车东门问长风飞马鸟龙华汉")
_TRADITIONAL_ONLY = frozenset("國見說會學時這個們為對關後萬與書電車東門問長風飛馬鳥龍華漢")


def _bucket(ch: str) -> str | None:
    cp = ord(ch)
    idx = bisect_right(_RANGE_STARTS, cp) - 1
    if idx < 0:
        return None
    lo, hi, bucket = _RANGES[idx]
    return bucket if lo <= cp <= hi else None


def detect_script(text: str) -> str:
    """Dominant ISO 15924-style code for `text`, or `unknown` when it carries no letters."""
    counts: dict[str, int] = {}
    for ch in text:
        bucket = _bucket(ch)
        if bucket is not None:
            counts[bucket] = counts.get(bucket, 0) + 1
    if not counts:
        return "unknown"
    return _resolve(counts, text)


def _resolve(counts: dict[str, int], text: str) -> str:
    han = counts.pop("han", 0)
    kana = counts.pop("kana", 0)
    hang = counts.pop("hang", 0)
    if kana:
        counts["jpan"] = kana + han
    elif hang:
        counts["kore"] = hang + han
    elif han:
        counts[_chinese_variant(text)] = han
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _chinese_variant(text: str) -> str:
    simplified = sum(1 for ch in text if ch in _SIMPLIFIED_ONLY)
    traditional = sum(1 for ch in text if ch in _TRADITIONAL_ONLY)
    return "hant" if traditional > simplified else "hans"


def lang_for_script(script: str) -> str:
    """Coarse language tag derived from script. Real language ID is not in scope (§11)."""
    return _SCRIPT_LANG.get(script, "und")


def _join_lines(text: str, script: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    sep = "" if script in NO_SPACE_SCRIPTS else " "
    hyphenating = script in HYPHENATING_SCRIPTS
    buf = lines[0]
    for line in lines[1:]:
        broken = (
            hyphenating
            and len(buf) >= 2
            and buf[-1] in _HYPHENS
            and buf[-2].isalpha()
            and line[:1].isalpha()
        )
        if not broken:
            buf = buf + sep + line
            continue
        # "photo-\nsynthesis" -> "photosynthesis", but "Anglo-\nSaxon" keeps its hyphen.
        buf = (buf[:-1] if line[0].islower() else buf) + line
    return buf


def normalize_text(text: str, script: str) -> str:
    cleaned = unicodedata.normalize("NFC", text).translate(_STRIP_CHARS)
    joined = _join_lines(cleaned, script)
    return re.sub(r"\s+", " ", joined).strip()


def normalize_block(block: Block) -> Block:
    script = detect_script(unicodedata.normalize("NFC", block.text))
    return replace(block, text=normalize_text(block.text, script), script=script)


def normalize_blocks(
    blocks: Sequence[Block],
    *,
    cache: StageCache | None = None,
    config_hash: str = "none",
) -> list[Block]:
    """Blocks that normalise to empty text are dropped — they carry no retrievable content."""
    input_hash = hash_rows(blocks)
    with stage_timer("normalize", input_hash) as span:

        def compute() -> list[Block]:
            out = [normalize_block(b) for b in blocks]
            return [b for b in out if b.text]

        if cache is None:
            result = compute()
            span.n_out = len(result)
        else:
            key = CacheKey(
                stage="normalize",
                input_hash=input_hash,
                stage_version=STAGE_VERSION,
                config_hash=config_hash,
            )
            result = cache.get_or_compute(key, Block, compute, span=span)
        span.extra["dropped"] = len(blocks) - len(result)
    return result
