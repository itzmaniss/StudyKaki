"""Structure-aware chunking — ARCHITECTURE.md §3 `chunk`.

The highest-leverage knob in the system, so the rules are explicit:

* Never split mid-block. A block over `target_tokens` becomes a chunk on its own.
* Group by reading order within a heading. Chunks never straddle one, and neither does
  overlap — the tail of "3.1 Light Reactions" in the first chunk of "3.2 Calvin Cycle"
  pollutes both.
* Count characters for CJK/Thai, whitespace tokens elsewhere. A 400-"token" Chinese chunk
  measured by `str.split()` would be a whole chapter.
* `bbox_union` is the true union and `page_start`/`page_end` span it — citations are built
  from these, and provenance cannot be retrofitted (§0 non-negotiable 2).

`Chunk.text` is exactly its member blocks concatenated, so `token_count` describes what is
stored and `block_ids` accounts for every character. `heading_path` travels alongside for
`ingest/embed.py` to prefix; prepending it here would put text in the chunk that no block
can vouch for.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import structlog

from core.cache import CacheKey, StageCache, hash_rows, stage_timer
from core.config import ChunkConfig, _hash_obj
from core.schema import BBox, Block, Chunk
from ingest.normalize import lang_for_script

log = structlog.get_logger(__name__)

STAGE_VERSION = "chunk/2"

BLOCK_SEPARATOR = "\n\n"

# Not whitespace-delimited; one non-space char ≈ one subword token in BGE-M3's vocab.
CHAR_COUNTED_SCRIPTS = frozenset({"hans", "hant", "jpan", "thai"})

# "3.2 Photosynthesis" -> depth 2; "Chapter 3" has no numeral, so depth 1. That is §2's
# ["Chapter 3", "3.2 Photosynthesis"] example.
_NUM_PREFIX_RE = re.compile(r"^\s*(\d+(?:[.\-]\d+)*)\s*[.):]?\s+\S")


class MixedDocumentError(ValueError):
    """Chunking is per-document; `bbox_union` across two documents would be meaningless."""


def count_tokens(text: str, script: str) -> int:
    if script in CHAR_COUNTED_SCRIPTS:
        return sum(1 for ch in text if not ch.isspace())
    return len(text.split())


def heading_level(text: str) -> int:
    match = _NUM_PREFIX_RE.match(text)
    if match is None:
        return 1
    return len(re.split(r"[.\-]", match.group(1)))


@dataclass(frozen=True)
class _Section:
    heading_path: tuple[str, ...]
    blocks: tuple[Block, ...]


def _sections(blocks: Sequence[Block]) -> list[_Section]:
    stack: list[str] = []
    out: list[_Section] = []
    current: list[Block] = []
    path: tuple[str, ...] = ()

    def flush() -> None:
        if current:
            out.append(_Section(heading_path=path, blocks=tuple(current)))

    for block in blocks:
        if block.kind == "heading":
            flush()
            level = heading_level(block.text)
            del stack[level - 1 :]
            stack.append(block.text)
            path = tuple(stack)
            current = [block]
        else:
            current.append(block)
    flush()
    return out


def _is_redundant(section: _Section, following: _Section | None) -> bool:
    """A bare heading whose child section inherits it is not worth its own chunk.

    "Chapter 3" before "3.1 Light Reactions" would index as a two-token chunk that can only
    be a false positive. The text survives in the `heading_path` beneath it. A trailing bare
    heading is kept — nothing inherits it.
    """
    if following is None or any(b.kind != "heading" for b in section.blocks):
        return False
    depth = len(section.heading_path)
    return (
        len(following.heading_path) > depth
        and following.heading_path[:depth] == section.heading_path
    )


def _overlap_tail(packed: Sequence[Block], budget: int) -> list[Block]:
    """Trailing blocks worth at most `budget` tokens, never the whole chunk.

    Everything would make the next chunk a superset of this one: same text embedded twice,
    two identical hits at query time.
    """
    if budget <= 0 or len(packed) < 2:
        return []
    tail: list[Block] = []
    used = 0
    for block in reversed(packed[1:]):
        cost = count_tokens(block.text, block.script)
        if used + cost > budget:
            break
        tail.insert(0, block)
        used += cost
    return tail


def _pack(blocks: Sequence[Block], params: ChunkConfig) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    used = 0
    for block in blocks:
        cost = count_tokens(block.text, block.script)
        if current and used + cost > params.target_tokens:
            groups.append(current)
            current = _overlap_tail(current, params.overlap)
            used = sum(count_tokens(b.text, b.script) for b in current)
        current.append(block)
        used += cost
    if current:
        groups.append(current)
    return groups


def _merge_short(groups: list[list[Block]], min_tokens: int) -> list[list[Block]]:
    """Fold an undersized chunk into its predecessor rather than ship or drop it.

    Dropping loses content, shipping a 9-token chunk gives the retriever a contextless
    fragment. Merging keeps every block reachable for one slightly oversized chunk.
    """
    merged: list[list[Block]] = []
    for group in groups:
        if merged and _tokens(group) < min_tokens:
            seen = {b.block_id for b in merged[-1]}
            merged[-1].extend(b for b in group if b.block_id not in seen)
        else:
            merged.append(list(group))
    return merged


def _tokens(blocks: Iterable[Block]) -> int:
    return sum(count_tokens(b.text, b.script) for b in blocks)


def _dedup(blocks: Iterable[Block]) -> list[Block]:
    seen: set[str] = set()
    out: list[Block] = []
    for b in blocks:
        if b.block_id not in seen:
            seen.add(b.block_id)
            out.append(b)
    return out


def _fold_orphans(
    pairs: list[tuple[list[Block], tuple[str, ...]]], min_tokens: int
) -> list[tuple[list[Block], tuple[str, ...]]]:
    """Second pass across section boundaries, so `min_tokens` holds document-wide.

    `_merge_short` folds only within a section, so a section that is one short group has no
    predecessor and ships at whatever size. That is the common case: textbooks split the
    chapter number and title into two heading blocks, which `_sections` reads as siblings,
    so "CHAPTER 6" indexes as a two-token chunk — twelve per volume, all matching any query
    containing the word chapter.

    Folds forward: an orphan labels what comes after it, so it joins the following chunk and
    inherits its heading path. A trailing orphan labels nothing and is left to `_is_redundant`.
    """
    out: list[tuple[list[Block], tuple[str, ...]]] = []
    pending: list[Block] = []
    pending_path: tuple[str, ...] = ()
    for group, path in pairs:
        if pending:
            group = _dedup([*pending, *group])
        else:
            pending_path = path
        pending = []
        if _tokens(group) < min_tokens:
            pending = list(group)
            continue
        out.append((list(group), path))
    if pending:
        out.append((pending, pending_path))
    return out


def _bbox_union(blocks: Sequence[Block]) -> BBox:
    return (
        min(b.bbox[0] for b in blocks),
        min(b.bbox[1] for b in blocks),
        max(b.bbox[2] for b in blocks),
        max(b.bbox[3] for b in blocks),
    )


def _dominant_script(blocks: Sequence[Block]) -> str:
    weights: dict[str, int] = {}
    for block in blocks:
        if block.script == "unknown":
            continue
        weights[block.script] = weights.get(block.script, 0) + count_tokens(
            block.text, block.script
        )
    if not weights:
        return "unknown"
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _build(blocks: Sequence[Block], heading_path: Sequence[str]) -> Chunk:
    doc_id = blocks[0].doc_id
    block_ids = [b.block_id for b in blocks]
    script = _dominant_script(blocks)
    digest = hashlib.sha256(f"{doc_id}|{','.join(block_ids)}".encode()).hexdigest()
    return Chunk(
        chunk_id=digest[:32],
        doc_id=doc_id,
        page_start=min(b.page for b in blocks),
        page_end=max(b.page for b in blocks),
        block_ids=block_ids,
        bbox_union=_bbox_union(blocks),
        heading_path=list(heading_path),
        text=BLOCK_SEPARATOR.join(b.text for b in blocks),
        token_count=sum(count_tokens(b.text, b.script) for b in blocks),
        lang=lang_for_script(script),
        script=script,
    )


def chunk_blocks(
    blocks: Sequence[Block],
    params: ChunkConfig,
    *,
    cache: StageCache | None = None,
    config_hash: str | None = None,
) -> list[Chunk]:
    """Blocks (already normalised) -> chunks, in reading order, one document at a time.

    `config_hash` defaults to `params` alone, matching `Config.chunk_config_hash`: retuning
    `retrieve.tau` must not invalidate this cache.
    """
    doc_ids = {b.doc_id for b in blocks}
    if len(doc_ids) > 1:
        raise MixedDocumentError(
            f"chunk_blocks takes one document at a time, got {sorted(doc_ids)}"
        )

    ordered = sorted(blocks, key=lambda b: (b.reading_order, b.page))
    input_hash = hash_rows(ordered)
    with stage_timer("chunk", input_hash) as span:

        def compute() -> list[Chunk]:
            sections = _sections(ordered)
            pairs: list[tuple[list[Block], tuple[str, ...]]] = []
            for i, section in enumerate(sections):
                following = sections[i + 1] if i + 1 < len(sections) else None
                if _is_redundant(section, following):
                    continue
                for group in _merge_short(_pack(section.blocks, params), params.min_tokens):
                    pairs.append((group, section.heading_path))
            return [_build(group, path) for group, path in _fold_orphans(pairs, params.min_tokens)]

        if cache is None:
            result = compute()
            span.n_out = len(result)
        else:
            key = CacheKey(
                stage="chunk",
                input_hash=input_hash,
                stage_version=STAGE_VERSION,
                config_hash=config_hash or _hash_obj(params.model_dump(mode="json")),
            )
            result = cache.get_or_compute(key, Chunk, compute, span=span)
        span.extra["n_blocks"] = len(ordered)
        span.extra["target_tokens"] = params.target_tokens
    return result
