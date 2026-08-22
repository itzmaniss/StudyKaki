"""Structure-aware chunking — ARCHITECTURE.md §3 `chunk`.

§3 calls this the highest-leverage knob in the system, so the rules it has to obey are worth
stating plainly:

* **Never split mid-block.** A block is the atom. A block larger than `target_tokens` becomes
  a chunk on its own rather than being cut in half.
* **Group by reading order within a heading.** A heading opens a section; every block until the
  next heading belongs to it. Chunks never straddle a heading, and overlap never crosses one
  either — carrying the tail of "3.1 Light Reactions" into the first chunk of "3.2 Calvin Cycle"
  pollutes both.
* **Count characters for CJK/Thai**, whitespace tokens elsewhere. A 400-"token" Chinese chunk
  measured by `str.split()` would be an entire chapter.
* **`bbox_union` is the true union of member block bboxes and `page_start`/`page_end` span
  them.** Citations are built from these and §0 non-negotiable 2 says provenance cannot be
  retrofitted.

`Chunk.text` is exactly the concatenation of its member blocks — nothing else — so
`token_count` describes what is actually stored and `block_ids` accounts for every character.
`heading_path` travels alongside for `ingest/embed.py` to prefix if it wants to; prepending it
here would put text in the chunk that no block can vouch for.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from core.cache import CacheKey, StageCache, hash_rows, stage_timer
from core.config import ChunkConfig, _hash_obj
from core.schema import BBox, Block, Chunk
from ingest.normalize import lang_for_script

log = structlog.get_logger(__name__)

STAGE_VERSION = "chunk/1"

BLOCK_SEPARATOR = "\n\n"

# CJK and Thai are not whitespace-delimited; one non-space character is roughly one subword
# token for the XLM-R style vocabulary BGE-M3 uses.
CHAR_COUNTED_SCRIPTS = frozenset({"hans", "hant", "jpan", "thai"})

# "3.2 Photosynthesis" -> depth 2. "Chapter 3" has no leading numeral, so it is depth 1 —
# which is exactly the ["Chapter 3", "3.2 Photosynthesis"] path §2 uses as its example.
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

    "Chapter 3" immediately followed by "3.1 Light Reactions" would otherwise index as a
    two-token chunk that can only ever be a false positive. Its text is not lost: it stays in
    the `heading_path` of every section beneath it. A trailing bare heading with nothing under
    it *is* kept, because then the path is the only place the text would survive.
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

    Returning everything would make the next chunk a strict superset of this one — the same
    text embedded twice, and two identical hits at query time.
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
    """Fold an undersized chunk back into its predecessor rather than shipping or dropping it.

    Dropping loses content; shipping a 9-token chunk gives the retriever a fragment with no
    context. Merging keeps every block reachable, at the cost of one slightly oversized chunk.
    """
    merged: list[list[Block]] = []
    for group in groups:
        tokens = sum(count_tokens(b.text, b.script) for b in group)
        if merged and tokens < min_tokens:
            seen = {b.block_id for b in merged[-1]}
            merged[-1].extend(b for b in group if b.block_id not in seen)
        else:
            merged.append(list(group))
    return merged


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

    `config_hash` defaults to the hash of `params` alone, which is byte-identical to
    `Config.chunk_config_hash` — retuning `retrieve.tau` must not invalidate this cache.
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
            out: list[Chunk] = []
            for i, section in enumerate(sections):
                following = sections[i + 1] if i + 1 < len(sections) else None
                if _is_redundant(section, following):
                    continue
                groups = _merge_short(_pack(section.blocks, params), params.min_tokens)
                out.extend(_build(group, section.heading_path) for group in groups)
            return out

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
