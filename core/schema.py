"""Data contract for the whole pipeline — ARCHITECTURE.md §2.

Provenance (`doc_id`, `page`, `bbox`) rides on every Block and Chunk. Citations depend on
it and it cannot be retrofitted, so nothing downstream may drop these fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, TypeVar

BBox = tuple[float, float, float, float]

SCRIPTS: frozenset[str] = frozenset(
    {"latn", "hans", "hant", "jpan", "kore", "taml", "thai", "deva", "arab", "cyrl", "unknown"}
)

BLOCK_KINDS: frozenset[str] = frozenset({"heading", "paragraph", "table", "caption", "list"})

T = TypeVar("T", bound="_Row")


def _check_bbox(bbox: BBox, field_name: str = "bbox") -> None:
    if len(bbox) != 4:
        raise ValueError(f"{field_name} must have 4 elements, got {len(bbox)}")
    x0, y0, x1, y1 = bbox
    for name, v in zip(("x0", "y0", "x1", "y1"), bbox, strict=True):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{field_name}.{name} must be normalised to 0-1, got {v}")
    if x1 < x0 or y1 < y0:
        raise ValueError(f"{field_name} is inverted: {bbox}")


@dataclass(frozen=True)
class _Row:
    """Shared parquet round-trip behaviour. Subclasses stay plain frozen dataclasses."""

    # Tuple-typed fields are stored as lists in parquet and rebuilt on read.
    _TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls: type[T], row: dict[str, Any]) -> T:
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in row.items() if k in known}
        for name in cls._TUPLE_FIELDS:
            if name in data and data[name] is not None:
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class Document(_Row):
    doc_id: str
    filename: str
    mime: str
    n_pages: int
    has_text_layer: bool
    pipeline_version: str

    def __post_init__(self) -> None:
        if self.n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {self.n_pages}")
        if not self.doc_id:
            raise ValueError("doc_id must not be empty")


@dataclass(frozen=True)
class Block(_Row):
    block_id: str
    doc_id: str
    page: int
    bbox: BBox
    kind: str
    reading_order: int
    script: str
    text: str
    ocr_confidence: float | None

    _TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("bbox",)

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"page is 1-indexed, got {self.page}")
        _check_bbox(self.bbox)
        if self.kind not in BLOCK_KINDS:
            raise ValueError(
                f"unknown block kind {self.kind!r}, expected one of {sorted(BLOCK_KINDS)}"
            )
        if self.script not in SCRIPTS:
            raise ValueError(f"unknown script {self.script!r}, expected one of {sorted(SCRIPTS)}")
        if self.ocr_confidence is not None and not 0.0 <= self.ocr_confidence <= 1.0:
            raise ValueError(f"ocr_confidence must be 0-1, got {self.ocr_confidence}")


@dataclass(frozen=True)
class Chunk(_Row):
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    block_ids: list[str]
    bbox_union: BBox
    heading_path: list[str]
    text: str
    token_count: int
    lang: str
    script: str

    _TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("bbox_union",)

    def __post_init__(self) -> None:
        if self.page_start < 1:
            raise ValueError(f"page_start is 1-indexed, got {self.page_start}")
        if self.page_end < self.page_start:
            raise ValueError(f"page_end {self.page_end} precedes page_start {self.page_start}")
        _check_bbox(self.bbox_union, "bbox_union")
        if not self.block_ids:
            raise ValueError("chunk must reference at least one block")
        if self.token_count < 0:
            raise ValueError(f"token_count must be >= 0, got {self.token_count}")


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank is 1-indexed, got {self.rank}")


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Retrieved]
    abstained: bool
    trace_id: str

    def __post_init__(self) -> None:
        if self.abstained and self.citations:
            raise ValueError("an abstained answer must carry no citations")
