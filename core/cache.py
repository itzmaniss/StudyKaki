"""Content-addressed stage cache — ARCHITECTURE.md §0 non-negotiable 1, §0.1 storage layout.

Every ingest stage is keyed on `(input_hash, stage_version, config_hash)` and lands at
`<root>/<stage>/<content_hash>.parquet` with zstd compression. Without this the developer
re-OCRs the whole corpus every time he tunes a chunk parameter.

The root is an argument, never a module constant: tests point it at `tmp_path`, production
points it at `cfg.paths.data_dir / "cache"`.

Writes go to a temp file and are `os.replace`d into position, so an interrupted run leaves
either the previous entry or nothing — never a half-written parquet that a later run would
happily read back as truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import polars as pl
import structlog

from core.config import Config
from core.schema import Block, Chunk, Document

log = structlog.get_logger(__name__)

RowT = TypeVar("RowT", Document, Block, Chunk)

_STAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_SCALAR_DTYPES: dict[type, pl.DataType] = {
    str: pl.Utf8(),
    bool: pl.Boolean(),
    int: pl.Int64(),
    float: pl.Float64(),
}


class CacheCorruptError(RuntimeError):
    """A cache entry exists but cannot be read back as the requested row type."""


@dataclass(frozen=True)
class CacheKey:
    stage: str
    input_hash: str
    stage_version: str
    config_hash: str

    def __post_init__(self) -> None:
        if not _STAGE_RE.match(self.stage):
            raise ValueError(
                f"stage {self.stage!r} must match {_STAGE_RE.pattern} — it becomes a directory name"
            )
        for name in ("input_hash", "stage_version", "config_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")

    @property
    def content_hash(self) -> str:
        payload = "\x1f".join(
            (self.stage, self.input_hash, self.stage_version, self.config_hash)
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class StageSpan:
    """Mutable handle a stage fills in so `stage_timer` can log the outcome."""

    stage: str
    input_hash: str
    n_out: int = 0
    cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@contextmanager
def stage_timer(stage: str, input_hash: str) -> Iterator[StageSpan]:
    span = StageSpan(stage=stage, input_hash=input_hash)
    started = time.perf_counter()
    try:
        yield span
    finally:
        log.info(
            "ingest.stage",
            stage=stage,
            input_hash=input_hash[:16],
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            n_out=span.n_out,
            cached=span.cached,
            **span.extra,
        )


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_rows(rows: Sequence[RowT]) -> str:
    """Stable hash of a stage's input rows, so stage N+1 keys off stage N's exact output."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row.to_row(), sort_keys=True, separators=(",", ":"), default=str).encode()
        )
        digest.update(b"\x1e")
    return digest.hexdigest()


def _dtype_for(annotation: Any) -> pl.DataType:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"cannot map union {annotation!r} to a parquet dtype")
        return _dtype_for(inner[0])
    if origin in (list, tuple):
        args = [a for a in get_args(annotation) if a is not Ellipsis]
        return pl.List(_dtype_for(args[0] if args else str))
    try:
        return _SCALAR_DTYPES[annotation]
    except KeyError as exc:
        raise TypeError(f"no parquet dtype for {annotation!r}") from exc


def _schema_for(cls: type[RowT]) -> dict[str, pl.DataType]:
    """Explicit schema so a legitimately empty stage output still round-trips."""
    hints = get_type_hints(cls)
    return {f.name: _dtype_for(hints[f.name]) for f in fields(cls)}


class StageCache:
    """Idempotent, resumable parquet cache. `root` is injected; nothing here knows about `data/`."""

    def __init__(self, root: Path, *, enabled: bool = True, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.compression = compression

    @classmethod
    def from_config(cls, cfg: Config, **kw: Any) -> StageCache:
        return cls(cfg.resolve(cfg.paths.data_dir) / "cache", **kw)

    def path(self, key: CacheKey) -> Path:
        return self.root / key.stage / f"{key.content_hash}.parquet"

    def has(self, key: CacheKey) -> bool:
        return self.enabled and self.path(key).is_file()

    def load(self, key: CacheKey, row_type: type[RowT]) -> list[RowT] | None:
        if not self.enabled:
            return None
        target = self.path(key)
        if not target.is_file():
            return None
        try:
            frame = pl.read_parquet(target)
        except (pl.exceptions.ComputeError, OSError) as exc:
            log.warning("cache.unreadable", stage=key.stage, path=str(target), error=str(exc))
            return None
        try:
            return [row_type.from_row(row) for row in frame.to_dicts()]
        except (TypeError, ValueError) as exc:
            raise CacheCorruptError(
                f"cache entry {target} does not decode as {row_type.__name__}: {exc}"
            ) from exc

    def store(self, key: CacheKey, rows: Sequence[RowT], row_type: type[RowT]) -> Path:
        target = self.path(key)
        if not self.enabled:
            return target
        for row in rows:
            if not isinstance(row, row_type):
                raise TypeError(f"expected all rows to be {row_type.__name__}, got {type(row)}")
        frame = pl.DataFrame([row.to_row() for row in rows], schema=_schema_for(row_type))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            frame.write_parquet(tmp, compression=self.compression)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        return target

    def get_or_compute(
        self,
        key: CacheKey,
        row_type: type[RowT],
        compute: Callable[[], Sequence[RowT]],
        *,
        span: StageSpan | None = None,
    ) -> list[RowT]:
        hit = self.load(key, row_type)
        if hit is not None:
            if span is not None:
                span.cached = True
                span.n_out = len(hit)
            return hit
        rows = list(compute())
        self.store(key, rows, row_type)
        if span is not None:
            span.cached = False
            span.n_out = len(rows)
        return rows
