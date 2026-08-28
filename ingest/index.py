"""Flat float32 index writer — ARCHITECTURE.md §3 `index`, §3.1, §0.1.

    chunks + vectors -> data/index/<index_id>/{chunks.parquet, vectors.npy, index_manifest.json}

Flat and exact (§3): no ANN structure until `eval/run.py` proves one is needed. Vectors are
`.npy`, never parquet (§0.1) — `retrieve/dense.py` memmaps and scans in blocks, where a
parquet round-trip would materialise the whole matrix.

Row `i` of `vectors.npy` is row `i` of `chunks.parquet`; nothing else joins them. A one-row
shift does not fail, it cites the wrong page for every answer, so alignment is verified
against what landed on disk before the manifest is written.

The manifest is the commit record: written last and atomically, so an interrupted build
leaves something `load_index` refuses rather than something it trusts. Its `embedder` block
is the §3.1 fingerprint. No network (§0.3).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

# Reused, not re-derived, so a frame written here decodes identically to one from the
# stage cache — and an empty index round-trips instead of losing its columns.
from core.cache import _schema_for, hash_rows, stage_timer
from core.config import Config
from core.schema import Chunk
from models.registry import embedder_fingerprint

log = structlog.get_logger(__name__)

STAGE_VERSION = "index/1"

# Re-declared rather than imported: ingest must not depend on retrieve. test_index.py
# asserts the two agree, so a rename fails a test instead of orphaning an index.
CHUNKS_NAME = "chunks.parquet"
VECTORS_NAME = "vectors.npy"
INDEX_MANIFEST_NAME = "index_manifest.json"

COMPRESSION = "zstd"

INDEX_ID_PREFIX = "sha256:"


class IndexBuildError(RuntimeError):
    """A refusal to write an index that would be wrong. Always names the next action."""


@dataclass(frozen=True)
class BuiltIndex:
    """What a build produced, and where. `reused` is True when nothing was rewritten."""

    path: Path
    index_id: str
    manifest: dict[str, Any]
    reused: bool = False

    @property
    def n_vectors(self) -> int:
        return int(self.manifest["n_vectors"])

    @property
    def dim(self) -> int:
        return int(self.manifest["embedder"]["dim"])

    @property
    def chunks_path(self) -> Path:
        return self.path / CHUNKS_NAME

    @property
    def vectors_path(self) -> Path:
        return self.path / VECTORS_NAME

    @property
    def manifest_path(self) -> Path:
        return self.path / INDEX_MANIFEST_NAME


def index_root(cfg: Config) -> Path:
    """`data/index/` (§0.1), resolved through the config so tests can point it at `tmp_path`."""
    return cfg.resolve(cfg.paths.data_dir) / "index"


def index_id_for(
    chunks: Sequence[Chunk],
    fingerprint: dict[str, Any],
    chunk_config_hash: str,
) -> str:
    """Content address: chunks, the embedder that vectorised them, and the chunking.

    An unchanged corpus lands on the same directory; a changed embedder or chunk parameter
    produces a different index rather than a half-updated one.
    """
    digest = hashlib.sha256()
    digest.update(hash_rows(chunks).encode())
    digest.update(b"\x1f")
    digest.update(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\x1f")
    digest.update(chunk_config_hash.encode())
    return INDEX_ID_PREFIX + digest.hexdigest()


def index_dir_name(index_id: str) -> str:
    """`sha256:abc...` -> `abc...`. The colon is legal on APFS but not everywhere; the full
    prefixed id stays in the manifest, which is what §3.1 actually pins."""
    return index_id.removeprefix(INDEX_ID_PREFIX)


def index_path(root: str | Path, index_id: str) -> Path:
    return Path(root) / index_dir_name(index_id)


def build_index(
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    cfg: Config,
    *,
    root: str | Path | None = None,
    fingerprint: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    reuse_existing: bool = True,
) -> BuiltIndex:
    """Write `chunks` + `vectors` as a flat index and return where it landed.

    `fingerprint` is injectable only so a caller that already computed it (the pipeline does,
    to key the embed cache) does not re-hash 544 MB of weights; when omitted it is read from
    the model manifest. It is never inferred from the vectors — §3.1 rule 4 is that two INT8
    quantizations of one checkpoint are different models, and the array cannot tell them apart.
    """
    chunks = list(chunks)
    fingerprint = fingerprint or embedder_fingerprint(cfg, manifest_path=manifest_path)
    matrix = _validated(chunks, vectors, fingerprint)

    index_id = index_id_for(chunks, fingerprint, cfg.chunk_config_hash)
    target = index_path(root if root is not None else index_root(cfg), index_id)

    with stage_timer("index", index_dir_name(index_id)) as span:
        span.extra["dim"] = int(matrix.shape[1])
        span.extra["path"] = str(target)

        if reuse_existing:
            existing = _reusable(target, index_id, fingerprint, len(chunks))
            if existing is not None:
                span.cached = True
                span.n_out = len(chunks)
                return BuiltIndex(path=target, index_id=index_id, manifest=existing, reused=True)

        manifest = {
            "index_id": index_id,
            "n_vectors": len(chunks),
            "chunk_config_hash": cfg.chunk_config_hash,
            # Writer's own version, so a future layout change is detectable by a reader.
            "stage_version": STAGE_VERSION,
            "embedder": dict(fingerprint),
        }
        _write_index(target, chunks, matrix, manifest)
        span.n_out = len(chunks)

    log.info(
        "index.built",
        index_id=index_id[:19],
        path=str(target),
        n_vectors=len(chunks),
        dim=int(matrix.shape[1]),
        n_documents=len({c.doc_id for c in chunks}),
    )
    return BuiltIndex(path=target, index_id=index_id, manifest=manifest, reused=False)


def _validated(
    chunks: Sequence[Chunk], vectors: np.ndarray, fingerprint: dict[str, Any]
) -> np.ndarray:
    """Every way an index can be silently wrong, refused before anything is written."""
    matrix = np.ascontiguousarray(np.asarray(vectors), dtype=np.float32)
    if matrix.ndim != 2:
        raise IndexBuildError(
            f"vectors have shape {matrix.shape}, expected 2-D [n_chunks, dim] (§0.1)"
        )
    if matrix.shape[0] != len(chunks):
        raise IndexBuildError(
            f"{len(chunks)} chunks but {matrix.shape[0]} vectors — the index joins them by row "
            f"position, so a mismatch means every citation is wrong"
        )
    declared_dim = int(fingerprint["dim"])
    if matrix.shape[0] and matrix.shape[1] != declared_dim:
        raise IndexBuildError(
            f"vectors are {matrix.shape[1]}-wide but the embedder fingerprint declares "
            f"dim={declared_dim} — the vectors did not come from the manifest's embedder (§3.1)"
        )
    if matrix.size and not bool(np.isfinite(matrix).all()):
        raise IndexBuildError(
            "vectors contain NaN or inf — a non-finite row poisons every cosine score it "
            "touches; re-run the embed stage rather than indexing this"
        )
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise IndexBuildError(
            f"duplicate chunk_id(s) {duplicated} — a citation must resolve to one chunk; the "
            f"same document was probably ingested twice"
        )
    if matrix.shape[0] == 0:
        # Legitimate for image-only PDFs with OCR off, but it looks exactly like a
        # broken pipeline, so it is loud.
        log.warning("index.empty", hint="no chunks to index — every query will abstain")
    return matrix


def _reusable(
    target: Path, index_id: str, fingerprint: dict[str, Any], n_vectors: int
) -> dict[str, Any] | None:
    """A complete index already at `target` for this exact id, embedder and row count.

    Resumability (§0 non-negotiable 1), byte-identical builds only. Anything partial or
    disagreeing is rewritten rather than trusted.
    """
    manifest_file = target / INDEX_MANIFEST_NAME
    present = (manifest_file, target / CHUNKS_NAME, target / VECTORS_NAME)
    if not all(p.is_file() for p in present):
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("index.manifest_unreadable", path=str(manifest_file), error=str(e))
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("index_id") != index_id or int(manifest.get("n_vectors", -1)) != n_vectors:
        return None
    if manifest.get("embedder") != fingerprint:
        return None
    log.info("index.reused", index_id=index_id[:19], path=str(target), n_vectors=n_vectors)
    return manifest


def _write_index(
    target: Path,
    chunks: Sequence[Chunk],
    matrix: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    """Chunks, then vectors, then the manifest — each atomic, in that order.

    Manifest last because `load_index` treats it as proof of a complete index.
    """
    target.mkdir(parents=True, exist_ok=True)

    frame = pl.DataFrame([c.to_row() for c in chunks], schema=_schema_for(Chunk))
    _atomic(target / CHUNKS_NAME, lambda p: frame.write_parquet(p, compression=COMPRESSION))
    _atomic(target / VECTORS_NAME, lambda p: _save_vectors(p, matrix))
    _verify_written(target, chunks, matrix)
    _atomic(
        target / INDEX_MANIFEST_NAME,
        lambda p: p.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n"),
    )


def _save_vectors(path: Path, matrix: np.ndarray) -> None:
    # A file handle, so np.save does not helpfully append a second `.npy` to the temp name.
    with path.open("wb") as fh:
        np.save(fh, matrix)


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _verify_written(target: Path, chunks: Sequence[Chunk], matrix: np.ndarray) -> None:
    """Read back what landed and prove the two files line up row for row.

    Against the disk, not the in-memory sequences — the failure guarded here is a write
    that reordered or dropped rows. Costs one column read and a memmap header.
    """
    written_ids = pl.read_parquet(target / CHUNKS_NAME, columns=["chunk_id"])["chunk_id"].to_list()
    expected_ids = [c.chunk_id for c in chunks]
    if written_ids != expected_ids:
        raise IndexBuildError(
            f"{target / CHUNKS_NAME} came back in a different row order than it was written — "
            f"row i of {VECTORS_NAME} would no longer be row i of {CHUNKS_NAME}"
        )
    stored = np.load(target / VECTORS_NAME, mmap_mode="r")
    if stored.shape != matrix.shape or stored.dtype != np.float32:
        raise IndexBuildError(
            f"{target / VECTORS_NAME} read back as {stored.dtype} {stored.shape}, expected "
            f"float32 {matrix.shape}"
        )
