"""Dense retrieval — ARCHITECTURE.md §4, §3, §3.1.

    query -> verify fingerprint -> embed (manifest's query_prefix)
          -> exact cosine over the memmapped flat index -> top-k -> Retrieved

Flat exact scan, blocked over the memmap: no ANN at this scale, and peak RSS stays a
constant rather than tracking corpus size. Norms are computed in the same pass, so an
un-normalised index still ranks by angle.

Fingerprint is verified on every query (§3.1 rule 2), not once per process. Embedder drift
does not look like drift — retrieval collapses and reads as a chunking bug — so it must
hard-fail. `ir_sha256` memoises, so the check is a JSON read.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
import structlog

from core.config import Config
from core.schema import Chunk, Retrieved
from models.registry import verify_fingerprint
from retrieve.retriever import abstains

log = structlog.get_logger("retrieve.dense")

CHUNKS_NAME = "chunks.parquet"
VECTORS_NAME = "vectors.npy"
INDEX_MANIFEST_NAME = "index_manifest.json"

#: ~32 MB at dim=1024 float32 — big enough for BLAS, small enough to bound RSS.
DEFAULT_BLOCK_ROWS = 8192

_CHUNK_COLUMNS = frozenset(f.name for f in fields(Chunk))


class DenseIndexError(RuntimeError):
    """Base for index-side failures. Always names the next action."""


class IndexNotFound(DenseIndexError):
    pass


class IndexCorrupt(DenseIndexError):
    pass


@runtime_checkable
class QueryEncoder(Protocol):
    """Already-prefixed texts -> `[n, dim]` float32.

    Prefix and model choice both come from the index manifest, never from here — that is
    what keeps query-time and index-time embeddings identical (§3.1).
    """

    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class DenseIndex:
    """On-disk flat index (§0.1), read-only.

    Chunks stay a polars frame, decoded to `Chunk` only for rows that make top-k.
    """

    path: Path
    manifest: dict[str, Any]
    frame: pl.DataFrame
    vectors: np.ndarray

    @property
    def n_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def index_id(self) -> str:
        return str(self.manifest.get("index_id", ""))

    @property
    def embedder(self) -> dict[str, Any]:
        block = self.manifest.get("embedder")
        return block if isinstance(block, dict) else {}

    @property
    def query_prefix(self) -> str:
        """§3.1 rule 3 — the prefix travels with the index, never with the code."""
        return str(self.embedder.get("query_prefix", ""))

    def chunk_at(self, row: int) -> Chunk:
        return Chunk.from_row(self.frame.row(row, named=True))


def load_index(path: str | Path) -> DenseIndex:
    """Open `<index_dir>/{chunks.parquet, vectors.npy, index_manifest.json}` (§0.1).

    Vectors are memmapped: opening costs no RAM until the scan.
    """
    index_dir = Path(path)
    if not index_dir.is_dir():
        raise IndexNotFound(
            f"index directory not found: {index_dir} — build one with `ingest/pipeline.py`"
        )

    manifest_path = index_dir / INDEX_MANIFEST_NAME
    chunks_path = index_dir / CHUNKS_NAME
    vectors_path = index_dir / VECTORS_NAME
    for required in (manifest_path, chunks_path, vectors_path):
        if not required.is_file():
            raise IndexNotFound(
                f"{index_dir} is not a complete index — missing {required.name} "
                f"(§0.1 requires {CHUNKS_NAME}, {VECTORS_NAME} and {INDEX_MANIFEST_NAME})"
            )

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise IndexCorrupt(f"{manifest_path} is not valid JSON: {e}") from e
    if not isinstance(manifest, dict):
        raise IndexCorrupt(f"{manifest_path} must contain a JSON object")

    try:
        frame = pl.read_parquet(chunks_path)
    except (pl.exceptions.PolarsError, OSError) as e:
        raise IndexCorrupt(f"{chunks_path} is unreadable: {e}") from e

    missing = _CHUNK_COLUMNS - set(frame.columns)
    if missing:
        raise IndexCorrupt(
            f"{chunks_path} is missing Chunk columns {sorted(missing)} — provenance cannot "
            f"be retrofitted (§0.2); re-index"
        )

    vectors = _load_vectors(vectors_path)
    _check_alignment(index_dir, manifest, frame, vectors)

    if frame.height:
        # Fail now, not on the query that happens to retrieve the bad row.
        try:
            Chunk.from_row(frame.row(0, named=True))
        except (TypeError, ValueError) as e:
            raise IndexCorrupt(f"{chunks_path} does not decode as Chunk: {e}") from e

    log.info(
        "index.opened",
        path=str(index_dir),
        index_id=str(manifest.get("index_id", ""))[:19],
        n_vectors=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
    )
    return DenseIndex(path=index_dir, manifest=manifest, frame=frame, vectors=vectors)


def _load_vectors(vectors_path: Path) -> np.ndarray:
    try:
        vectors = np.load(vectors_path, mmap_mode="r")
    except (ValueError, OSError) as e:
        raise IndexCorrupt(f"{vectors_path} is not a readable .npy array: {e}") from e

    if vectors.ndim != 2:
        raise IndexCorrupt(
            f"{vectors_path} has shape {vectors.shape}, expected 2-D [n_chunks, dim]"
        )
    if vectors.dtype != np.float32:
        raise IndexCorrupt(
            f"{vectors_path} has dtype {vectors.dtype}, expected float32 — §0.1 pins "
            f"contiguous float32; re-index rather than casting at query time"
        )
    return vectors


def _check_alignment(
    index_dir: Path,
    manifest: dict[str, Any],
    frame: pl.DataFrame,
    vectors: np.ndarray,
) -> None:
    """Row i of vectors.npy is row i of chunks.parquet. Nothing else joins them."""
    if frame.height != vectors.shape[0]:
        raise IndexCorrupt(
            f"{index_dir}: {frame.height} chunks but {vectors.shape[0]} vectors — the index "
            f"joins them by row position, so a mismatch means every citation is wrong; re-index"
        )
    declared = manifest.get("n_vectors")
    if declared is not None and int(declared) != int(vectors.shape[0]):
        raise IndexCorrupt(
            f"{index_dir}: manifest declares n_vectors={declared} but vectors.npy holds "
            f"{vectors.shape[0]} — re-index"
        )
    embedder = manifest.get("embedder")
    declared_dim = embedder.get("dim") if isinstance(embedder, dict) else None
    if declared_dim is not None and int(declared_dim) != int(vectors.shape[1]):
        raise IndexCorrupt(
            f"{index_dir}: manifest declares dim={declared_dim} but vectors.npy is "
            f"{vectors.shape[1]}-wide — re-index"
        )


def default_index_dir(cfg: Config) -> Path:
    """Resolve `data/index/<index_id>/` when exactly one index exists.

    Refuses to guess between several — callers that know pass the path.
    """
    root = cfg.resolve(cfg.paths.data_dir) / "index"
    if not root.is_dir():
        raise IndexNotFound(f"no index root at {root} — run the ingest pipeline first")
    candidates = sorted(p for p in root.iterdir() if (p / INDEX_MANIFEST_NAME).is_file())
    if not candidates:
        raise IndexNotFound(f"{root} contains no index (no {INDEX_MANIFEST_NAME} found)")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise IndexNotFound(f"{root} holds several indexes ({names}) — pass one explicitly")
    return candidates[0]


class TextEmbedder(Protocol):
    """The slice of `ingest.embed.Embedder` this module needs."""

    def embed_texts(self, texts: Sequence[str], *, prefix: str = "") -> np.ndarray: ...


def query_encoder(
    cfg: Config,
    *,
    models_manifest_path: str | Path | None = None,
    embedder: TextEmbedder | None = None,
) -> QueryEncoder:
    """Bind the query path to `ingest/embed.py` — the model that built the index.

    Lazy import: `ingest.embed` compiles a 544 MB IR, and this module must stay importable
    without it. Empty prefix on purpose — `DenseRetriever` already applied the *index*
    manifest's prefix, which is the authority (§3.1 rule 3); prefixing again would double it.
    """
    if embedder is None:
        from ingest.embed import Embedder

        embedder = Embedder.load(cfg, manifest_path=models_manifest_path)
    bound = embedder

    def encode(texts: Sequence[str]) -> np.ndarray:
        return bound.embed_texts(texts, prefix="")

    return encode


class DenseRetriever:
    """`retrieve.retriever.Retriever` over a flat float32 index.

    `encode` is injected so tests can assert ranking arithmetic without a 544 MB IR.
    """

    @classmethod
    def open(
        cls,
        cfg: Config,
        index_path: str | Path | None = None,
        *,
        models_manifest_path: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        block_rows: int = DEFAULT_BLOCK_ROWS,
    ) -> DenseRetriever:
        """Resolve the index, compile the embedder once, return a retriever.

        Once here, not per query: a `retrieve()` that reloaded would be all `compile_model`.
        """
        index = load_index(index_path if index_path is not None else default_index_dir(cfg))
        return cls(
            index,
            cfg,
            query_encoder(cfg, models_manifest_path=models_manifest_path, embedder=embedder),
            models_manifest_path=models_manifest_path,
            block_rows=block_rows,
        )

    def __init__(
        self,
        index: DenseIndex,
        cfg: Config,
        encode: QueryEncoder,
        *,
        models_manifest_path: str | Path | None = None,
        block_rows: int = DEFAULT_BLOCK_ROWS,
    ) -> None:
        if block_rows < 1:
            raise ValueError(f"block_rows must be >= 1, got {block_rows}")
        self.index = index
        self.cfg = cfg
        self.encode = encode
        self.models_manifest_path = models_manifest_path
        self.block_rows = block_rows

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        """Top-`k` chunks by cosine similarity, rank 1 = best (§4)."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not query.strip():
            raise ValueError("query must not be empty")

        started = time.perf_counter()

        # §3.1 rule 2 — every query, before anything else.
        verify_fingerprint(self.index.manifest, self.cfg, manifest_path=self.models_manifest_path)

        vector = self._embed_query(query)
        scores = _cosine_scores(self.index.vectors, vector, self.block_rows)
        rows = _top_k_rows(scores, k)
        hits = [
            Retrieved(chunk=self.index.chunk_at(int(row)), score=float(scores[row]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

        log.info(
            "retrieve.dense",
            index_id=self.index.index_id[:19],
            n_vectors=self.index.n_vectors,
            k=k,
            n_hits=len(hits),
            top_score=round(hits[0].score, 4) if hits else None,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return hits

    def abstains(self, hits: list[Retrieved]) -> bool:
        """§4 threshold, at this retriever's configured tau."""
        return abstains(hits, self.cfg.retrieve.tau)

    def _embed_query(self, query: str) -> np.ndarray:
        prefix = self.index.query_prefix
        raw = self.encode([prefix + query])
        vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim == 2:
            if vector.shape[0] != 1:
                raise DenseIndexError(f"encoder returned {vector.shape[0]} vectors for one query")
            vector = vector[0]
        elif vector.ndim != 1:
            raise DenseIndexError(f"encoder returned shape {vector.shape}, expected [1, dim]")
        if vector.shape[0] != self.index.dim:
            raise DenseIndexError(
                f"query embedding is {vector.shape[0]}-dimensional but the index is "
                f"{self.index.dim}-dimensional — the query was embedded by a different model"
            )
        return vector


def _cosine_scores(vectors: np.ndarray, query: np.ndarray, block_rows: int) -> np.ndarray:
    """Exact cosine over a memmap, blocked to bound RSS.

    Norms come from the same pass, so correctness does not depend on a normalised index.
    Zero-norm rows score 0, not NaN — a degenerate chunk must not outrank a real one.
    """
    n = int(vectors.shape[0])
    scores = np.zeros(n, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if n == 0 or query_norm == 0.0:
        return scores

    # errstate covers the BLAS kernel, not our arithmetic: Accelerate on Apple Silicon sets
    # the sticky FP flags on masked-out padding lanes, so numpy warns on every correct query.
    # The case it could mask is handled below — `where=denom > 0.0` keeps zero-norm rows at 0.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        for start in range(0, n, block_rows):
            block = np.asarray(vectors[start : start + block_rows], dtype=np.float32)
            dots = block @ query
            denom = np.linalg.norm(block, axis=1) * query_norm
            out = scores[start : start + block.shape[0]]
            np.divide(dots, denom, out=out, where=denom > 0.0)
    return scores


def _top_k_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Row indices of the `k` best scores, descending, ties broken by row index.

    `argpartition` is O(n) vs `argsort`'s O(n log n) — scan-bound rather than sort-bound.
    """
    n = int(scores.shape[0])
    k = min(k, n)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    return candidates[np.lexsort((candidates, -scores[candidates]))]
