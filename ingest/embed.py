"""BGE-M3 INT8 embeddings through OpenVINO — ARCHITECTURE.md §3 `embed`, §3.1, §7.5.

* Batch 16–32, never one call per chunk — a per-chunk loop pays full graph overhead for one
  row and is ~10x slower for no benefit (§7.5).
* The fingerprint decides the maths, not this module. `pooling`, `normalize`, `max_len` and
  the prefixes come from the manifest (§3.1 rule 3); hardcoding BGE-M3's empty prefixes
  would destroy retrieval the day an E5-family model is swapped in.
* Vectors cache as `.npy`, never parquet (§0.1), keyed on
  `(text hash, stage_version, fingerprint hash)`. A cache hit never compiles the IR.

`embed_queries` and `embed_passages` differ only in prefix; everything after is identical,
which is what makes query- and index-time vectors comparable.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import openvino as ov
import structlog

from core.cache import CacheKey, StageCache, StageSpan, stage_timer
from core.config import Config, _hash_obj
from core.schema import Chunk
from models.registry import (
    EmbeddingSpec,
    LoadedModel,
    RegistryError,
    embedder_fingerprint,
    load_model,
)

log = structlog.get_logger(__name__)

STAGE_VERSION = "embed/1"

#: §3 / §7.5. 32 is the top of the sanctioned range and the best throughput on CPU here.
DEFAULT_BATCH_SIZE = 32

TOKENIZER_XML_NAME = "openvino_tokenizer.xml"
HIDDEN_STATE_OUTPUT = "last_hidden_state"

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"

#: openvino_tokenizers' custom ops read weights as f32, but arm64's CPU plugin defaults the
#: hint to f16 and they fail with "element type f16, is not representable as pointer to
#: f32". Pinning it on the tokenizer compile costs nothing on Intel.
TOKENIZER_OV_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}


class EmbedError(RuntimeError):
    """Base for embedding failures. Always names the next action."""


class TokenizerUnavailable(EmbedError):
    pass


class Tokenizer(Protocol):
    """`texts -> (input_ids, attention_mask)`, both int64 `[batch, seq]`, padded per batch."""

    def __call__(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...


class OVTokenizer:
    """The `openvino_tokenizer.xml` written next to the model IR by `models/convert.py`.

    Behind the `Tokenizer` protocol so tests substitute a trivial one — the real IR is 22 MB
    and sits beside uncommitted weights (§7.3).
    """

    def __init__(self, compiled: ov.CompiledModel) -> None:
        self._compiled = compiled
        self._request = compiled.create_infer_request()

    @classmethod
    def from_ir_dir(cls, ir_dir: Path, *, core: ov.Core | None = None) -> OVTokenizer:
        xml = Path(ir_dir) / TOKENIZER_XML_NAME
        if not xml.exists():
            raise TokenizerUnavailable(
                f"tokenizer IR missing at {xml} — run `uv run python -m scripts.setup` to "
                f"convert it (§7.3)"
            )
        try:
            import openvino_tokenizers
        except ImportError as e:
            raise TokenizerUnavailable(
                "openvino-tokenizers is not importable — check that openvino, "
                "openvino-tokenizers and openvino-genai are the same version (§7.1)"
            ) from e

        core = core or ov.Core()
        # Importing only extends `ov.Core`s created after it, and the registry hands us an
        # older one. Explicit registration makes load order irrelevant.
        try:
            core.add_extension(str(openvino_tokenizers._ext_path))
        except (RuntimeError, AttributeError) as e:
            log.warning("embed.tokenizer_extension_not_registered", error=str(e))
        try:
            compiled = core.compile_model(str(xml), "CPU", TOKENIZER_OV_CONFIG)
        except RuntimeError as e:
            raise TokenizerUnavailable(f"could not compile {xml} on CPU: {e}") from e
        log.info("embed.tokenizer_loaded", ir=str(xml), device="CPU")
        return cls(compiled)

    def __call__(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        # A `string` tensor fills from a numpy unicode array (`<U`). An `object` array is
        # rejected outright, and a bare list reads as a scalar and fails the shape check.
        result = self._request.infer([np.array(list(texts))])
        return (
            np.asarray(result[INPUT_IDS], dtype=np.int64),
            np.asarray(result[ATTENTION_MASK], dtype=np.int64),
        )


class Embedder:
    """A compiled encoder plus the fingerprint that says how to use it.

    Not thread-safe — one `InferRequest` is reused across batches, because allocating per
    batch is measurable at this size.
    """

    def __init__(
        self,
        *,
        compiled: ov.CompiledModel,
        tokenizer: Tokenizer,
        spec: EmbeddingSpec,
        fingerprint: dict[str, Any],
        name: str = "embedder",
        device: str = "CPU",
    ) -> None:
        self.spec = spec
        self.fingerprint = dict(fingerprint)
        self.name = name
        self.device = device
        self._tokenizer = tokenizer
        self._compiled = compiled
        self._request = compiled.create_infer_request()
        self._inputs = {n for port in compiled.inputs for n in port.get_names()}
        outputs = {n for port in compiled.outputs for n in port.get_names()}
        self._output: str | int = HIDDEN_STATE_OUTPUT if HIDDEN_STATE_OUTPUT in outputs else 0

    @classmethod
    def load(
        cls,
        cfg: Config,
        *,
        manifest_path: str | Path | None = None,
        core: ov.Core | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> Embedder:
        core = core or ov.Core()
        model: LoadedModel = load_model("embedder", cfg, manifest_path=manifest_path, core=core)
        fingerprint = embedder_fingerprint(cfg, manifest_path=manifest_path)
        spec = model.entry.embedding
        if spec is None:  # pragma: no cover — embedder_fingerprint raises first
            raise RegistryError(f"{model.name}: manifest entry has no 'embedding' block (§3.1)")
        return cls(
            compiled=model.compiled,
            tokenizer=tokenizer or OVTokenizer.from_ir_dir(model.entry.ir_dir, core=core),
            spec=spec,
            fingerprint=fingerprint,
            name=model.name,
            device=model.device,
        )

    @property
    def dim(self) -> int:
        return self.spec.dim

    def embed_queries(self, texts: Sequence[str], **kw: Any) -> np.ndarray:
        return self.embed_texts(texts, prefix=self.spec.query_prefix, **kw)

    def embed_passages(self, texts: Sequence[str], **kw: Any) -> np.ndarray:
        return self.embed_texts(texts, prefix=self.spec.passage_prefix, **kw)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        prefix: str = "",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prefixed = [prefix + t for t in texts] if prefix else texts
        out = np.empty((len(prefixed), self.dim), dtype=np.float32)
        truncated = 0
        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start : start + batch_size]
            ids, mask = self._tokenizer(batch)
            if ids.shape[1] > self.spec.max_len:
                truncated += int((mask.sum(axis=1) > self.spec.max_len).sum())
                ids = ids[:, : self.spec.max_len]
                mask = mask[:, : self.spec.max_len]
            out[start : start + len(batch)] = self._forward(ids, mask)

        if truncated:
            log.warning(
                "embed.truncated",
                model=self.name,
                n_texts=truncated,
                max_len=self.spec.max_len,
                hint="chunks longer than the embedder window lose their tail",
            )
        if self.spec.normalize:
            out = _l2_normalize(out)
        return out

    def _forward(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feed: dict[str, np.ndarray] = {INPUT_IDS: ids}
        if ATTENTION_MASK in self._inputs:
            feed[ATTENTION_MASK] = mask
        if TOKEN_TYPE_IDS in self._inputs:
            feed[TOKEN_TYPE_IDS] = np.zeros_like(ids)

        hidden = np.asarray(self._request.infer(feed)[self._output], dtype=np.float32)
        pooled = _pool(hidden, mask, self.spec.pooling)
        if pooled.shape[1] != self.dim:
            raise EmbedError(
                f"{self.name}: model produced dim {pooled.shape[1]} but the manifest promises "
                f"{self.dim} — the IR and its 'embedding' block disagree (§3.1)"
            )
        return pooled


def _pool(hidden: np.ndarray, mask: np.ndarray, pooling: str) -> np.ndarray:
    """`hidden` is `[batch, seq, dim]`; a model that already pools gives `[batch, dim]`."""
    if hidden.ndim == 2:
        return hidden
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        weights = mask[:, : hidden.shape[1], None].astype(np.float32)
        return (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
    raise EmbedError(
        f"unsupported pooling {pooling!r} — the manifest's embedding block must say "
        f"'cls' or 'mean' (§3.1)"
    )


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Zero rows stay zero rather than becoming NaN — an empty chunk must not poison a search."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0.0, 1.0, norms)


class VectorCache:
    """`.npy` sibling of `core.cache.StageCache` — §0.1 forbids vectors in parquet.

    Same key, root and atomic-replace discipline; the extension prevents collision with a
    `StageCache` entry in the same stage dir.
    """

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    @classmethod
    def from_stage_cache(cls, cache: StageCache) -> VectorCache:
        return cls(cache.root, enabled=cache.enabled)

    def path(self, key: CacheKey) -> Path:
        return self.root / key.stage / f"{key.content_hash}.npy"

    def has(self, key: CacheKey) -> bool:
        return self.enabled and self.path(key).is_file()

    def load(self, key: CacheKey) -> np.ndarray | None:
        if not self.enabled:
            return None
        target = self.path(key)
        if not target.is_file():
            return None
        try:
            return np.load(target)
        except (OSError, ValueError) as e:
            log.warning("cache.unreadable", stage=key.stage, path=str(target), error=str(e))
            return None

    def store(self, key: CacheKey, vectors: np.ndarray) -> Path:
        target = self.path(key)
        if not self.enabled:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with tmp.open("wb") as fh:  # a handle keeps np.save from appending its own .npy
                np.save(fh, np.ascontiguousarray(vectors, dtype=np.float32))
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        return target

    def get_or_compute(
        self,
        key: CacheKey,
        compute: Callable[[], np.ndarray],
        *,
        span: StageSpan | None = None,
    ) -> np.ndarray:
        hit = self.load(key)
        if hit is not None:
            if span is not None:
                span.cached = True
                span.n_out = int(hit.shape[0])
            return hit
        vectors = compute()
        self.store(key, vectors)
        if span is not None:
            span.cached = False
            span.n_out = int(vectors.shape[0])
        return vectors


def hash_texts(texts: Sequence[str]) -> str:
    """Embedding is a pure function of text, so the cache keys on text alone.

    Not `hash_rows(chunks)`: a chunk whose bbox moved a pixel embeds identically, and
    re-encoding an unchanged corpus is what §0 non-negotiable 1 exists to prevent.
    """
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def embed_config_hash(fingerprint: dict[str, Any]) -> str:
    """The embedder *is* the config for this stage — weights, pooling, prefixes and all."""
    return _hash_obj(fingerprint)


def embed_chunks(
    chunks: Sequence[Chunk],
    cfg: Config,
    *,
    embedder: Embedder | None = None,
    cache: StageCache | VectorCache | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    manifest_path: str | Path | None = None,
) -> np.ndarray:
    """Chunks -> `[n_chunks, dim]` float32, row-aligned with `chunks`, cached on `.npy`.

    `embedder` is optional so a cache hit never compiles the IR — the key needs only the
    fingerprint, a hash of the weights on disk.
    """
    texts = [c.text for c in chunks]
    fingerprint = (
        embedder.fingerprint
        if embedder is not None
        else embedder_fingerprint(cfg, manifest_path=manifest_path)
    )
    input_hash = hash_texts(texts)

    with stage_timer("embed", input_hash) as span:

        def compute() -> np.ndarray:
            nonlocal embedder
            if embedder is None:
                embedder = Embedder.load(cfg, manifest_path=manifest_path)
            return embedder.embed_passages(texts, batch_size=batch_size)

        vectors = _resolve_cache(cache)
        if vectors is None:
            result = compute()
            span.n_out = len(result)
        else:
            key = CacheKey(
                stage="embed",
                input_hash=input_hash,
                stage_version=STAGE_VERSION,
                config_hash=embed_config_hash(fingerprint),
            )
            result = vectors.get_or_compute(key, compute, span=span)
        span.extra["dim"] = int(result.shape[1]) if result.size else int(fingerprint["dim"])
        span.extra["batch_size"] = batch_size

    if len(result) != len(chunks):
        raise EmbedError(
            f"embedded {len(result)} vectors for {len(chunks)} chunks — the cache entry does "
            f"not match its key; delete it and re-run"
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def _resolve_cache(cache: StageCache | VectorCache | None) -> VectorCache | None:
    if cache is None:
        return None
    if isinstance(cache, VectorCache):
        return cache
    return VectorCache.from_stage_cache(cache)
