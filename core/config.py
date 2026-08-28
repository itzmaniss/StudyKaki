"""Config load + hash — ARCHITECTURE.md §6, §0.1.

Config is validated on load, never a bare dict. `config_hash` feeds every stage cache key
(§0 non-negotiable 1), so it must be stable across runs and sensitive to any tunable.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"

Device = Literal["CPU", "GPU", "NPU", "AUTO"]
Precision = Literal["fp32", "fp16", "int8", "int4"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Strict):
    name: str
    device: Device
    precision: Precision


class ModelsConfig(_Strict):
    ocr_det: ModelSpec
    ocr_rec: ModelSpec
    embedder: ModelSpec
    generator: ModelSpec
    # §3 per-script head. Optional: `ocr_rec` alone reads whatever its charset covers, and
    # v5's default is zh+en, so a Tamil page without this decodes to CJK noise at high
    # confidence — worse than failing. Absent means no dedicated head, not broken.
    ocr_rec_taml: ModelSpec | None = None
    ocr_rec_latn: ModelSpec | None = None
    #: §10 cross-encoder. Absent means the rerank arm cannot be enabled, not broken.
    reranker: ModelSpec | None = None


class ChunkConfig(_Strict):
    target_tokens: int = Field(gt=0)
    overlap: int = Field(ge=0)
    min_tokens: int = Field(ge=0)


class RerankConfig(_Strict):
    """§10 cross-encoder rerank. `top_n` is the TTFT mitigation: 20 passes on CPU is 1-2s,
    and recall@20 == recall@10 == 0.980 here, so reranking 10 costs nothing measurable."""

    enabled: bool = False
    top_n: int = Field(default=20, gt=0)


class HybridConfig(_Strict):
    """§10 dense + BM25 through reciprocal rank fusion."""

    enabled: bool = False
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=1.0, ge=0.0)
    lexical_weight: float = Field(default=1.0, ge=0.0)


class RewriteConfig(_Strict):
    """§10 conditional query rewrite. Fires *only* on a short query or an unresolved pronoun —
    a full generation round-trip before retrieval costs 2-4s on CPU, so it must stay rare."""

    enabled: bool = False
    trigger_max_tokens: int = Field(default=5, gt=0)


class RetrieveConfig(_Strict):
    k: int = Field(gt=0)
    n_context: int = Field(gt=0)
    tau: float = Field(ge=0.0, le=1.0)
    # Every V2 arm defaults off, so an unmodified base.yaml is the V1 baseline the
    # before/after numbers are measured against (§10).
    rerank: RerankConfig = RerankConfig()
    hybrid: HybridConfig = HybridConfig()
    rewrite: RewriteConfig = RewriteConfig()

    @property
    def v2_arms(self) -> tuple[str, ...]:
        """Which V2 arms are live, for run labels and trace provenance."""
        on = []
        if self.rerank.enabled:
            on.append("rerank")
        if self.hybrid.enabled:
            on.append("hybrid")
        if self.rewrite.enabled:
            on.append("rewrite")
        return tuple(on)


class GenerateConfig(_Strict):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    #: Ceiling before context is trimmed. Not a model-context limit — INT4 CPU MatMul fails
    #: above ~7k tokens on this build, and Tamil hits it first at ~1.1 chars/token against
    #: English's 2.33. Without a budget the Tamil corpus is unanswerable.
    max_prompt_tokens: int = Field(default=6000, gt=0)
    #: Restate the question's language between the context and "Answer:" (BLOCKERS #21).
    #: `SYSTEM_INSTRUCTION` already says it, but thousands of tokens earlier, and a weaker
    #: model loses it: gemma-4 answered 9 of 12 English cross-lingual questions in the
    #: document's language, 12/12 with the reminder. qwen3-4b scores 1.000 without it, so
    #: this is off by default — a per-model trade, and every prompt change re-baselines.
    language_reminder: bool = False


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    ov_cache_dir: Path = Path(".ov_cache")


class Config(_Strict):
    models: ModelsConfig
    chunk: ChunkConfig
    retrieve: RetrieveConfig
    generate: GenerateConfig
    paths: PathsConfig = PathsConfig()

    def model_post_init(self, _: Any) -> None:
        if self.retrieve.n_context > self.retrieve.k:
            raise ValueError(
                f"n_context ({self.retrieve.n_context}) cannot exceed k ({self.retrieve.k})"
            )
        if self.chunk.overlap >= self.chunk.target_tokens:
            raise ValueError(
                f"overlap ({self.chunk.overlap}) must be < target_tokens "
                f"({self.chunk.target_tokens}) or chunking cannot advance"
            )
        if self.retrieve.rerank.enabled and self.models.reranker is None:
            raise ValueError(
                "retrieve.rerank.enabled is true but models.reranker is not configured — "
                "add it to configs/base.yaml and run `uv run python -m scripts.setup`"
            )

    @property
    def config_hash(self) -> str:
        return _hash_obj(self.model_dump(mode="json"))

    @property
    def chunk_config_hash(self) -> str:
        """Isolated so retuning `retrieve.tau` does not invalidate the chunk cache."""
        return _hash_obj(self.chunk.model_dump(mode="json"))

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path


def _hash_obj(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)


@lru_cache(maxsize=8)
def cached_config(path: str = str(DEFAULT_CONFIG)) -> Config:
    return load_config(path)
