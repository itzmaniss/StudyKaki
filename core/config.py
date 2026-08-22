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


class ChunkConfig(_Strict):
    target_tokens: int = Field(gt=0)
    overlap: int = Field(ge=0)
    min_tokens: int = Field(ge=0)


class RetrieveConfig(_Strict):
    k: int = Field(gt=0)
    n_context: int = Field(gt=0)
    tau: float = Field(ge=0.0, le=1.0)


class GenerateConfig(_Strict):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)


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
    raw = yaml.safe_load(p.read_text()) or {}
    return Config.model_validate(raw)


@lru_cache(maxsize=8)
def cached_config(path: str = str(DEFAULT_CONFIG)) -> Config:
    return load_config(path)
