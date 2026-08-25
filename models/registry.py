"""Model registry — load a converted model by name, with mandatory device fallback.

ARCHITECTURE.md §1 (`models/registry.py`), §3.1 (embedder fingerprint), §7.2 (`CACHE_DIR`),
§7.4 (GPU -> CPU fallback, log which device won).

Two contracts live here:

1. **Device fallback.** `configs/base.yaml` states *intent* (`device: GPU`). The machine
   decides. Every load tries the configured device, falls back to CPU, and reports the
   device it actually got on `LoadedModel.device` so `eval/bench.py` can be honest.
2. **Embedder fingerprint.** Query-time and index-time embeddings must come from a
   byte-identical model. `verify_fingerprint` raises rather than degrading quietly.

Paths inside `manifest.json` are relative to the manifest file's own directory, so a
manifest plus its `ir/` tree can be relocated (or built under a test's `tmp_path`) intact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openvino as ov
import structlog

from core.config import Config, ModelSpec

log = structlog.get_logger("models.registry")

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

CPU = "CPU"
IR_XML_NAME = "openvino_model.xml"
IR_BIN_NAME = "openvino_model.bin"

# §7.3 `hf_vlm` kind (models/convert.py:convert_vlm) exports through
# `OVModelForVisualCausalLM`, which writes a multi-part IR instead of the single
# `openvino_model.{xml,bin}` pair every other kind produces: the language tower, the text
# and vision embedding towers, and (for Gemma 4) the per-layer embeddings each get their own
# graph. `openvino_language_model.{xml,bin}` is the authoritative "is this converted / what's
# its fingerprint" pair for such an entry — detected from file presence, not a manifest field,
# so this module stays independent of convert.py's SOURCES/Kind table and no
# MANIFEST_SCHEMA_VERSION bump is needed.
VLM_LANGUAGE_MODEL_XML_NAME = "openvino_language_model.xml"
VLM_LANGUAGE_MODEL_BIN_NAME = "openvino_language_model.bin"

ROLES = (
    "ocr_det",
    "ocr_rec",
    "embedder",
    "generator",
    # Optional per-script recognition heads (§3). Unset in config means "no dedicated head".
    "ocr_rec_taml",
    "ocr_rec_latn",
    # §10 cross-encoder. Unset means the rerank arm cannot be switched on (core/config.py
    # refuses the combination), not that retrieval is broken.
    "reranker",
)

#: Roles a config may legitimately leave unset.
OPTIONAL_ROLES = frozenset({"ocr_rec_taml", "ocr_rec_latn", "reranker"})

#: §3.1 — the embedder block, in the order the architecture lists it.
EMBEDDER_FINGERPRINT_KEYS = (
    "hf_id",
    "hf_revision",
    "ir_sha256",
    "ov_version",
    "precision",
    "dim",
    "pooling",
    "normalize",
    "max_len",
    "query_prefix",
    "passage_prefix",
)


class RegistryError(RuntimeError):
    """Base for every registry failure. Always carries an actionable next step."""


class ModelNotFound(RegistryError):
    pass


class FingerprintMismatch(RegistryError):
    pass


@dataclass(frozen=True)
class EmbeddingSpec:
    """The half of the fingerprint that is not the weights themselves (§3.1 rule 3)."""

    dim: int
    pooling: str
    normalize: bool
    max_len: int
    query_prefix: str = ""
    passage_prefix: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> EmbeddingSpec:
        return cls(
            dim=int(row["dim"]),
            pooling=str(row["pooling"]),
            normalize=bool(row["normalize"]),
            max_len=int(row["max_len"]),
            query_prefix=str(row.get("query_prefix", "")),
            passage_prefix=str(row.get("passage_prefix", "")),
        )


@dataclass(frozen=True)
class ModelEntry:
    name: str
    role: str
    hf_id: str
    hf_revision: str
    precision: str
    ir_dir: Path
    ov_version: str
    converted_at: str
    recorded_ir_sha256: str
    embedding: EmbeddingSpec | None = None

    @property
    def ir_xml(self) -> Path:
        return self.ir_dir / IR_XML_NAME

    @property
    def ir_bin(self) -> Path:
        return self.ir_dir / IR_BIN_NAME

    @property
    def vlm_language_model_xml(self) -> Path:
        return self.ir_dir / VLM_LANGUAGE_MODEL_XML_NAME

    @property
    def vlm_language_model_bin(self) -> Path:
        return self.ir_dir / VLM_LANGUAGE_MODEL_BIN_NAME

    @property
    def is_vlm(self) -> bool:
        """Multi-part IR (§7.3 `hf_vlm` kind), detected by file presence on disk.

        `openvino_genai.VLMPipeline` is what `answer/generate.py:load_generator` builds for
        an entry that answers True here, in place of `LLMPipeline`.
        """
        return not self.ir_xml.exists() and self.vlm_language_model_xml.exists()

    @property
    def is_converted(self) -> bool:
        if self.ir_xml.exists() and self.ir_bin.exists():
            return True
        return self.vlm_language_model_xml.exists() and self.vlm_language_model_bin.exists()

    @classmethod
    def from_row(cls, name: str, row: Mapping[str, Any], base_dir: Path) -> ModelEntry:
        try:
            ir_dir = Path(row["ir_dir"])
            return cls(
                name=name,
                role=str(row["role"]),
                hf_id=str(row["hf_id"]),
                hf_revision=str(row["hf_revision"]),
                precision=str(row["precision"]),
                ir_dir=ir_dir if ir_dir.is_absolute() else (base_dir / ir_dir).resolve(),
                ov_version=str(row["ov_version"]),
                converted_at=str(row.get("converted_at", "")),
                recorded_ir_sha256=str(row.get("ir_sha256", "")),
                embedding=(
                    EmbeddingSpec.from_row(row["embedding"]) if row.get("embedding") else None
                ),
            )
        except KeyError as e:
            raise RegistryError(f"manifest entry {name!r} is missing required key {e}") from e


@dataclass(frozen=True)
class Manifest:
    path: Path
    ov_version: str
    entries: tuple[ModelEntry, ...]

    def by_name(self, name: str) -> ModelEntry:
        for e in self.entries:
            if e.name == name:
                return e
        known = ", ".join(sorted(e.name for e in self.entries)) or "<empty>"
        raise ModelNotFound(
            f"model {name!r} is not in {self.path} (have: {known}) — "
            f"run `uv run python -m scripts.setup` to download and convert it"
        )

    def by_role(self, role: str) -> ModelEntry:
        for e in self.entries:
            if e.role == role:
                return e
        raise ModelNotFound(f"no model with role {role!r} in {self.path}")


@dataclass(frozen=True)
class LoadedModel:
    """A compiled OpenVINO model plus the truth about where it ended up running."""

    name: str
    role: str
    entry: ModelEntry
    requested_device: str
    device: str
    compiled: ov.CompiledModel

    @property
    def fell_back(self) -> bool:
        return self.device != self.requested_device

    @property
    def precision(self) -> str:
        return self.entry.precision


def ov_version() -> str:
    """`ov.__version__` is `2026.3.0-22451-<sha>`; the fingerprint wants `2026.3.0`."""
    return str(ov.__version__).split("-", 1)[0]


def load_manifest(path: str | Path | None = None) -> Manifest:
    p = Path(path) if path is not None else MANIFEST_PATH
    if not p.exists():
        raise ModelNotFound(
            f"{p} not found — it is generated by models/convert.py (§7.3). "
            f"Run `uv run python -m scripts.setup` first."
        )
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise RegistryError(f"{p} is not valid JSON: {e}") from e

    version = int(raw.get("schema_version", 0))
    if version != MANIFEST_SCHEMA_VERSION:
        raise RegistryError(
            f"{p} has schema_version {version}, expected {MANIFEST_SCHEMA_VERSION} — "
            f"re-run models/convert.py"
        )

    base_dir = p.resolve().parent
    models = raw.get("models") or {}
    if not isinstance(models, dict):
        raise RegistryError(f"{p}: 'models' must be an object keyed by model name")
    return Manifest(
        path=p,
        ov_version=str(raw.get("ov_version", "")),
        entries=tuple(ModelEntry.from_row(k, v, base_dir) for k, v in sorted(models.items())),
    )


def spec_for(name: str, cfg: Config) -> tuple[str, ModelSpec]:
    """Resolve `name` — either a config role (`embedder`) or a model name (`bge-m3`)."""
    if name in ROLES:
        spec = getattr(cfg.models, name)
        if spec is None:
            raise ModelNotFound(
                f"role {name!r} is not configured in configs/base.yaml — it is optional, so "
                f"either add it or stop asking for it"
            )
        return name, spec
    for role in ROLES:
        spec = getattr(cfg.models, role)
        if spec is not None and spec.name == name:
            return role, spec
    raise ModelNotFound(
        f"{name!r} is neither a config role ({', '.join(ROLES)}) nor a model named in "
        f"configs/base.yaml"
    )


def select_device(requested: str, core: ov.Core | None = None) -> str:
    """§7.4 — the device we will actually get, without compiling anything.

    `openvino_genai` pipelines take a device string rather than a compiled model, so
    `answer/generate.py` needs the fallback decision separately from `load_model`.
    """
    core = core or ov.Core()
    available = set(core.available_devices)
    if requested in available or requested.startswith("AUTO") or requested.startswith("HETERO"):
        return requested
    log.warning(
        "model.device_unavailable",
        requested=requested,
        available=sorted(available),
        falling_back_to=CPU,
    )
    return CPU


def load_model(
    name: str,
    cfg: Config,
    *,
    manifest_path: str | Path | None = None,
    core: ov.Core | None = None,
) -> LoadedModel:
    """Compile a converted model, honouring config intent and falling back to CPU.

    `name` is a config role (`embedder`, `generator`, `ocr_det`, `ocr_rec`) or the model
    name that role points at (`bge-m3`).
    """
    role, spec = spec_for(name, cfg)
    manifest = load_manifest(manifest_path)
    entry = manifest.by_name(spec.name)

    if entry.precision != spec.precision:
        raise RegistryError(
            f"{entry.name}: config asks for {spec.precision} but manifest holds "
            f"{entry.precision} — re-run models/convert.py or fix configs/base.yaml"
        )
    if not entry.is_converted:
        raise ModelNotFound(
            f"{entry.name}: IR missing at {entry.ir_dir} (weights are not committed, §7.3) — "
            f"run `uv run python -m scripts.setup`"
        )
    if entry.ov_version and entry.ov_version != ov_version():
        log.warning(
            "model.ov_version_drift",
            model=entry.name,
            ir_built_with=entry.ov_version,
            runtime=ov_version(),
            hint="§7.1 — align openvino, openvino-tokenizers and openvino-genai before debugging",
        )

    compiled, device = _compile_with_fallback(
        entry=entry,
        requested=spec.device,
        cache_dir=cfg.resolve(cfg.paths.ov_cache_dir),
        core=core or ov.Core(),
    )
    return LoadedModel(
        name=entry.name,
        role=role,
        entry=entry,
        requested_device=spec.device,
        device=device,
        compiled=compiled,
    )


def _compile_with_fallback(
    entry: ModelEntry,
    requested: str,
    cache_dir: Path,
    core: ov.Core,
) -> tuple[ov.CompiledModel, str]:
    ov_config: dict[str, Any] = {}
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ov_config["CACHE_DIR"] = str(cache_dir)
    except OSError as e:
        log.warning("model.cache_dir_unwritable", cache_dir=str(cache_dir), error=str(e))

    available = set(core.available_devices)
    candidates: list[str] = []
    for dev in (requested, CPU):
        if dev not in candidates:
            candidates.append(dev)

    last_error: Exception | None = None
    for dev in candidates:
        if dev not in available and not dev.startswith(("AUTO", "HETERO")):
            log.warning(
                "model.device_unavailable",
                model=entry.name,
                requested=dev,
                available=sorted(available),
            )
            continue
        try:
            compiled = core.compile_model(str(entry.ir_xml), dev, ov_config)
        except RuntimeError as e:
            last_error = e
            log.warning("model.compile_failed", model=entry.name, device=dev, error=str(e))
            continue
        actual = _execution_device(compiled, dev)
        log.info(
            "model.loaded",
            model=entry.name,
            role=entry.role,
            precision=entry.precision,
            requested_device=requested,
            device=actual,
            fell_back=actual != requested,
            cached=bool(ov_config),
        )
        return compiled, actual

    raise RegistryError(
        f"{entry.name}: could not compile on any of {candidates} "
        f"(available: {sorted(available)}) — last error: {last_error}"
    )


def _execution_device(compiled: ov.CompiledModel, fallback: str) -> str:
    """`AUTO` hides the real device; ask the compiled model rather than guessing."""
    try:
        devices = compiled.get_property("EXECUTION_DEVICES")
    except (RuntimeError, KeyError, TypeError):
        return fallback
    if isinstance(devices, str):
        return devices
    if isinstance(devices, (list, tuple)) and devices:
        return ",".join(str(d) for d in devices)
    return fallback


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def ir_sha256(entry: ModelEntry) -> str:
    """§3.1 rule 4 — the weights are the identity, so hash them, don't trust the label.

    A multi-part VLM entry (§7.3 `hf_vlm`) has no `openvino_model.bin`; the language model
    part stands in as the weights that identify it, same as `is_converted`.

    Memoised on (path, size, mtime): re-hashing 2 GB of INT8 weights on every query would
    dominate retrieval latency, and any edit to the file moves mtime.
    """
    bin_path = entry.vlm_language_model_bin if entry.is_vlm else entry.ir_bin
    if not bin_path.exists():
        if entry.recorded_ir_sha256:
            log.warning(
                "model.ir_missing_using_recorded_sha",
                model=entry.name,
                ir_bin=str(bin_path),
                hint="weights are not committed (§7.3); run scripts/setup.py before indexing",
            )
            return entry.recorded_ir_sha256
        raise ModelNotFound(
            f"{entry.name}: cannot fingerprint — {bin_path} is missing and the manifest "
            f"records no ir_sha256"
        )
    stat = bin_path.stat()
    key = (str(bin_path), stat.st_size, stat.st_mtime_ns)
    cached = _SHA_CACHE.get(key)
    if cached is None:
        cached = file_sha256(bin_path)
        _SHA_CACHE[key] = cached
    return cached


def embedder_fingerprint(
    cfg: Config,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """The §3.1 `embedder` block, verbatim — write this into `index_manifest.json`.

    `ov_version` is the *runtime* OpenVINO version, not the one recorded at conversion:
    §3.1 rule 2 says the fingerprint is recomputed at query time, and §7.1 treats an
    OpenVINO version change as a reason to distrust a model artefact. An upgrade
    therefore forces a re-index rather than a silent quality drop.
    """
    manifest = load_manifest(manifest_path)
    entry = manifest.by_name(cfg.models.embedder.name)
    if entry.embedding is None:
        raise RegistryError(
            f"{entry.name}: manifest entry has no 'embedding' block — an embedder without "
            f"dim/pooling/normalize/max_len cannot be fingerprinted (§3.1)"
        )
    emb = entry.embedding
    return {
        "hf_id": entry.hf_id,
        "hf_revision": entry.hf_revision,
        "ir_sha256": ir_sha256(entry),
        "ov_version": ov_version(),
        "precision": entry.precision,
        "dim": emb.dim,
        "pooling": emb.pooling,
        "normalize": emb.normalize,
        "max_len": emb.max_len,
        "query_prefix": emb.query_prefix,
        "passage_prefix": emb.passage_prefix,
    }


def verify_fingerprint(
    index_manifest: Mapping[str, Any] | str | Path,
    cfg: Config,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hard-fail if the index was built with a different embedder (§3.1 rule 2).

    Accepts a loaded `index_manifest.json`, the bare `embedder` block, or a path to
    either. Returns the runtime fingerprint on success so callers can log it.
    """
    stored = _embedder_block(index_manifest)
    runtime = embedder_fingerprint(cfg, manifest_path=manifest_path)

    diffs = [
        f"  {key}: index={stored.get(key, '<missing>')!r} runtime={runtime[key]!r}"
        for key in EMBEDDER_FINGERPRINT_KEYS
        if stored.get(key) != runtime[key]
    ]
    if diffs:
        raise FingerprintMismatch(
            "index built with a different embedder — re-index required\n" + "\n".join(diffs)
        )
    log.info(
        "fingerprint.verified",
        hf_id=runtime["hf_id"],
        precision=runtime["precision"],
        ir_sha256=runtime["ir_sha256"][:12],
        ov_version=runtime["ov_version"],
    )
    return runtime


def _embedder_block(index_manifest: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(index_manifest, (str, Path)):
        p = Path(index_manifest)
        if not p.exists():
            raise ModelNotFound(f"index manifest not found: {p}")
        try:
            index_manifest = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise RegistryError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(index_manifest, Mapping):
        raise RegistryError("index manifest must be a mapping or a path to one")
    block = index_manifest.get("embedder", index_manifest)
    if not isinstance(block, Mapping):
        raise RegistryError("index manifest 'embedder' must be an object (§3.1)")
    if "ir_sha256" not in block:
        raise FingerprintMismatch(
            "index manifest carries no embedder fingerprint — re-index required (§3.1)"
        )
    return block
