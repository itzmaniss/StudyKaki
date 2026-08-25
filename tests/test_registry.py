"""Registry, device fallback, and embedder fingerprint — ARCHITECTURE.md §3.1, §7.2, §7.4.

Everything here runs against a **stub IR** built in `tmp_path`: a four-node OpenVINO model
with a 1024-wide constant. That is deliberate. The real BGE-M3 INT8 IR is 544 MB and is not
committed (§7.3), so a test that needed it would be a test that never runs on a clean
checkout. The stub exercises the same code path — compile, fall back, hash the weights,
compare the fingerprint — in under a second.

No test may touch `data/`; caches are redirected into `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from models.registry import (
    EMBEDDER_FINGERPRINT_KEYS,
    IR_BIN_NAME,
    IR_XML_NAME,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    VLM_LANGUAGE_MODEL_BIN_NAME,
    VLM_LANGUAGE_MODEL_XML_NAME,
    FingerprintMismatch,
    ModelNotFound,
    RegistryError,
    embedder_fingerprint,
    ir_sha256,
    load_manifest,
    load_model,
    ov_version,
    select_device,
    spec_for,
    verify_fingerprint,
)

EMBEDDING = {
    "dim": 1024,
    "pooling": "cls",
    "normalize": True,
    "max_len": 8192,
    "query_prefix": "",
    "passage_prefix": "",
}


def make_stub_ir(ir_dir: Path, fill: float = 0.5, dim: int = 1024) -> Path:
    """A compilable stand-in for an encoder: [batch, seq] int64 -> [batch, dim] float32."""
    ir_dir.mkdir(parents=True, exist_ok=True)
    ids = ops.parameter([-1, -1], ov.Type.i64, name="input_ids")
    as_float = ops.convert(ids, ov.Type.f32)
    pooled = ops.reduce_mean(as_float, ops.constant([1], ov.Type.i32), keep_dims=True)
    weights = ops.constant(np.full((1, dim), fill, dtype=np.float32))
    out = ops.matmul(pooled, weights, False, False)
    model = ov.Model([out], [ids], "stub-embedder")
    ov.save_model(model, ir_dir / IR_XML_NAME, compress_to_fp16=False)
    return ir_dir


def make_stub_vlm_ir(ir_dir: Path, fill: float = 0.5, dim: int = 8) -> Path:
    """A multi-part IR stand-in (§7.3 `hf_vlm`) — only the language-model part, since that
    is the only file `models/registry.py` reads. No `openvino_model.xml` is written, which
    is exactly what distinguishes a converted VLM entry from a converted single-file one.
    """
    ir_dir.mkdir(parents=True, exist_ok=True)
    ids = ops.parameter([-1, -1], ov.Type.i64, name="input_ids")
    as_float = ops.convert(ids, ov.Type.f32)
    pooled = ops.reduce_mean(as_float, ops.constant([1], ov.Type.i32), keep_dims=True)
    weights = ops.constant(np.full((1, dim), fill, dtype=np.float32))
    out = ops.matmul(pooled, weights, False, False)
    model = ov.Model([out], [ids], "stub-language-model")
    ov.save_model(model, ir_dir / VLM_LANGUAGE_MODEL_XML_NAME, compress_to_fp16=False)
    return ir_dir


def write_manifest(
    root: Path,
    *,
    name: str = "bge-m3",
    role: str = "embedder",
    precision: str = "int8",
    ir_dir: str = "ir/stub-int8",
    embedding: dict | None = None,
    schema_version: int = MANIFEST_SCHEMA_VERSION,
    build_ir: bool = True,
    fill: float = 0.5,
) -> Path:
    if build_ir:
        make_stub_ir(root / ir_dir, fill=fill)
    entry = {
        "role": role,
        "hf_id": "BAAI/bge-m3",
        "hf_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "precision": precision,
        "ir_dir": ir_dir,
        "ir_sha256": "",
        "ov_version": ov_version(),
        "converted_at": "2026-08-22T00:00:00Z",
    }
    if embedding is not None:
        entry["embedding"] = embedding
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "generated_by": "tests",
                "ov_version": ov_version(),
                "models": {name: entry},
            }
        )
    )
    return path


def write_vlm_manifest(
    root: Path,
    *,
    name: str = "gemma-4-e2b-it",
    ir_dir: str = "ir/gemma-4-e2b-it-int4",
    build_ir: bool = True,
    fill: float = 0.5,
) -> Path:
    """A manifest entry for the §7.3 `hf_vlm` multi-part IR — no `embedding` block, same as
    the generator role, and `ir_sha256: ""` because that is what `models/convert.py` records
    today for a kind whose weights it cannot hash with the single-file convention (BLOCKERS #16).
    """
    if build_ir:
        make_stub_vlm_ir(root / ir_dir, fill=fill)
    entry = {
        "role": "generator",
        "hf_id": "google/gemma-4-E2B-it",
        "hf_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "precision": "int4",
        "ir_dir": ir_dir,
        "ir_sha256": "",
        "ov_version": ov_version(),
        "converted_at": "2026-08-25T00:00:00Z",
    }
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "generated_by": "tests",
                "ov_version": ov_version(),
                "models": {name: entry},
            }
        )
    )
    return path


@pytest.fixture
def cfg(tmp_path):
    """Base config with caches redirected — nothing in tests may write to `data/`."""
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(
                data_dir=tmp_path / "data",
                ov_cache_dir=tmp_path / "ov_cache",
            )
        }
    )


@pytest.fixture
def manifest(tmp_path):
    return write_manifest(tmp_path, embedding=EMBEDDING)


# --- manifest parsing -------------------------------------------------------------


def test_load_manifest_resolves_ir_paths_relative_to_itself(manifest, tmp_path):
    entry = load_manifest(manifest).by_name("bge-m3")
    assert entry.ir_dir == (tmp_path / "ir" / "stub-int8").resolve()
    assert entry.is_converted


def test_manifest_tree_is_relocatable(manifest, tmp_path):
    """Paths are manifest-relative, so moving the tree must not break resolution."""
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "manifest.json").write_text(manifest.read_text())
    make_stub_ir(moved / "ir" / "stub-int8")
    entry = load_manifest(moved / "manifest.json").by_name("bge-m3")
    assert entry.ir_dir == (moved / "ir" / "stub-int8").resolve()
    assert entry.is_converted


# --- multi-part IR (§7.3 `hf_vlm`, BLOCKERS #16) -----------------------------------


def test_multi_part_ir_is_converted_and_flagged_vlm(tmp_path):
    path = write_vlm_manifest(tmp_path)
    entry = load_manifest(path).by_name("gemma-4-e2b-it")
    assert entry.is_vlm
    assert entry.is_converted


def test_single_file_ir_is_not_flagged_vlm(manifest):
    entry = load_manifest(manifest).by_name("bge-m3")
    assert not entry.is_vlm
    assert entry.is_converted


def test_neither_layout_present_is_not_converted(tmp_path):
    path = write_vlm_manifest(tmp_path, build_ir=False)
    entry = load_manifest(path).by_name("gemma-4-e2b-it")
    assert not entry.is_vlm
    assert not entry.is_converted


def test_vlm_language_model_xml_without_its_bin_is_not_converted(tmp_path):
    """A partial write (xml landed, bin did not) must not read as converted."""
    path = write_vlm_manifest(tmp_path, build_ir=False)
    entry = load_manifest(path).by_name("gemma-4-e2b-it")
    entry.ir_dir.mkdir(parents=True, exist_ok=True)
    entry.vlm_language_model_xml.write_text("<not-a-real-ir/>")
    assert not entry.vlm_language_model_bin.exists()
    assert not entry.is_converted


def test_registry_does_not_import_convert():
    """VLM detection is file-presence, on purpose (§7.3) — registry.py stays independent of
    convert.py's SOURCES/Kind table, which is offline conversion tooling, not a runtime dep.
    """
    import models.registry as registry

    source = Path(registry.__file__).read_text()
    assert "models.convert" not in source
    assert "from models import convert" not in source


def test_missing_manifest_names_the_setup_script(tmp_path):
    with pytest.raises(ModelNotFound, match="scripts.setup"):
        load_manifest(tmp_path / "nope.json")


def test_schema_version_mismatch_raises(tmp_path):
    path = write_manifest(tmp_path, schema_version=MANIFEST_SCHEMA_VERSION + 1)
    with pytest.raises(RegistryError, match="schema_version"):
        load_manifest(path)


def test_malformed_manifest_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    with pytest.raises(RegistryError, match="not valid JSON"):
        load_manifest(path)


def test_manifest_entry_missing_key_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "models": {"bge-m3": {"role": "embedder", "hf_id": "BAAI/bge-m3"}},
            }
        )
    )
    with pytest.raises(RegistryError, match="missing required key"):
        load_manifest(path)


def test_unknown_model_lists_what_is_available(manifest):
    with pytest.raises(ModelNotFound, match="have: bge-m3"):
        load_manifest(manifest).by_name("not-a-model")


# --- name resolution --------------------------------------------------------------


def test_spec_for_accepts_role_or_model_name(cfg):
    assert spec_for("embedder", cfg) == ("embedder", cfg.models.embedder)
    assert spec_for("bge-m3", cfg) == ("embedder", cfg.models.embedder)
    assert spec_for("generator", cfg)[0] == "generator"


def test_spec_for_rejects_unknown(cfg):
    with pytest.raises(ModelNotFound, match="neither a config role"):
        spec_for("llama-9000", cfg)


# --- device fallback (§7.4) -------------------------------------------------------


def test_load_falls_back_to_cpu_and_reports_the_device_it_got(cfg, manifest):
    """`base.yaml` asks for GPU as target intent; the load must report what it got."""
    loaded = load_model("embedder", cfg, manifest_path=manifest)
    assert loaded.device in set(ov.Core().available_devices)
    assert loaded.name == "bge-m3"
    assert loaded.role == "embedder"
    assert loaded.precision == "int8"
    if cfg.models.embedder.device not in ov.Core().available_devices:
        assert loaded.device == "CPU"
        assert loaded.fell_back
        assert loaded.requested_device == cfg.models.embedder.device


def test_loaded_model_is_actually_usable(cfg, manifest):
    loaded = load_model("bge-m3", cfg, manifest_path=manifest)
    out = loaded.compiled({"input_ids": np.array([[1, 2, 3]], dtype=np.int64)})
    assert out[loaded.compiled.output(0)].shape == (1, EMBEDDING["dim"])


def test_cache_dir_is_configured(cfg, manifest):
    """§7.2 — an uncached first compile is the slowest thing in the pipeline."""
    load_model("embedder", cfg, manifest_path=manifest)
    assert cfg.resolve(cfg.paths.ov_cache_dir).is_dir()


def test_unwritable_cache_dir_does_not_block_loading(cfg, manifest, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    stubborn = cfg.model_copy(
        update={"paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=blocked)}
    )
    assert load_model("embedder", stubborn, manifest_path=manifest).device


def test_select_device_falls_back_for_an_absent_device():
    assert select_device("NOSUCHDEVICE") == "CPU"


def test_select_device_keeps_an_available_device():
    assert select_device("CPU") == "CPU"


def test_missing_ir_points_at_setup(cfg, tmp_path):
    path = write_manifest(tmp_path, embedding=EMBEDDING, build_ir=False)
    with pytest.raises(ModelNotFound, match="scripts.setup"):
        load_model("embedder", cfg, manifest_path=path)


def test_precision_mismatch_refuses_to_load(cfg, tmp_path):
    """Config asking int8 while the manifest holds fp16 is a re-conversion, not a warning."""
    path = write_manifest(tmp_path, precision="fp16", embedding=EMBEDDING)
    with pytest.raises(RegistryError, match="config asks for int8"):
        load_model("embedder", cfg, manifest_path=path)


# --- fingerprint (§3.1) -----------------------------------------------------------


def test_fingerprint_has_exactly_the_architecture_fields(cfg, manifest):
    fp = embedder_fingerprint(cfg, manifest_path=manifest)
    assert list(fp) == list(EMBEDDER_FINGERPRINT_KEYS)
    assert fp["hf_id"] == "BAAI/bge-m3"
    assert fp["precision"] == "int8"
    assert fp["dim"] == 1024
    assert fp["pooling"] == "cls"
    assert fp["normalize"] is True
    assert fp["max_len"] == 8192
    assert fp["query_prefix"] == fp["passage_prefix"] == ""
    assert fp["ov_version"] == ov_version()


def test_hf_revision_is_a_commit_sha_not_a_branch(cfg, manifest):
    """§3.1 rule 1 — HF models are updated in place; a branch name pins nothing."""
    revision = embedder_fingerprint(cfg, manifest_path=manifest)["hf_revision"]
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)


def test_ir_sha256_is_the_hash_of_the_weights_on_disk(cfg, manifest, tmp_path):
    import hashlib

    fp = embedder_fingerprint(cfg, manifest_path=manifest)
    weights = tmp_path / "ir" / "stub-int8" / IR_BIN_NAME
    assert weights.stat().st_size > 0
    assert fp["ir_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()


def test_different_weights_are_a_different_model(tmp_path):
    """§3.1 rule 4 — two quantizations of one checkpoint are not the same model."""
    a = load_manifest(write_manifest(tmp_path / "a", embedding=EMBEDDING, fill=0.5))
    b = load_manifest(write_manifest(tmp_path / "b", embedding=EMBEDDING, fill=0.25))
    assert ir_sha256(a.by_name("bge-m3")) != ir_sha256(b.by_name("bge-m3"))


def test_fingerprint_without_an_embedding_block_raises(cfg, tmp_path):
    path = write_manifest(tmp_path, embedding=None)
    with pytest.raises(RegistryError, match="no 'embedding' block"):
        embedder_fingerprint(cfg, manifest_path=path)


def test_ir_sha256_falls_back_to_the_recorded_hash_when_weights_are_absent(tmp_path):
    path = write_manifest(tmp_path, embedding=EMBEDDING, build_ir=False)
    raw = json.loads(path.read_text())
    raw["models"]["bge-m3"]["ir_sha256"] = "de" * 32
    path.write_text(json.dumps(raw))
    assert ir_sha256(load_manifest(path).by_name("bge-m3")) == "de" * 32


def test_ir_sha256_raises_when_there_is_nothing_to_hash(tmp_path):
    path = write_manifest(tmp_path, embedding=EMBEDDING, build_ir=False)
    with pytest.raises(ModelNotFound, match="cannot fingerprint"):
        ir_sha256(load_manifest(path).by_name("bge-m3"))


def test_ir_sha256_hashes_the_language_model_part_for_a_vlm_entry(tmp_path):
    import hashlib

    path = write_vlm_manifest(tmp_path)
    entry = load_manifest(path).by_name("gemma-4-e2b-it")
    weights = tmp_path / "ir" / "gemma-4-e2b-it-int4" / VLM_LANGUAGE_MODEL_BIN_NAME
    assert weights.stat().st_size > 0
    assert ir_sha256(entry) == hashlib.sha256(weights.read_bytes()).hexdigest()


def test_ir_sha256_for_a_vlm_entry_falls_back_when_the_language_model_is_absent(tmp_path):
    path = write_vlm_manifest(tmp_path, build_ir=False)
    raw = json.loads(path.read_text())
    raw["models"]["gemma-4-e2b-it"]["ir_sha256"] = "ab" * 32
    path.write_text(json.dumps(raw))
    assert ir_sha256(load_manifest(path).by_name("gemma-4-e2b-it")) == "ab" * 32


# --- verify_fingerprint: the one that must raise ----------------------------------


def index_manifest_for(cfg, manifest) -> dict:
    return {
        "index_id": "sha256:abc",
        "n_vectors": 12,
        "chunk_config_hash": cfg.chunk_config_hash,
        "embedder": embedder_fingerprint(cfg, manifest_path=manifest),
    }


def test_matching_fingerprint_verifies(cfg, manifest):
    stored = index_manifest_for(cfg, manifest)
    assert verify_fingerprint(stored, cfg, manifest_path=manifest) == stored["embedder"]


def test_verify_accepts_a_bare_embedder_block(cfg, manifest):
    block = index_manifest_for(cfg, manifest)["embedder"]
    assert verify_fingerprint(block, cfg, manifest_path=manifest) == block


def test_verify_accepts_a_path_to_index_manifest(cfg, manifest, tmp_path):
    path = tmp_path / "index_manifest.json"
    path.write_text(json.dumps(index_manifest_for(cfg, manifest)))
    assert verify_fingerprint(path, cfg, manifest_path=manifest)


@pytest.mark.parametrize("field", EMBEDDER_FINGERPRINT_KEYS)
def test_a_mismatched_embedder_raises_rather_than_returning_results(cfg, manifest, field):
    """CLAUDE.md testing expectations + §3.1 rule 2 — hard fail, never degrade quietly.

    Every field of the fingerprint is load-bearing: a changed `max_len` truncates
    differently, a changed `query_prefix` shifts the whole query embedding.
    """
    stored = index_manifest_for(cfg, manifest)
    current = stored["embedder"][field]
    stored["embedder"][field] = (
        (not current)
        if isinstance(current, bool)
        else (current + 1)
        if isinstance(current, int)
        else f"{current}-drifted"
    )

    with pytest.raises(FingerprintMismatch) as excinfo:
        verify_fingerprint(stored, cfg, manifest_path=manifest)
    message = str(excinfo.value)
    assert "re-index required" in message
    assert field in message


def test_missing_field_is_a_mismatch(cfg, manifest):
    stored = index_manifest_for(cfg, manifest)
    del stored["embedder"]["pooling"]
    with pytest.raises(FingerprintMismatch, match="re-index required"):
        verify_fingerprint(stored, cfg, manifest_path=manifest)


def test_an_index_with_no_fingerprint_at_all_raises(cfg, manifest):
    with pytest.raises(FingerprintMismatch, match="no embedder fingerprint"):
        verify_fingerprint({"index_id": "sha256:abc", "n_vectors": 3}, cfg, manifest_path=manifest)


def test_verify_rejects_a_missing_index_manifest(cfg, manifest, tmp_path):
    with pytest.raises(ModelNotFound, match="index manifest not found"):
        verify_fingerprint(tmp_path / "gone.json", cfg, manifest_path=manifest)


# --- the committed manifest -------------------------------------------------------


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="models/manifest.json not generated yet")
def test_committed_manifest_is_loadable_and_matches_config():
    """`manifest.json` is committed (§7.3); the IR beside it is not, so only parse here."""
    cfg = load_config(DEFAULT_CONFIG)
    manifest = load_manifest()
    entry = manifest.by_name(cfg.models.embedder.name)
    assert entry.role == "embedder"
    assert entry.precision == cfg.models.embedder.precision
    assert entry.embedding is not None
    assert entry.embedding.dim == 1024
    assert len(entry.hf_revision) == 40
    assert entry.ir_xml.name == IR_XML_NAME
