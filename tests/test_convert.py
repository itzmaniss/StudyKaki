"""Conversion — ARCHITECTURE.md §3.1 (fingerprint), §7.3 (convert once, offline).

Most of this runs against fakes, because a real conversion downloads 2 GB and quantises for
minutes. The exception is deliberate: **every test that touched the tokenizer used a fake,
which is why a tokenizer IR that could not execute shipped green.** The gated tests at the
bottom load the artefact that actually sits in `models/ir/` and run text through it. Set
`INTEL2026_REAL_MODELS=1` to include them; without the gate the fast suite stays fast.

No test may touch `data/` or the network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

import models.convert as convert_mod
from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from models.convert import (
    DETOKENIZER_XML_NAME,
    TOKENIZER_XML_NAME,
    ConversionError,
    _save_ov_tokenizer,
    regenerate_tokenizer,
    source_for,
)
from models.registry import IR_BIN_NAME, IR_XML_NAME, MANIFEST_PATH, load_manifest

TEXTS = [
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "ஒளிச்சேர்க்கை என்பது ஒளி ஆற்றலை வேதி ஆற்றலாக மாற்றும் செயல்முறை ஆகும்.",
    "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖的过程。",
]


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=tmp_path / "ov_cache")
        }
    )


class FakeTokenizer:
    """Stands in for a `transformers` tokenizer: `_save_ov_tokenizer` only passes it on."""

    def __init__(self) -> None:
        self.saved_to: list[Path] = []

    def save_pretrained(self, path) -> None:
        self.saved_to.append(Path(path))


def record_saves(monkeypatch) -> list[dict]:
    """Capture every `ov.save_model` call `_save_ov_tokenizer` makes, without writing IR."""
    calls: list[dict] = []
    tok, detok = object(), object()

    def fake_convert_tokenizer(tokenizer, with_detokenizer=False, number_of_inputs=1):
        return (tok, detok) if with_detokenizer else tok

    def fake_save_model(model, path, compress_to_fp16=True, **kw):
        calls.append({"model": model, "path": Path(path), "compress_to_fp16": compress_to_fp16})

    import openvino_tokenizers

    monkeypatch.setattr(openvino_tokenizers, "convert_tokenizer", fake_convert_tokenizer)
    monkeypatch.setattr(convert_mod.ov, "save_model", fake_save_model)
    return calls


# --- the tokenizer must not be weight-compressed -------------------------------------


def test_tokenizer_ir_is_saved_uncompressed(tmp_path, monkeypatch):
    """A tokenizer's weights are a vocabulary table, and f16 rounds its unigram scores.

    `ov.save_model` defaults to `compress_to_fp16=True`. That default is right for encoder
    activations and wrong here — it changes tokenisation at the margin, and consumers that
    do not pin `INFERENCE_PRECISION_HINT` cannot read the f16 constants at all.
    """
    calls = record_saves(monkeypatch)
    _save_ov_tokenizer(FakeTokenizer(), tmp_path / "ir", with_detokenizer=False)

    assert [c["path"].name for c in calls] == [TOKENIZER_XML_NAME]
    assert calls[0]["compress_to_fp16"] is False


def test_detokenizer_ir_is_saved_uncompressed_too(tmp_path, monkeypatch):
    calls = record_saves(monkeypatch)
    _save_ov_tokenizer(FakeTokenizer(), tmp_path / "ir", with_detokenizer=True)

    assert [c["path"].name for c in calls] == [TOKENIZER_XML_NAME, DETOKENIZER_XML_NAME]
    assert all(c["compress_to_fp16"] is False for c in calls)


def test_tokenizer_save_is_skipped_not_fatal_when_conversion_fails(tmp_path, monkeypatch):
    """A tokenizer that will not convert must not take the model conversion down with it."""
    import openvino_tokenizers

    def boom(tokenizer, with_detokenizer=False):
        raise NotImplementedError("no converter for this tokenizer type")

    monkeypatch.setattr(openvino_tokenizers, "convert_tokenizer", boom)
    _save_ov_tokenizer(FakeTokenizer(), tmp_path / "ir", with_detokenizer=False)

    assert not (tmp_path / "ir" / TOKENIZER_XML_NAME).exists()


# --- regenerating the tokenizer must not disturb the weights --------------------------


def make_stub_ir(ir_dir: Path) -> Path:
    ir_dir.mkdir(parents=True, exist_ok=True)
    ids = ops.parameter([-1, -1], ov.Type.i64, name="input_ids")
    out = ops.convert(ids, ov.Type.f32)
    ov.save_model(ov.Model([out], [ids], "stub"), ir_dir / IR_XML_NAME, compress_to_fp16=False)
    return ir_dir


def test_regenerate_tokenizer_leaves_the_model_weights_byte_identical(tmp_path, cfg, monkeypatch):
    """§3.1 rule 4 — re-quantising would change `ir_sha256` and invalidate every index."""
    src = source_for(cfg.models.embedder.name)
    ir_dir = make_stub_ir(tmp_path / f"{src.name}-{cfg.models.embedder.precision}")
    before = (ir_dir / IR_BIN_NAME).read_bytes()

    fake = FakeTokenizer()
    monkeypatch.setattr(convert_mod, "snapshot_dir", lambda s, **kw: tmp_path / "snapshot")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", classmethod(lambda cls, *a, **kw: fake)
    )
    calls = record_saves(monkeypatch)

    out = regenerate_tokenizer("embedder", cfg, out_root=tmp_path)

    assert out == ir_dir
    assert (ir_dir / IR_BIN_NAME).read_bytes() == before
    assert [c["path"].name for c in calls] == [TOKENIZER_XML_NAME]
    assert fake.saved_to == [ir_dir]


def test_regenerate_tokenizer_refuses_when_there_is_no_ir_to_sit_beside(tmp_path, cfg):
    with pytest.raises(ConversionError, match="no converted IR"):
        regenerate_tokenizer("embedder", cfg, out_root=tmp_path)


# --- the artefact in models/ir/, when it is actually on disk --------------------------

real_models = pytest.mark.skipif(
    os.environ.get("INTEL2026_REAL_MODELS") != "1",
    reason="set INTEL2026_REAL_MODELS=1 to run against the converted IR in models/ir/ (§7.3)",
)


@pytest.fixture
def embedder_ir_dir():
    if not MANIFEST_PATH.exists():
        pytest.skip(f"{MANIFEST_PATH} not generated — run `uv run python -m scripts.setup`")
    entry = load_manifest(MANIFEST_PATH).by_role("embedder")
    if not (entry.ir_dir / TOKENIZER_XML_NAME).exists():
        pytest.skip(f"no tokenizer IR at {entry.ir_dir} — run `uv run python -m scripts.setup`")
    return entry.ir_dir


@real_models
def test_the_real_tokenizer_ir_holds_no_fp16_constants(embedder_ir_dir):
    """The regression guard for the artefact, not the code path that writes it.

    BGE-M3's Unigram op is a *reference* implementation that reads its vocab scores through
    an f32 pointer. Compressed weights make it unreadable, and the failure surfaces inside
    the embedder — three layers from the line that caused it.
    """
    xml = (embedder_ir_dir / TOKENIZER_XML_NAME).read_text()
    assert 'element_type="f16"' not in xml
    assert 'precision="FP16"' not in xml


@real_models
def test_the_real_tokenizer_ir_actually_executes(embedder_ir_dir):
    """Every other tokenizer test in this repo uses a fake. This one runs the artefact.

    A fake kept the suite green while `openvino_tokenizer.xml` could not tokenise a single
    string — the whole embedder was broken behind a passing test.
    """
    from ingest.embed import OVTokenizer

    tokenizer = OVTokenizer.from_ir_dir(embedder_ir_dir)
    ids, mask = tokenizer(TEXTS)

    assert ids.shape == mask.shape
    assert ids.shape[0] == len(TEXTS)
    assert ids.dtype == np.int64 and mask.dtype == np.int64
    assert set(np.unique(mask)) <= {0, 1}
    # Every text is non-empty, so every row must carry at least one real token; an all-pad
    # row is how a silently-broken tokenizer looks from the embedder's side.
    assert (mask.sum(axis=1) > 0).all()


@real_models
def test_the_real_tokenizer_ir_distinguishes_scripts(embedder_ir_dir):
    """Row alignment and per-text distinctness — Latin, Tamil and Han must not collide."""
    from ingest.embed import OVTokenizer

    tokenizer = OVTokenizer.from_ir_dir(embedder_ir_dir)
    ids, mask = tokenizer(TEXTS)

    kept = {tuple(row[m.astype(bool)].tolist()) for row, m in zip(ids, mask, strict=True)}
    assert len(kept) == len(TEXTS)


@real_models
def test_the_converted_weights_still_match_the_manifest_fingerprint():
    """§3.1 rule 4 — if `ir_sha256` drifted, every index built against it is stale."""
    from models.registry import ir_sha256

    entry = load_manifest(MANIFEST_PATH).by_role("embedder")
    if not entry.ir_bin.exists():
        pytest.skip(f"weights are not committed (§7.3); {entry.ir_bin} is absent")
    assert entry.recorded_ir_sha256, "manifest records no ir_sha256"
    assert ir_sha256(entry) == entry.recorded_ir_sha256


@real_models
def test_the_manifest_embedding_block_matches_the_declared_source():
    """A checkpoint swap that changed dim or pooling would silently break retrieval."""
    entry = load_manifest(MANIFEST_PATH).by_role("embedder")
    declared = source_for(entry.name).embedding
    assert declared is not None
    assert entry.embedding is not None
    assert entry.hf_revision == source_for(entry.name).hf_revision
    assert entry.embedding.dim == declared["dim"]
    assert entry.embedding.pooling == declared["pooling"]
    assert entry.embedding.normalize == declared["normalize"]
    assert entry.embedding.max_len == declared["max_len"]


@real_models
def test_the_manifest_on_disk_is_the_schema_the_registry_expects():
    raw = json.loads(MANIFEST_PATH.read_text())
    assert raw["generated_by"] == "models/convert.py"
    assert raw["models"], "manifest lists no models"
