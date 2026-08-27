"""HF / Paddle checkpoint -> OpenVINO IR -> INT8/INT4, and write `models/manifest.json`.

ARCHITECTURE.md §1, §3.1, §7.3. No network: checkpoints come from the local HF cache
(`local_files_only=True`), and `scripts/setup.py` is the only downloader (§0.3).

Uses OpenVINO's PyTorch frontend, not optimum's ONNX exporter — the latter imports private
`torch.onnx.symbolic_opset14` symbols torch 2.13 removed, so `import optimum.intel` raises
here. Stateful causal-LM export needs optimum anyway; see `convert_causal_lm`.

    uv run python -m models.convert --only embedder
    uv run python -m models.convert --all
    uv run python -m models.convert --only embedder --tokenizer-only
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import openvino as ov
import structlog

from core.config import Config, load_config
from models.registry import (
    IR_BIN_NAME,
    IR_XML_NAME,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    ROLES,
    VLM_LANGUAGE_MODEL_BIN_NAME,
    VLM_LANGUAGE_MODEL_XML_NAME,
    file_sha256,
    ov_version,
)

log = structlog.get_logger("models.convert")

IR_ROOT = Path(__file__).resolve().parent / "ir"

TOKENIZER_XML_NAME = "openvino_tokenizer.xml"
DETOKENIZER_XML_NAME = "openvino_detokenizer.xml"

# Covers every op PP-OCR mobile uses, and OpenVINO's ONNX frontend reads it as-is.
PADDLE_ONNX_OPSET = 14

Kind = Literal["hf_encoder", "hf_causal_lm", "hf_vlm", "hf_reranker", "paddle"]

#: Kinds that emit text, so they need a detokenizer. Encoders/rerankers only consume.
_CHAT_KINDS: frozenset[str] = frozenset({"hf_causal_lm", "hf_vlm"})


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSource:
    """Where a model in `configs/base.yaml` comes from.

    `hf_revision` is a commit SHA, never a branch: HF updates in place, and a branch would
    let weights change under a fingerprint that says they didn't (§3.1 rule 1).
    """

    name: str
    role: str
    kind: Kind
    hf_id: str
    hf_revision: str
    #: §3.1 fingerprint fields, cross-checked against the checkpoint at conversion.
    embedding: dict[str, Any] | None = None
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = field(default_factory=tuple)


SOURCES: dict[str, ModelSource] = {
    "bge-m3": ModelSource(
        name="bge-m3",
        role="embedder",
        kind="hf_encoder",
        hf_id="BAAI/bge-m3",
        hf_revision="5617a9f61b028005a4858fdac845db406aefb181",
        embedding={
            "dim": 1024,
            "pooling": "cls",
            "normalize": True,
            "max_len": 8192,
            # BGE-M3 is symmetric; E5-family needs prefixes. Travels with the index (§3.1 rule 3).
            "query_prefix": "",
            "passage_prefix": "",
        },
        ignore_patterns=("onnx/*", "*.onnx", "*.onnx_data", "imgs/*", "*.msgpack", "*.h5"),
    ),
    "bge-reranker-v2-m3": ModelSource(
        name="bge-reranker-v2-m3",
        role="reranker",
        kind="hf_reranker",
        hf_id="BAAI/bge-reranker-v2-m3",
        hf_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        # No `embedding` block: a logit, not a vector, so no fingerprint to invalidate.
        ignore_patterns=("onnx/*", "*.onnx", "*.onnx_data", "*.msgpack", "*.h5"),
    ),
    "qwen3-4b-instruct": ModelSource(
        name="qwen3-4b-instruct",
        role="generator",
        kind="hf_causal_lm",
        hf_id="Qwen/Qwen3-4B-Instruct-2507",
        hf_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        ignore_patterns=("original/*", "*.gguf"),
    ),
    # Smaller generator, against BLOCKERS #14 (memory) and the Tamil failure. ~5B raw
    # weights running as ~2B effective, so it converts like 5B and runs like 2B.
    "gemma-4-e2b-it": ModelSource(
        name="gemma-4-e2b-it",
        role="generator",
        kind="hf_vlm",
        hf_id="google/gemma-4-E2B-it",
        hf_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        # Vision/audio towers live inside model.safetensors — nothing to exclude.
        ignore_patterns=("*.gguf", "*.task", "*.litertlm"),
    ),
    # base.yaml names v6, but PaddlePaddle/PP-OCRv6_* 401s on HF — no public checkpoint.
    # v5 mobile is the newest available and what PaddleOCR ships. See BLOCKERS.md.
    "PP-OCRv5_mobile_det": ModelSource(
        name="PP-OCRv5_mobile_det",
        role="ocr_det",
        kind="paddle",
        hf_id="PaddlePaddle/PP-OCRv5_mobile_det",
        hf_revision="0d63e78e2b680928f6b1747d76a08db6e645efb7",
    ),
    "PP-OCRv5_mobile_rec": ModelSource(
        name="PP-OCRv5_mobile_rec",
        role="ocr_rec",
        kind="paddle",
        hf_id="PaddlePaddle/PP-OCRv5_mobile_rec",
        hf_revision="682f20538d8c086cb2128e5cfac775e6c4904e85",
    ),
    # §3 per-script heads. v5's default charset (18385 classes, zh+en) has no Tamil, so a
    # Tamil page decodes to CJK noise at high confidence.
    "ta_PP-OCRv5_mobile_rec": ModelSource(
        name="ta_PP-OCRv5_mobile_rec",
        role="ocr_rec_taml",
        kind="paddle",
        hf_id="PaddlePaddle/ta_PP-OCRv5_mobile_rec",
        hf_revision="1bb164dad1d8eb23c7f7a382827e5305b37868d4",
    ),
    "latin_PP-OCRv5_mobile_rec": ModelSource(
        name="latin_PP-OCRv5_mobile_rec",
        role="ocr_rec_latn",
        kind="paddle",
        hf_id="PaddlePaddle/latin_PP-OCRv5_mobile_rec",
        hf_revision="ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57",
    ),
}


def source_for(name: str) -> ModelSource:
    src = SOURCES.get(name)
    if src is None:
        raise ConversionError(
            f"no conversion source registered for model {name!r} — known: "
            f"{', '.join(sorted(SOURCES))}. Add it to models/convert.py SOURCES, or fix "
            f"the name in configs/base.yaml."
        )
    return src


def snapshot_dir(src: ModelSource, *, allow_download: bool = False) -> Path:
    """Locate the checkpoint. Offline by default — downloading is `scripts/setup.py`'s job."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(
            snapshot_download(
                src.hf_id,
                revision=src.hf_revision,
                local_files_only=not allow_download,
                ignore_patterns=list(src.ignore_patterns) or None,
                allow_patterns=list(src.allow_patterns) or None,
            )
        )
    except (LocalEntryNotFoundError, FileNotFoundError, OSError) as e:
        if allow_download:
            raise ConversionError(f"could not fetch {src.hf_id}@{src.hf_revision[:8]}: {e}") from e
        raise ConversionError(
            f"{src.hf_id}@{src.hf_revision[:8]} is not in the local HF cache and "
            f"models/convert.py is offline by design (§0.3) — run "
            f"`uv run python -m scripts.setup` first"
        ) from e


def _ir_dir(src: ModelSource, precision: str, out_root: Path) -> Path:
    return out_root / f"{src.name}-{precision}"


def _compress(model: ov.Model, precision: str, name: str) -> ov.Model:
    if precision in ("fp32", "fp16"):
        return model
    import nncf

    modes = {"int8": nncf.CompressWeightsMode.INT8_ASYM, "int4": nncf.CompressWeightsMode.INT4_SYM}
    mode = modes.get(precision)
    if mode is None:
        raise ConversionError(f"{name}: unsupported precision {precision!r}")
    log.info("convert.compress_weights", model=name, precision=precision)
    kwargs: dict[str, Any] = {"mode": mode}
    if precision == "int4":
        kwargs.update(group_size=128, ratio=0.8)
    return nncf.compress_weights(model, **kwargs)


def _save(model: ov.Model, ir_dir: Path, precision: str) -> None:
    ir_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, ir_dir / IR_XML_NAME, compress_to_fp16=precision != "fp32")


def _checked_embedding(src: ModelSource, snapshot: Path) -> dict[str, Any]:
    """Derive dim/pooling/max_len from the checkpoint; refuse to disagree with SOURCES.

    Fails the conversion rather than the retrieval — this is the drift §3.1 exists for.
    """
    declared = dict(src.embedding or {})
    if not declared:
        return {}

    found: dict[str, Any] = {}
    cfg_json = snapshot / "config.json"
    if cfg_json.exists():
        hidden = json.loads(cfg_json.read_text()).get("hidden_size")
        if hidden:
            found["dim"] = int(hidden)
    sbert = snapshot / "sentence_bert_config.json"
    if sbert.exists():
        max_len = json.loads(sbert.read_text()).get("max_seq_length")
        if max_len:
            found["max_len"] = int(max_len)
    pooling_json = snapshot / "1_Pooling" / "config.json"
    if pooling_json.exists():
        pool = json.loads(pooling_json.read_text())
        if pool.get("pooling_mode_cls_token"):
            found["pooling"] = "cls"
        elif pool.get("pooling_mode_mean_tokens"):
            found["pooling"] = "mean"

    drift = {k: (declared[k], v) for k, v in found.items() if declared.get(k) != v}
    if drift:
        raise ConversionError(
            f"{src.name}: checkpoint disagrees with the declared fingerprint (§3.1) — "
            + "; ".join(f"{k}: declared={d!r} checkpoint={f!r}" for k, (d, f) in drift.items())
            + ". Fix models/convert.py SOURCES and re-index everything."
        )
    return declared


def convert_encoder(src: ModelSource, precision: str, out_root: Path) -> Path:
    """BGE-M3 and friends, via OpenVINO's PyTorch frontend (no ONNX round-trip)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    snapshot = snapshot_dir(src)
    ir_dir = _ir_dir(src, precision, out_root)

    log.info("convert.load_checkpoint", model=src.name, snapshot=str(snapshot))
    model = AutoModel.from_pretrained(snapshot, torch_dtype=torch.float32)
    model.eval()

    example = {
        "input_ids": torch.ones(1, 16, dtype=torch.int64),
        "attention_mask": torch.ones(1, 16, dtype=torch.int64),
    }
    with torch.no_grad():
        ov_model = ov.convert_model(
            model,
            example_input=example,
            input=[
                ("input_ids", ov.PartialShape([-1, -1]), ov.Type.i64),
                ("attention_mask", ov.PartialShape([-1, -1]), ov.Type.i64),
            ],
        )
    _name_tensors(
        ov_model,
        inputs=["input_ids", "attention_mask"],
        outputs=["last_hidden_state", "pooler_output"],
    )
    ov_model = _compress(ov_model, precision, src.name)
    _save(ov_model, ir_dir, precision)

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.save_pretrained(ir_dir)
    _save_ov_tokenizer(tokenizer, ir_dir, with_detokenizer=False)
    return ir_dir


def convert_reranker(src: ModelSource, precision: str, out_root: Path) -> Path:
    """Cross-encoder for GenAI's `TextRerankPipeline` (§10).

    Via optimum-intel, not the PyTorch frontend: the pipeline expects optimum's
    sequence-classification layout, and a hand-rolled graph does not load.
    """
    try:
        from optimum.intel import OVModelForSequenceClassification, OVWeightQuantizationConfig
    except ImportError as e:
        raise ConversionError(
            f"{src.name}: optimum-intel cannot be imported ({e}) — see convert_causal_lm "
            f"for the torch pin this usually needs."
        ) from e

    if precision not in ("int8", "fp32", "fp16"):
        raise ConversionError(f"{src.name}: unsupported reranker precision {precision!r}")

    snapshot = snapshot_dir(src)
    ir_dir = _ir_dir(src, precision, out_root)

    log.info("convert.export_reranker", model=src.name, precision=precision)
    # Quantize during export: post-hoc compress_weights rewrites the .xml/.bin in place,
    # dies mid-write, and leaves a model the pipeline cannot load.
    quant = OVWeightQuantizationConfig(bits=8, sym=False) if precision == "int8" else None
    ov_model = OVModelForSequenceClassification.from_pretrained(
        snapshot, export=True, quantization_config=quant
    )
    ir_dir.mkdir(parents=True, exist_ok=True)
    ov_model.save_pretrained(ir_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.save_pretrained(ir_dir)
    # Two inputs: a cross-encoder scores (query, passage), not one sequence.
    _save_ov_tokenizer(tokenizer, ir_dir, with_detokenizer=False, number_of_inputs=2)
    return ir_dir


def _name_tensors(model: ov.Model, inputs: list[str], outputs: list[str]) -> None:
    """The PyTorch frontend names tensors after graph node ids (`43`, `1893`).

    Callers address tensors by name, and an unnamed `last_hidden_state` invites reading
    `pooler_output` by mistake — silent retrieval degradation.
    """
    for port, name in zip(model.inputs, inputs, strict=False):
        port.get_tensor().set_names({name})
    for port, name in zip(model.outputs, outputs, strict=False):
        port.get_tensor().set_names({name})


def convert_causal_lm(src: ModelSource, precision: str, out_root: Path) -> Path:
    """Qwen3-4B INT4 — needs optimum-intel's stateful export for `LLMPipeline` KV cache.

    Fails loudly if optimum cannot import, rather than shipping an IR that runs 10x slow.
    """
    try:
        from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
    except ImportError as e:
        raise ConversionError(
            f"{src.name}: optimum-intel cannot be imported, so the stateful causal-LM "
            f"export is unavailable ({e}). optimum 1.27 imports private symbols that "
            f"torch 2.13 removed from torch.onnx.symbolic_opset14. Fix by pinning torch "
            f"below 2.13 (`uv add 'torch<2.13'`) — do not hand-roll the KV-cache export."
        ) from e

    snapshot = snapshot_dir(src)
    ir_dir = _ir_dir(src, precision, out_root)
    bits = {"int4": 4, "int8": 8}.get(precision)
    if bits is None:
        raise ConversionError(f"{src.name}: unsupported generator precision {precision!r}")

    log.info("convert.export_causal_lm", model=src.name, precision=precision)
    quant = OVWeightQuantizationConfig(bits=bits, group_size=128, ratio=0.8, sym=True)
    ov_model = OVModelForCausalLM.from_pretrained(
        snapshot, export=True, stateful=True, quantization_config=quant
    )
    ir_dir.mkdir(parents=True, exist_ok=True)
    ov_model.save_pretrained(ir_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.save_pretrained(ir_dir)
    _save_ov_tokenizer(tokenizer, ir_dir, with_detokenizer=True)
    return ir_dir


def convert_vlm(src: ModelSource, precision: str, out_root: Path) -> Path:
    """Gemma 4 E2B — one checkpoint, several IRs, loaded together by `VLMPipeline`.

    The text tower does not stand alone: the per-layer embeddings that make ~5B run as ~2B
    are a separate graph the pipeline feeds in. `OVModelForCausalLM` would export the
    language model with no way to receive them, so this takes the image-text-to-text path
    even though the project only sends text.
    """
    # OVModelForImageTextToText infers `image-to-text-with-past`, which the gemma4 exporter
    # rejects. This class's export_feature is the `image-text-to-text` optimum registered.
    from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig

    snapshot = snapshot_dir(src)
    ir_dir = _ir_dir(src, precision, out_root)
    bits = {"int4": 4, "int8": 8, "fp16": None, "fp32": None}
    if precision not in bits:
        raise ConversionError(f"{src.name}: unsupported generator precision {precision!r}")
    bits = bits[precision]  # type: ignore[assignment]

    log.info("convert.export_vlm", model=src.name, precision=precision)
    # 8-bit takes no mixed-precision knobs: `ratio` is the INT4 share, so optimum-intel 2.1
    # requires 1.0 and group_size=-1. 4-bit matches convert_causal_lm, so the two generators
    # compress alike and the comparison is about the models.
    # No compression at fp16/fp32: NNCF computes scales by running OpenVINO ops per weight,
    # and on arm64 the reduce hits an unimplemented CPU executor (`ReduceMin`, at 4 and 8
    # bits alike). Uncompressed is the only export that completes here, at ~10 GB RAM — a
    # way to evaluate the model on this machine, not to ship it.
    quant = None
    if bits == 4:
        quant = OVWeightQuantizationConfig(bits=4, group_size=128, ratio=0.8, sym=True)
    elif bits == 8:
        quant = OVWeightQuantizationConfig(bits=8, group_size=-1, ratio=1.0, sym=True)
    ov_model = OVModelForVisualCausalLM.from_pretrained(
        snapshot, export=True, quantization_config=quant
    )
    ir_dir.mkdir(parents=True, exist_ok=True)
    ov_model.save_pretrained(ir_dir)

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    _fix_chat_template_for_minja(tokenizer, model=src.name)
    tokenizer.save_pretrained(ir_dir)
    _save_ov_tokenizer(tokenizer, ir_dir, with_detokenizer=True)
    # VLMPipeline needs the processor config beside the IR, or it fails at load.
    try:
        processor = AutoProcessor.from_pretrained(snapshot)
        _fix_chat_template_for_minja(processor, model=src.name)
        processor.save_pretrained(ir_dir)
    except (OSError, ValueError) as e:
        log.warning("convert.processor_unavailable", model=src.name, error=str(e))
    return ir_dir


_ADJACENT_STRING_LITERALS = re.compile(r'("(?:[^"\\]|\\.)*")(\s+)("(?:[^"\\]|\\.)*")')


def _fix_chat_template_for_minja(carrier: Any, *, model: str) -> None:
    """Collapse Jinja's adjacent-string-literal concatenation on `carrier.chat_template`.

    minja (the engine openvino_genai uses) does not support `"a" "b"` == `"a" + "b"` and
    fails the whole template with `Expected closing parenthesis`. Gemma 4 hits this in a
    tool-calling branch that is dead here, but minja parses up front, so it is fatal.

    Must run on the in-memory template *before* save: `convert_tokenizer` bakes it into the
    tokenizer IR's rt_info, and `VLMPipeline` reads the baked copy, not the loose file.

    Regex rather than a literal, so Google editing the message text does not silently
    break the match; logs when it changes something.
    """
    template = getattr(carrier, "chat_template", None)
    if not template:
        return
    patched = template
    while True:
        merged = _ADJACENT_STRING_LITERALS.sub(
            lambda m: f'"{m.group(1)[1:-1]}{m.group(3)[1:-1]}"', patched
        )
        if merged == patched:
            break
        patched = merged
    if patched != template:
        carrier.chat_template = patched
        log.info("convert.chat_template_patched", model=model, carrier=type(carrier).__name__)


def _is_pir(model_file: Path) -> bool:
    """PaddlePaddle 3.0 replaced the protobuf `.pdmodel` with a JSON PIR program.

    OpenVINO's Paddle frontend reads legacy protobuf only, and fails PIR with "Cannot
    recognize input model." Every PaddlePaddle HF repo now ships PIR, so this is the norm.
    """
    if model_file.suffix != ".json":
        return False
    with model_file.open("rb") as fh:
        return b'"magic":"pir"' in fh.read(256).replace(b" ", b"")


def _pir_to_onnx(model_file: Path, params_file: Path, dest: Path) -> Path:
    """PIR -> ONNX via Paddle's own exporter, so provenance stays with PaddlePaddle.

    §3.1 makes model identity a hard requirement, so an unvetted community ONNX re-upload
    loses to the official converter — even though it drags in the paddle runtime.
    """
    try:
        import paddle2onnx
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConversionError(
            "paddle2onnx is required to convert PIR-format Paddle models; run `uv sync`"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx.export(
        model_filename=str(model_file),
        params_filename=str(params_file),
        save_file=str(dest),
        opset_version=PADDLE_ONNX_OPSET,
        auto_upgrade_opset=True,
    )
    if not dest.exists():
        raise ConversionError(f"paddle2onnx reported success but {dest} was not written")
    return dest


def convert_paddle(src: ModelSource, precision: str, out_root: Path) -> Path:
    """PaddleOCR mobile det/rec. PIR models route through ONNX; legacy ones go direct."""
    snapshot = snapshot_dir(src)
    candidates = [snapshot / "inference.json", snapshot / "inference.pdmodel"]
    model_file = next((c for c in candidates if c.exists()), None)
    if model_file is None:
        raise ConversionError(
            f"{src.name}: no Paddle inference model in {snapshot} "
            f"(looked for {', '.join(c.name for c in candidates)})"
        )

    if _is_pir(model_file):
        params = snapshot / "inference.pdiparams"
        if not params.exists():
            raise ConversionError(f"{src.name}: PIR model at {model_file} has no {params.name}")
        onnx_path = _ir_dir(src, precision, out_root) / f"{src.name}.onnx"
        log.info("convert.paddle_pir", model=src.name, source=str(model_file), via="onnx")
        model_file = _pir_to_onnx(model_file, params, onnx_path)

    log.info("convert.paddle", model=src.name, source=str(model_file))
    ov_model = ov.convert_model(model_file)
    ov_model = _compress(ov_model, precision, src.name)
    ir_dir = _ir_dir(src, precision, out_root)
    _save(ov_model, ir_dir, precision)
    for extra in ("inference.yml", "config.json"):
        p = snapshot / extra
        if p.exists():
            (ir_dir / extra).write_bytes(p.read_bytes())
    return ir_dir


def _save_ov_tokenizer(
    tokenizer: Any, ir_dir: Path, *, with_detokenizer: bool, number_of_inputs: int = 1
) -> None:
    """§0.4 — every model runs through OpenVINO, tokenisers included.

    Saved uncompressed: `save_model` defaults to fp16, but a tokenizer's weights are a
    vocab table, and BGE-M3's Unigram op is a reference implementation that reads scores as
    f32. Compressed it dies at infer time, three layers from its cause.
    """
    try:
        from openvino_tokenizers import convert_tokenizer
    except ImportError as e:
        log.warning("convert.ov_tokenizer_unavailable", ir_dir=str(ir_dir), error=str(e))
        return
    try:
        converted = convert_tokenizer(
            tokenizer, with_detokenizer=with_detokenizer, number_of_inputs=number_of_inputs
        )
    except (RuntimeError, TypeError, NotImplementedError, OSError) as e:
        log.warning("convert.ov_tokenizer_failed", ir_dir=str(ir_dir), error=str(e))
        return
    tok, detok = converted if isinstance(converted, tuple) else (converted, None)
    ir_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(tok, ir_dir / TOKENIZER_XML_NAME, compress_to_fp16=False)
    if detok is not None:
        ov.save_model(detok, ir_dir / DETOKENIZER_XML_NAME, compress_to_fp16=False)
    log.info(
        "convert.ov_tokenizer_saved",
        ir_dir=str(ir_dir),
        detokenizer=detok is not None,
        compressed=False,
    )


def _is_converted(ir_dir: Path) -> bool:
    """Is there a usable IR here already? Mirrors `registry.ModelEntry.is_converted`.

    `hf_vlm` writes no `openvino_model.*`, so a single-file check never held for a VLM and
    setup re-ran the ~14-minute gemma-4 build every time. Checks xml *and* bin, so a
    half-written export re-converts rather than being trusted.
    """
    pairs = (
        (IR_XML_NAME, IR_BIN_NAME),
        (VLM_LANGUAGE_MODEL_XML_NAME, VLM_LANGUAGE_MODEL_BIN_NAME),
    )
    return any(all((ir_dir / n).exists() for n in pair) for pair in pairs)


def regenerate_tokenizer(name: str, cfg: Config, *, out_root: Path = IR_ROOT) -> Path:
    """Rewrite only `openvino_tokenizer.xml` beside an already-converted model.

    Separate from `convert(overwrite=True)`: a full re-convert re-quantises, changing
    `ir_sha256` and invalidating every index (§3.1 rule 4). A tokenizer defect is not a
    reason to force a re-index.

    Mirrors what the entry's converter writes — for `hf_vlm` that is the minja-patched chat
    template (baked into rt_info at save time) and the processor config as well.
    """
    from models.registry import spec_for

    _, spec = spec_for(name, cfg)
    src = source_for(spec.name)
    ir_dir = _ir_dir(src, spec.precision, out_root)
    # hf_vlm has no `openvino_model.xml`; the language tower is the converted-marker.
    if not _is_converted(ir_dir):
        raise ConversionError(
            f"{src.name}: no converted IR at {ir_dir} — there is nothing to regenerate a "
            f"tokenizer beside. Run `uv run python -m scripts.setup --only {name}` first."
        )

    from transformers import AutoTokenizer

    snapshot = snapshot_dir(src)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    _fix_chat_template_for_minja(tokenizer, model=src.name)
    tokenizer.save_pretrained(ir_dir)
    _save_ov_tokenizer(tokenizer, ir_dir, with_detokenizer=src.kind in _CHAT_KINDS)

    if src.kind == "hf_vlm":
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(snapshot)
            _fix_chat_template_for_minja(processor, model=src.name)
            processor.save_pretrained(ir_dir)
        except (OSError, ValueError) as e:
            log.warning("convert.processor_unavailable", model=src.name, error=str(e))
    return ir_dir


_CONVERTERS = {
    "hf_encoder": convert_encoder,
    "hf_causal_lm": convert_causal_lm,
    "hf_vlm": convert_vlm,
    "hf_reranker": convert_reranker,
    "paddle": convert_paddle,
}


def convert(
    name: str,
    cfg: Config,
    *,
    out_root: Path = IR_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one model and return its `manifest.json` entry."""
    from models.registry import spec_for

    role, spec = spec_for(name, cfg)
    src = source_for(spec.name)
    if src.role != role:
        raise ConversionError(f"{spec.name}: registered as role {src.role!r}, config says {role!r}")

    ir_dir = _ir_dir(src, spec.precision, out_root)
    started = datetime.now(UTC)
    if _is_converted(ir_dir) and not overwrite:
        log.info("convert.skip_existing", model=src.name, ir_dir=str(ir_dir))
    else:
        snapshot = snapshot_dir(src)
        _checked_embedding(src, snapshot)
        _CONVERTERS[src.kind](src, spec.precision, out_root)
        log.info(
            "convert.done",
            model=src.name,
            precision=spec.precision,
            ir_dir=str(ir_dir),
            seconds=round((datetime.now(UTC) - started).total_seconds(), 1),
        )

    # §3.1 rule 4. hf_vlm has no `openvino_model.bin`, so hash the language tower — the
    # same pair registry.verify_fingerprint checks. Hashing anything else checks nothing.
    bin_path = ir_dir / VLM_LANGUAGE_MODEL_BIN_NAME
    if not bin_path.exists():
        bin_path = ir_dir / IR_BIN_NAME
    return {
        "role": src.role,
        "hf_id": src.hf_id,
        "hf_revision": src.hf_revision,
        "precision": spec.precision,
        "ir_dir": _relative_to_manifest(ir_dir),
        "ir_sha256": file_sha256(bin_path) if bin_path.exists() else "",
        "ov_version": ov_version(),
        "converted_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **({"embedding": dict(src.embedding)} if src.embedding else {}),
    }


def _relative_to_manifest(ir_dir: Path, manifest_path: Path = MANIFEST_PATH) -> str:
    """Manifest paths are relative to the manifest file, so the tree can be relocated.

    `.as_posix()`, not `str()`: the manifest is committed and read cross-platform (§7.3),
    and `WindowsPath.__str__` emits separators the other machines cannot use.
    """
    base = manifest_path.resolve().parent
    try:
        return ir_dir.resolve().relative_to(base).as_posix()
    except ValueError:
        return ir_dir.resolve().as_posix()


def write_manifest(
    entries: dict[str, dict[str, Any]],
    path: Path = MANIFEST_PATH,
) -> Path:
    """Merge into the existing manifest — converting one model must not drop the others."""
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("models", {})
        except json.JSONDecodeError:
            log.warning("convert.manifest_unreadable_rewriting", path=str(path))
    merged = {**existing, **entries}
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_by": "models/convert.py",
        "ov_version": ov_version(),
        "models": dict(sorted(merged.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    log.info("convert.manifest_written", path=str(path), models=sorted(merged))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HF/Paddle -> OpenVINO IR (ARCHITECTURE.md §7.3)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--only", action="append", default=None, help=f"role or name; {ROLES}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-root", type=Path, default=IR_ROOT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="rewrite openvino_tokenizer.xml beside existing IR; leaves ir_sha256 alone",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    targets = list(args.only or []) or (
        [r for r in ROLES if getattr(cfg.models, r, None) is not None] if args.all else []
    )
    if not targets:
        ap.error("pass --only <role|name> (repeatable) or --all")

    entries: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for target in targets:
        try:
            from models.registry import spec_for

            _, spec = spec_for(target, cfg)
            if args.tokenizer_only:
                ir_dir = regenerate_tokenizer(target, cfg, out_root=args.out_root)
                print(f"tokenizer regenerated: {ir_dir / TOKENIZER_XML_NAME}")
                continue
            entries[spec.name] = convert(
                target, cfg, out_root=args.out_root, overwrite=args.overwrite
            )
        except (ConversionError, RuntimeError, OSError) as e:
            log.error("convert.failed", target=target, error=str(e))
            failures.append(f"{target}: {e}")

    if entries:
        write_manifest(entries, args.manifest)
    for f in failures:
        print(f"FAILED {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
