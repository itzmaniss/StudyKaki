# Offline Multilingual Study RAG — Architecture Contract

> **This file is the contract.** Every agent working on this repo reads it first.
> Do not add components not listed here. Do not change the schema without updating this file.
> Target: Intel Core i5 (11th gen, Iris Xe), 8–16 GB RAM, 100% offline, no NPU.

---

## 0. Non-negotiables

1. **Every stage is idempotent and cached** by `(input_hash, stage_version, config_hash)`.
2. **Every chunk carries provenance**: `doc_id`, `page`, `bbox`. Citations depend on this. It cannot be retrofitted.
3. **Nothing hits the network at runtime.** Model download happens once, in a setup script.
4. **Every model runs through OpenVINO.** No raw PyTorch in the runtime path.
5. **No feature ships without an eval number.** If you can't measure it, don't add it.
6. **Abstain beats hallucinate.** Below the score threshold, say "I couldn't find this in your documents."

---

## 0.1 Toolchain — pinned, not negotiable

| Concern | Choice | Rule |
|---|---|---|
| Python env + deps | **uv** | `uv sync` only. **Never** `pip install`. `uv.lock` is committed and authoritative. |
| Python version | 3.11 | Pinned in `.python-version`. Not 3.12+ — OpenVINO wheel coverage lags. |
| Tabular data | **polars** | Not pandas. Lazy frames for anything over 10k rows. |
| Columnar storage | **parquet** | Blocks, chunks, eval runs, telemetry. Compression `zstd`. |
| Vectors | **`.npy` memmap** | **Not** parquet. Contiguous float32, `np.load(..., mmap_mode="r")`. Parquet round-trips are slow and cost you RAM you don't have. |
| Lint + format | **ruff** | `uv run ruff check --fix` and `uv run ruff format` before every commit. |
| Tests | **pytest** | `uv run pytest -q`. |
| Config | **pydantic-settings** + YAML | Config is validated on load, never a bare dict. |

Storage layout:

```
data/
  cache/<stage>/<content_hash>.parquet   # per-stage ingest cache
  index/<index_id>/
    chunks.parquet                       # Chunk rows, polars-readable
    vectors.npy                          # float32 [n_chunks, dim], memmapped
    index_manifest.json                  # §3.1 — embedder fingerprint
  eval/runs/<timestamp>_<config>.parquet # one row per question, appendable
  traces/<date>.parquet                  # per-query telemetry
```

### Version pinning — the OpenVINO trap

These three **must** be the same version. Mismatch produces confusing runtime errors that
look like model bugs. Pin exact, no carets:

```toml
[project]
requires-python = "==3.11.*"
dependencies = [
  "openvino==2026.3.0",
  "openvino-tokenizers==2026.3.0",
  "openvino-genai==2026.3.0",
  "nncf==2.18.0",
  "optimum-intel==1.25.2",
  "polars>=1.20",
  "pyarrow>=19",
  "numpy>=2.1",
  "pymupdf>=1.25",
  "pydantic-settings>=2.7",
]
```

Run `uv lock --check` in CI and before any commit that touches dependencies.
If a model fails to load, **check version alignment before debugging the model.**

---

## 1. Layout

```
configs/
  base.yaml            # single source of truth for models, devices, thresholds
core/
  schema.py            # dataclasses below — WRITE THIS FIRST
  config.py            # loads + hashes config
  cache.py             # content-addressed stage cache
  telemetry.py         # per-query JSON traces
models/
  convert.py           # HF -> OpenVINO IR -> INT8/INT4, writes manifest.json
  manifest.json        # GENERATED. Rubric evidence for Intel tech usage.
  registry.py          # load a model by name from manifest, with device fallback
ingest/
  load.py              # bytes -> pages (+ native text layer if present)
  ocr.py               # PaddleOCR via OpenVINO -> Blocks with bbox
  normalize.py         # NFC, script detect, de-hyphenate, whitespace
  chunk.py             # structure-aware, heading-path preserving
  embed.py             # BGE-M3 INT8, batched
  index.py             # flat float32 index (exact; no recall to apologise for)
  pipeline.py          # orchestrates the above, resumable
retrieve/
  dense.py
  fusion.py            # interface exists even if only dense is wired
  retriever.py         # top-k + threshold + provenance
answer/
  prompt.py            # numbered context blocks
  generate.py          # OpenVINO GenAI LLMPipeline, streaming
  cite.py              # verify cited ids exist; drop invented ones
eval/
  golden.jsonl         # 40-60 questions minimum
  run.py               # recall@k, MRR@10, groundedness, latency
  bench.py             # TTFT, tok/s, peak RSS, per-stage timings, CPU vs GPU
ui/
  app.py               # thin. no logic here.
```

---

## 2. Schema (`core/schema.py`)

Write this before anything else. Everything downstream depends on it.

```python
@dataclass(frozen=True)
class Document:
    doc_id: str  # sha256 of file bytes
    filename: str
    mime: str
    n_pages: int
    has_text_layer: bool  # if True, skip OCR entirely
    pipeline_version: str


@dataclass(frozen=True)
class Block:
    block_id: str
    doc_id: str
    page: int  # 1-indexed
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 normalised 0-1
    kind: str  # heading | paragraph | table | caption | list
    reading_order: int
    script: str  # latn | hans | hant | jpan | kore | taml | thai | deva
    text: str
    ocr_confidence: float | None  # None when from native text layer


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    block_ids: list[str]
    bbox_union: tuple[float, float, float, float]
    heading_path: list[str]  # ["Chapter 3", "3.2 Photosynthesis"]
    text: str
    token_count: int
    lang: str
    script: str


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Retrieved]  # empty == abstained
    abstained: bool
    trace_id: str
```

---

## 3. Ingest pipeline

Each stage: pure function, cached, resumable. Log timing per stage.

| Stage | In | Out | Notes |
|---|---|---|---|
| `load` | file bytes | `Document` + page images + text layer | **If the PDF has a text layer, extract it with pymupdf and skip OCR.** Biggest single speedup available. |
| `ocr` | page images | `list[Block]` | PaddleOCR **mobile/tiny** det + rec. Never the server models. Shared detector, per-script recognition head. |
| `normalize` | blocks | blocks | Unicode NFC. Script detect per block. Join hyphen-broken lines. Collapse whitespace. |
| `chunk` | blocks | `list[Chunk]` | Group by reading order within a heading. Target ~400 tokens, overlap ~60. **Never split mid-block.** For CJK/Thai count characters, not whitespace tokens. |
| `embed` | chunks | vectors | BGE-M3 INT8. **Batch 16–32.** Never one call per chunk. |
| `index` | vectors | index file + `index_manifest.json` | Flat float32 + cosine. At student corpus scale this is exact and fast enough. HNSW only if eval shows you need it. **Must write the embedder fingerprint (§4.1).** |

**Chunking is the highest-leverage knob.** Tune it against the eval set before touching anything else.

---

### 3.1 Embedder fingerprint — MANDATORY

Query-time and index-time embeddings **must** come from a byte-identical model. Drift here
fails silently: retrieval quality collapses and it looks like a chunking bug. Cost you a day.

`index_manifest.json`, written at index build:

```json
{
  "index_id": "sha256:...",
  "n_vectors": 12043,
  "chunk_config_hash": "sha256:...",
  "embedder": {
    "hf_id": "BAAI/bge-m3",
    "hf_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "ir_sha256": "sha256 of openvino_model.bin",
    "ov_version": "2026.3.0",
    "precision": "int8",
    "dim": 1024,
    "pooling": "cls",
    "normalize": true,
    "max_len": 8192,
    "query_prefix": "",
    "passage_prefix": ""
  }
}
```

Rules:

1. **Pin `hf_revision` to a commit SHA**, never a branch. HF models are updated in place.
2. On every query, recompute the runtime fingerprint and compare. On mismatch, **hard fail**
   with "index built with a different embedder — re-index required." Never degrade quietly.
3. `query_prefix` / `passage_prefix` live in the manifest, not in code. BGE-M3 needs neither;
   E5-family models need asymmetric `query: ` / `passage: `. If you swap models and the prefix
   doesn't travel with the index, retrieval silently degrades.
4. `ir_sha256` is the real identity. Two INT8 quantizations of the same checkpoint are
   different models.

---

## 4. Retrieval and answering

```
query -> detect lang/script
      -> embed (same model, query prefix if the model wants one)
      -> dense top-k (k=20)
      -> threshold: if top score < tau, ABSTAIN
      -> take top-n (n=5) into context
      -> generate with numbered blocks [1]..[5]
      -> cite.verify(): strip any [n] the model invented
```

Prompt contract: context blocks are numbered and each carries `doc / p.N`. Instruct the
model to answer **in the language of the question** and to cite block numbers inline.
If context is insufficient, it must say so rather than guess.

---

## 5. Eval — build this on day 1 with a stub retriever

`eval/golden.jsonl`, one object per line:

```json
{"q": "...", "lang": "ta", "doc_id": "abc123", "gold_pages": [42], "note": "..."}
```

Aim for 40–60 questions, spread across your three languages, including:
- 10 cross-lingual (question language != document language)
- 5 unanswerable (must abstain — this tests the threshold)
- 5 that need a table or figure caption

`python -m eval.run --config configs/base.yaml` prints:

```
recall@1  recall@5  recall@10  MRR@10  abstain_precision  groundedness
```

`python -m eval.bench` prints TTFT, tok/s, peak RSS, per-stage ingest latency, for CPU and for GPU.

**These numbers are the credibility of your pitch video.** Build the harness with a random
retriever first so you know the harness itself works, then plug in the real one.

---

## 6. Config (`configs/base.yaml`)

Everything tunable lives here. Nothing is hardcoded.

```yaml
models:
  ocr_det:   { name: PP-OCRv6_mobile_det, device: GPU, precision: int8 }
  ocr_rec:   { name: PP-OCRv5_mobile_rec, device: GPU, precision: int8 }
  embedder:  { name: bge-m3,              device: GPU, precision: int8 }
  generator: { name: qwen3-4b-instruct,   device: CPU, precision: int4 }
chunk:   { target_tokens: 400, overlap: 60, min_tokens: 80 }
retrieve:{ k: 20, n_context: 5, tau: 0.35 }
generate:{ max_new_tokens: 512, temperature: 0.2 }
```

Embedder on iGPU + generator on CPU means ingestion and generation don't fight for the
same execution units. Measure both ways and keep the numbers — that comparison is a slide.

---

## 7. OpenVINO gotchas that will eat your day

1. **Pin `openvino`, `openvino-tokenizers`, `openvino-genai` to the exact same version.** Mismatched versions are the number one time sink. Pin in `requirements.txt` on hour one.
2. **Set `CACHE_DIR`.** First GPU compile of a model is slow; cached is seconds. Warm it in setup.
3. **Convert models once, offline**, in `models/convert.py`. Commit `manifest.json`, not weights.
4. **Always implement GPU -> CPU fallback** on device init failure. Log which device won.
5. **Batch embeddings.** A per-chunk `embed()` loop is 10x slower for no reason.
6. **PaddleOCR mobile/tiny only.** Server models will not hit interactive latency on an i5.

---

## 8. Build order (today)

| Block | Work | Done when |
|---|---|---|
| 0–1h | Repo skeleton, pinned deps, `schema.py`, `base.yaml`, `eval/run.py` with a **random** retriever | Eval harness prints a table of garbage numbers |
| 1–3h | `models/convert.py`, `manifest.json`, verify every model loads on CPU and GPU | `bench.py` prints real tok/s |
| 3–6h | Ingest pipeline end to end with caching, one language | One PDF indexed, chunks have correct page + bbox |
| 6–8h | Retrieve + answer + citation verify + abstain | Answer renders with a clickable page cite |
| 8–10h | 40 golden questions, tune chunk size against recall@5 | recall@5 measured and improving |
| Day 2 | UI, telemetry panel, languages 2 and 3, CPU-vs-GPU comparison numbers | **FEATURE FREEZE end of day 2** |

---

## 9. Answer source hierarchy

Three tiers, strictly ordered. The tier used **must** be visible in the UI on every answer.

| Tier | Source | Status | Behaviour |
|---|---|---|---|
| 1 | Local index | **BUILD** | Default. Cites `doc / p.N`. Used whenever top score ≥ `tau`. |
| 2 | Online documents | **INTERFACE ONLY — do not build** | Permission-gated toggle, requires network. Ship as roadmap. |
| 3 | Model parametric knowledge | **BUILD** | Default **OFF**. Opt-in only. |

Tier 3 contract — this is a study tool, and an ungrounded answer that contradicts the
student's syllabus is worse than no answer:

- Defaults to off; user must explicitly enable it.
- Renders in a visually distinct style (different background, warning icon).
- Carries the literal text: *"General knowledge — not from your materials. May not match your syllabus."*
- **Emits no citation markers.** Never fabricate a `[n]` or a page number.
- If Tier 1 abstained, say so explicitly before offering Tier 3.

Tier 2 stays a stubbed interface (`answer/sources/online.py` raising `NotImplementedError`)
so the architecture diagram is honest and the roadmap is concrete. Building it would cost a
day and undercut the "nothing leaves your device" claim, which is the strongest asset in the pitch.

---

## 10. V2 — tomorrow morning, gated on tonight's eval

Every V2 component ships with a before/after number from `eval/run.py`. If a component
doesn't move `recall@5` or `groundedness`, it gets cut. **No component ships on vibes.**

| Component | Build | Expected win | Watch out |
|---|---|---|---|
| **Cross-encoder rerank** | `retrieve/rerank.py`, `bge-reranker-v2-m3` INT8 via OpenVINO GenAI `TextRerankPipeline`. Dense k=20 → rerank → n=5. | Largest single precision gain. Also adds a 4th OpenVINO pipeline (Metric 03). | 20 cross-encoder passes on CPU is ~1–2s. Put it on iGPU, or rerank top-10 only, if TTFT suffers. |
| **Hybrid dense + BM25** | `retrieve/lexical.py` + `fusion.py` (reciprocal rank fusion, k=60). | Exact-term recall: formulas, proper nouns, technical vocabulary. The terms students actually look up. | **The work is per-script tokenization**, not BM25. jieba (zh), fugashi (ja), pythainlp (th), ICU fallback. Latin/Tamil/Devanagari split on whitespace. |
| **Conditional query rewrite** | `retrieve/rewrite.py`. Trigger **only** when query < 5 tokens OR contains an unresolved pronoun. | Fixes conversational follow-ups ("what about the second one?"). | Full generation round-trip before retrieval — 2–4s added on CPU. And BGE-M3 is *built* for cross-lingual retrieval, so rewriting a Tamil query into English may destroy the signal. **Measure before keeping.** |

Order: rerank → hybrid → rewrite. Stop when the clock says stop; V1 already works.

---

## 11. Still out of scope

Handwriting OCR. PPTX ingestion. HyDE. Agentic multi-hop. Installer / packaging.
Fine-tuning. More than three languages. Online document fetching (Tier 2).

Interfaces exist so these can be added later. **They are not being added this week.**
