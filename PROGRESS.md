# PROGRESS

Append-only. Newest at the bottom.

## 2026-08-22 02:35 — Block 0–1h: skeleton, schema, config, eval harness
Done: Repo skeleton per §1 (flat top-level packages; removed the `uv init` `src/` stub layout).
`pyproject.toml` repinned to §0.1 exactly — Python `==3.11.*`, openvino/-tokenizers/-genai all
`2026.3.0`, nncf 2.18.0, optimum-intel 1.25.2, polars/pyarrow/numpy/pymupdf/pydantic-settings,
plus structlog and pyyaml. `core/schema.py` (§2) with validation in `__post_init__` and a
parquet round-trip mixin that restores bbox tuples. `core/config.py` — pydantic, `extra="forbid"`,
frozen, with `config_hash` and a separate `chunk_config_hash` so retuning `tau` does not
invalidate the chunk cache. `configs/base.yaml` verbatim from §6. `retrieve/retriever.py`
Protocol + `abstains()`. `eval/metrics.py`, `eval/run.py` with the mandated **random** retriever.

Verified: `uv run pytest -q` → **45 passed**. `uv run ruff check .` clean, `uv lock --check` ok.
OpenVINO stack imports and version-aligns on this machine:
`openvino 2026.3.0 / genai 2026.3.0.0 / tokenizers 2026.3.0.0`.
Harness output (`uv run python -m eval.run --config configs/base.yaml`):

```
retriever=random  n=12
recall@1  recall@5  recall@10  MRR@10  abstain_precision  groundedness
----------------------------------------------------------------------
0.111     0.222     0.444      0.179   1.000              n/a
```

Garbage numbers by design — the §8 Block 0–1h gate is "the harness prints a table", and it does.
`groundedness` is `n/a` until `answer/` exists; `abstain_precision` is vacuously 1.000 because
20 uniform-random scores almost always clear `tau=0.35`, so nothing abstains.

Next: `models/convert.py` + `models/registry.py` with device fallback (§8 Block 1–3h).

Blockers raised: hardware is Apple M4 Pro, not Intel i5/Iris Xe — `available_devices == ['CPU']`,
so the CPU-vs-GPU slide cannot be produced here. See BLOCKERS.md.

## 2026-08-23 00:45 — Blocks 1–3h / 3–6h: wave 1 salvaged and committed
Done: Four parallel agents (models, ingest, answer, telemetry) each wrote their files and then
**all four were killed by a session limit at the reporting step** — after their code had landed.
Nothing was rebuilt. The lead verified, formatted, fixed one E501 and one blind-`except` lint,
and committed as `54020ed`.

Landed: `models/registry.py` (device fallback + §3.1 fingerprint + `verify_fingerprint` that
raises), `models/convert.py`, `scripts/setup.py`, `core/cache.py` (root injected — nothing under
`core/` knows about `data/`), `ingest/load.py` (text-layer detection so OCR is skipped),
`ingest/normalize.py`, `ingest/chunk.py`, `answer/prompt.py`, `answer/cite.py`,
`answer/sources/online.py` (stays a `NotImplementedError` stub per §9 Tier 2),
`core/telemetry.py`, `eval/bench.py`.

Verified: `uv run pytest -q` → **408 passed**. ruff clean, `uv lock --check` ok.
**BGE-M3 was really converted to INT8 OpenVINO IR and loads:**

```
model.device_unavailable  available=['CPU'] requested=GPU falling_back_to=CPU
model.loaded  model=bge-m3 device=CPU fell_back=True precision=int8 cached=True
loaded in 0.5s | inputs [input_ids, attention_mask] | outputs last_hidden_state [?,?,1024]
```

Device fallback works and logs honestly, which is the §7.4 requirement.

Next: wave 2 respun — `indexer` (embed/index/pipeline), `searcher` (dense/fusion),
`ocrsmith` (ingest/ocr.py — the one stage nobody had), `generatorsmith` (generate/ui).

Known debt: `eval/bench.py` is 565 lines, over CLAUDE.md's 500-line limit. Split it when touched.
`models/manifest.json` holds only `bge-m3`; the Qwen3-4B INT4 generator is not converted yet.

## 2026-08-23 07:10 — V1 COMPLETE through §8 Block 6–8h. Stopping per CLAUDE.md.
Done: Third wave (`indexer2`, `tokenfixer`, `finisher`). Added `ingest/index.py`,
`ingest/pipeline.py`, `ui/app.py`, and the OCR tests that never existed. Committed `dab663e`.

**Two real defects fixed that the fake-backed suite could not see:**

1. `models/convert.py:363` — tokenizer/detokenizer IR was saved with `ov.save_model`'s default
   `compress_to_fp16=True`. BGE-M3's Unigram/SentencePiece tokenizer op is a *reference*
   implementation that reads its vocab as f32, so **the real embedder could not run at all**:
   `Tensor data with element type f16, is not representable as pointer to f32`. Every tokenizer
   test used a fake, so 563 tests passed over a completely broken real path. Only the tokenizer
   IR was regenerated — re-quantising the model would change `ir_sha256` and invalidate every
   index (§3.1 rule 4).
2. `ingest/ocr.py` — per-script confidence thresholds were a **no-op in the shipping engine**.
   The default head is multilingual and has no script of its own, so a threshold keyed on
   `head.script` never matched the script of the text actually read. This is exactly the Tamil
   trap CLAUDE.md warns about: the thresholds existed but did not apply. Now falls back to
   `detect_script(text)`. `confidence_by_script` is empty by default — no per-script numbers
   were invented without ground truth.

Also fixed earlier this wave: `ingest/load.py` bbox on rotated pages (`get_text` returns
unrotated coordinates while `page.rect` is rotated, so a `/Rotate 180` page cited the opposite
corner).

Verified: **702 passed / 8 skipped** fast; **`INTEL2026_REAL_MODELS=1` → 710 passed, 0 skipped.**
ruff clean, `uv lock --check` ok.

**End-to-end on the real BGE-M3** (4-page generated PDF -> ingest -> index -> dense retrieve):

```
ingest: 1 doc in 0.3s -> index: 4 vectors, dim=1024, float32
Q: What enzyme fixes carbon dioxide?      220ms  top=0.552  abstain=False  -> p.3 Calvin cycle
Q: Where do the light reactions happen?    16ms  top=0.703  abstain=False  -> p.2 thylakoid
Q: What is the capital of France?          15ms  top=0.269  abstain=True   -> (correctly declined)
```

Retrieval is 15–16ms warm. The abstain threshold works: the out-of-corpus question scores 0.269
against `tau=0.35` and is declined. That is §0.6 demonstrated, not asserted.

Latest `eval/run.py` output — **still the random retriever**, because `eval/golden.jsonl` is
12 `PLACEHOLDER` entries against imaginary documents:

```
retriever=random  n=12
recall@1  recall@5  recall@10  MRR@10  abstain_precision  groundedness
0.111     0.222     0.444      0.179   1.000              n/a
```

**These numbers are still meaningless, and that is the honest state.** Wiring `DenseRetriever`
into `eval/run.py` is trivial, but it cannot produce a real recall@5 until the golden set points
at a real corpus. Block 8–10h ("40 golden questions, tune chunk size against recall@5") is
therefore blocked on the developer, not on code. See BLOCKERS.md #3.

**Stopping here. Not starting §10 (V2).** CLAUDE.md is explicit that rerank / hybrid / query
rewrite are gated on the developer reading tonight's eval numbers, and no component ships on
vibes. There are no real eval numbers yet to gate on.

Remaining V1 gaps, none blocking: `models/manifest.json` still has only `bge-m3` — the
Qwen3-4B INT4 generator is unconverted, so `answer/generate.py` is tested against a fake
pipeline and has never run a real token. Files over CLAUDE.md's 500-line limit:
`tests/test_dense.py` 788, `eval/bench.py` 565, `tests/test_generate.py` 549,
`answer/generate.py` 546, `models/convert.py` 533.

## 2026-08-24 11:15 — pre-V2 sweep: deps, chunker, groundedness, golden set
Done, in commit order:
- `deps: pin torch<2.9` — the generator export had been failing on
  `ImportError: _attention_scale`. torch 2.9 moved the TorchScript ONNX exporter to
  `torch.onnx._internal.torchscript_exporter` and left `symbolic_opset14` as a star-import
  shim, which cannot re-export underscore names. optimum-intel and nncf are pinned by
  ARCHITECTURE §7.1, so torch was the only lever. 2.8.0 imports clean; suite still green.
- `ingest: pick the OCR head per document` — `script_hint` was a pipeline-level setting, so a
  single run over a mixed ta/en/zh corpus could only be right for one language. All three
  Tamil PDFs route to OCR, so they would have gone to the Chinese+English default head and come
  back as CJK noise at high confidence — BLOCKERS #5 all over again. The hint now follows the
  corpus directory (`data/corpus/<lang>/`); `--script-hint` overrides.
- `ingest: enforce min_tokens across section boundaries (chunk/2)` — `_merge_short` only folds
  within a section, so a section that is one short group shipped at any size. std12_cs_vol2_en
  produced twelve `"CHAPTER n"` chunks of 2 tokens each, 10% of that document's index and
  twelve near-identical vectors that match any query containing the word chapter. **Stage
  version bumped: chunk and embed caches rebuild.**
- `eval: wire DenseRetriever into the harness` — `--retriever dense`, lazy import so the
  random baseline still runs with no IR on disk.
- `eval: wire the groundedness column` — fraction of the model's citation markers that survived
  `cite.verify`. Previously unobservable: `answer.text` is already cleaned, so invented markers
  are gone by the time a caller sees it. Off by default (`--groundedness`).
- `eval: replace the placeholder golden set` — **44 real questions**, every answer read off the
  page it cites. 8 cross-lingual, 5 unanswerable, 7 table/figure. No PLACEHOLDER remains.

Verified: `uv run pytest -q` → 732 passed, 8 skipped. `ruff format`/`check` clean, `uv lock --check` clean.
Corpus routing confirmed on real files: 3 Tamil PDFs -> OCR (`taml` head), 5 others -> text layer.
Smoke test end-to-end: std12_cs_vol2_en 240pp -> 2673 blocks -> 126 chunks -> index, 2m17s on CPU.

In flight (background, unattended): full-corpus ingest over all 8 PDFs / 2617 pages, and the
Qwen3-4B INT4 conversion. No GPU on this machine — everything falls back to CPU and says so.

Next: Tamil golden questions (blocked until OCR output is readable — BLOCKERS #3), then the
baseline `eval/run.py --retriever dense` table, then chunk-size tuning against recall@5.
