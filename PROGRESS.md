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
