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
