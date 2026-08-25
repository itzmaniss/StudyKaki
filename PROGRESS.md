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

## 2026-08-24 21:35 — BLOCKERS #11 actually fixed; V1 correctness before any V2 arm

Context for the developer: `aaf77c7` and `acc20e3` (Tamil golden set, V1 baseline) landed
without PROGRESS entries — `eval/baselines/v1.json` is the authoritative record of that
baseline, since `data/` is gitignored and run parquets cannot serve as the "before" column.

Done:
- `models: register the reranker role left out of the §10 config surface` (`cabf0a8`) —
  `564a12c` added `models.reranker` to `core/config.py` and a `reranker:` entry to
  `configs/base.yaml` but not to `registry.ROLES`, so `spec_for("reranker", cfg)` would have
  raised. Resolution is lazy, so nothing broke at load; the rerank arm simply could not be
  wired. Optional like the per-script OCR heads.
- `answer: trim context to a prompt-token budget (BLOCKERS #11)` (`ae14943`) — **the fix had
  been declared but never implemented.** `generate.max_prompt_tokens` existed in config and
  `base.yaml`; nothing read it. Every Tamil question still died on OpenVINO's INT4 CPU MatMul.
  `retrieve.n_context` is now a request, not a guarantee: `fit_context` drops blocks from the
  tail until the prompt fits, highest-scoring context surviving, logging
  `generate.context_trimmed`. A lone oversized block is attempted rather than abstained.

The part worth remembering: the budget needs a *real* token count. `ingest.chunk.count_tokens`
counts whitespace words and undercounts Tamil subwords ~7x — using it would have reproduced the
crash while looking correct, the same blindness that let `recall@5 = 0.898` coexist with zero
answerable Tamil questions. So generation uses the pipeline's own tokenizer, falling back to
`estimate_tokens`, a script-weighted character rate with a 15% safety margin. Against #11's
three measured prompts it over-counts by ~15% and never under.

Verified: `uv run ruff format`/`check` clean, `uv lock --check` clean,
`uv run pytest` → **764 passed, 8 skipped** (was 749; +15 covering the budget, the trim order,
the citation contract under trimming, and the estimate's floor).

Also found, not yet acted on: `data/index/` holds two indexes — the full corpus (2595 vectors,
the one `v1.json` pins) and a 126-vector smoke-test leftover from one document. Same
`chunk_config_hash`. `default_index_dir` refuses to choose between them, so `eval/run.py` and
anything built on it must pass `--index` explicitly. Left in place rather than deleted.

Next: `eval/run.py --retriever dense --groundedness` on the pinned index, to replace the null
groundedness column in `eval/baselines/v1.json`. That number is what §10 gates the V2 arms on.

## 2026-08-25 00:50 — V2 §10 arms built on branch `v2`

Branch, not main: V1 behaviour is untouched and every arm still defaults off.

- `retrieve: BM25 lexical arm + hybrid RRF retriever` (`f563b8a`) — per-script tokenization is
  the real work per §10. Whitespace for Latin/Tamil/Devanagari, character bigrams for CJK,
  jieba used if installed but not a dependency. Okapi BM25 hand-rolled; `rank_bm25`/`bm25s`
  are not worth a dep for 2595 chunks.
  **The Tamil trap:** a `\w`-based token regex drops Unicode combining marks (category Mn), so
  `ஜார்ஜ்` shatters into single letters and Tamil BM25 silently returns garbage. Caught by a
  test, fixed by splitting on whitespace literally as §10 says.
- `eval: two-phase §10 arm sweep` (`799327d`) — retrieval is ~30 ms/question, generation ~90 s,
  so all 8 permutations score on recall/MRR in seconds and `--groundedness` is opt-in for
  finalists only. A full 8-way generative sweep would have been ~11 hours.
- `retrieve: conditional query rewrite arm` (`9ac5ebf`) — §10 trigger only.
- `models: add bge-reranker-v2-m3 conversion` (`74cd945`) + `retrieve: cross-encoder rerank`
  (`4585e79`, `ac7635f`).

**Three bugs worth remembering, all silent:**
1. Tuples have an `.index` *method*, so `hasattr(item, "index")` is always true and picked the
   wrong branch for `(idx, score)` pairs.
2. Post-hoc `nncf.compress_weights` on a saved IR reads and writes the same file. It died
   mid-write leaving a model with **no tokenizer**, and a shell pipe reported exit 0. Quantize
   at export instead.
3. `TextRerankPipeline` loads clean on arm64 then throws on every call (BLOCKERS #13a). Our own
   graceful degradation then hid it — the first sweep reported rerank as "no gain" when the arm
   had never run once. **Degradation paths need to be loud enough to notice in a results table.**

Verified: `uv run ruff format`/`check` clean, `uv lock --check` clean, `uv run pytest` →
**786 passed, 8 skipped** at the sweep commit, plus rerank/rewrite/lexical suites since.

Measured, dense-vs-hybrid on the pinned index (54 questions, retrieval only):

```
arm                       recall@5        Δ   MRR@10        Δ     ms/q
dense (V1)                   0.898   -0.000    0.736   -0.000       39
hybrid                       0.939   +0.041    0.774   +0.038       36
rewrite                      0.898   -0.000    0.736   -0.000     1143
hybrid+rewrite               0.939   +0.041    0.774   +0.038     1070
```

Hybrid is the clear win: **+0.041 recall@5 for ~1 ms** of BM25. Rewrite fired on 4 of 54
questions and moved nothing, exactly as predicted — the golden set is all standalone questions,
so this is a legitimate "no effect", not a failure. Its 1143 ms/q is mostly one-off generator
load amortised over 54 questions, not per-query cost.

Two blockers raised: **#12** (hybrid RRF scores break the abstain gate — it measures fine but
cannot ship behind `answer/`) and **#13** (rerank costs 5-11 s/query, ~10x §10's estimate,
because our chunks are 400 tokens; §10's iGPU mitigation does not exist on this box).

Next: rerank quality numbers, then a decision on #12 and #13 before anything merges to main.

## 2026-08-25 01:05 — §10 sweep results, all arms working

54 questions, pinned index `73677e03`, retrieval only. **0 degraded calls across 108 rerank
invocations**, so these are real measurements (the first sweep's rerank row was not — see below).

```
arm                       recall@5        Δ   MRR@10        Δ     ms/q
dense (V1)                   0.898   -0.000    0.736   -0.000       37
hybrid                       0.939   +0.041    0.774   +0.038       34
rerank                       0.918   +0.020    0.891   +0.155     6781
rerank+hybrid                0.980   +0.082    0.917   +0.181     7614
rewrite                      0.898   -0.000    0.736   -0.000     1143
hybrid+rewrite               0.939   +0.041    0.774   +0.038     1070
```

**`rerank+hybrid` reaches recall@5 = 0.980, which is the corpus ceiling, not just a good
number.** `v1.json` records recall@10 == recall@20 == 0.9796 and one answerable question whose
gold chunk is never retrieved at any depth — so 48/49 is the maximum any retriever can score
here. The +0.082 headroom predicted from the baseline rank distribution is now fully consumed.

The bigger shift is MRR@10: 0.736 -> 0.917. The correct chunk is now nearly always rank 1, and
that is rerank's doing (it alone moves MRR +0.155 while moving recall only +0.020). It reorders
7.9 of 10 candidates on an average query — it is not a tiebreaker, it is re-deciding the top.

Division of labour is clean: **hybrid widens the candidate pool** (recall), **rerank orders it**
(precision). Neither substitutes for the other, which is why the combination hits the ceiling
and either alone does not.

Cost: rerank is **7.1 s mean, 2.6-17.1 s range** at `top_n: 10`. Hybrid is free (~1 ms of BM25;
its 34 ms/q is dense's own latency). See BLOCKERS #13 — this is ~10x §10's estimate because our
chunks are 400 tokens, so each pair is ~600.

Also this session: `eval+retrieve: never report a number an arm did not produce` (`149d6f6`).
The previous sweep printed rerank at 0.898 with no gain. That was not a result — the arm threw
on all 54 questions and degraded to its inner ranking. Arms now count degraded calls and the
sweep withholds metrics for any row with a non-zero count.

Verified: `uv run pytest` -> **817 passed, 8 skipped**, ruff + `uv lock --check` clean.
Sweep parquet: `data/eval/sweep_rerank.parquet`.

Next: groundedness on `rerank+hybrid` is the remaining half of the §10 gate — retrieval is only
one of the two metrics it cuts on. ~80 min. Nothing merges to main before that.

## 2026-08-25 12:40 — V2 verdict: ship hybrid, cut rerank, cut rewrite

§10's rule is "every V2 component ships with a before/after number; if it doesn't move recall@5
or groundedness, it gets cut." Measured on 54 questions, index `73677e03`, 0 degraded calls:

| arm | recall@5 | Δ | MRR@10 | Δ | cost/query | verdict |
|---|---|---|---|---|---|---|
| dense (V1) | 0.898 | — | 0.736 | — | 37 ms | baseline |
| **hybrid** | **0.939** | **+0.041** | **0.774** | +0.038 | **~1 ms** | **SHIP** |
| rerank | 0.918 | +0.020 | 0.891 | +0.155 | 6.6-19.5 s | cut |
| rerank+hybrid | 0.980 | +0.082 | 0.917 | +0.181 | 6.6-19.5 s | cut |
| rewrite | 0.898 | +0.000 | 0.736 | +0.000 | — | cut |

**Ship hybrid.** +0.041 recall@5 for ~1 ms of BM25. No new model, no new failure mode, and it is
the half of the ceiling that is free. Flip `retrieve.hybrid.enabled: true`.

**Cut rewrite.** Measured no effect. It fired on 4 of 54 questions; every golden question is a
standalone, so the §10 trigger almost never fires. Not a failure — the arm is for conversational
follow-ups this corpus does not contain. Keep the code, leave it off.

**Cut rerank — on cost, not quality.** It passes §10's quality gate outright: rerank+hybrid
reaches recall@5 = 0.980, which is the *corpus ceiling* (one answerable question's gold chunk is
never retrieved at any depth, so 48/49 is the maximum anything can score), and MRR@10 0.736 ->
0.917 means the right chunk is nearly always rank 1.

It fails on latency, in exactly the way §10 predicted and in a way §10's mitigations cannot fix:
- §10 budgeted 1-2 s. Measured 6.6 s (en/zh), **19.5 s (Tamil)**, 42.5 s worst (BLOCKERS #13).
- §10's fixes were "iGPU, or top-10 only". `top_n` is already 10 — free, per the rank
  distribution — and there is no iGPU on this machine (BLOCKERS #1).
- §5 pitches TTFT. Twenty seconds before the first token, on a third of the corpus, is not a
  study tool.

Cost scales with the *language of the retrieved chunks*, not the query — the same tokenizer
density that caused BLOCKERS #11. Tamil is where the multilingual claim lives and where this is
worst.

**Groundedness for rerank+hybrid was not obtained, and I stopped trying.** Three runs died at
question 45; the cause is BLOCKERS #14 (memory), not the arm. More importantly BLOCKERS #15 shows
the metric scores Tamil word-salad at 1.00, so the number would not have decided anything.

Two blockers raised that outlive V2: **#14** — V1 needs 13.5 GB steady and 24 GB peak against a
stated 8-16 GB target, so V1 does not fit its own target machine; **#15** — groundedness measures
whether citations resolve, never whether answers are good.

Branch `v2`, 15 commits, `main` untouched. Suite: 817 passed, 8 skipped.

Recommended next action: `git merge v2` (all arms still default off), then flip
`retrieve.hybrid.enabled: true` and re-run `eval/run.py --retriever dense` to confirm 0.939.

## 2026-08-25 14:05 — ui/web.py, ui/index.html

Done: browser UI, for when the demo needs a screen rather than a shell. `http.server` + one
HTML file — §0.1 pins the dependency set and carries no web framework, so no Streamlit and no
FastAPI. Binds 127.0.0.1, loads no CDN font or script, serves cited PDFs from the local
registry: the page works with the wifi off, which is the pitch (§0.3). `ui/app.py` stays the
reference UI; this is the same `Session` wiring behind a page.

Answers stream over SSE. Every answer carries its tier (§9), and Tier 3 renders as a
warning-framed card with the literal disclaimer, no citations, and the abstention said out loud
first — with a one-click "answer from general knowledge" offer, so the opt-in stays an opt-in.
Sources show the page, the heading path and **the first 220 characters of the cited chunk**, so
grounding is legible without opening the PDF. When `answer/cite.py` drops an invented marker
the card says so ("checked — 1 invented citation removed") rather than silently swapping in a
cleaner answer.

Fixed while wiring it: `meta()` read `retriever.index`, which is `None` under
`HybridRetriever` (its index sits on the dense arm) — the shipping config would have reported
an empty corpus in the header.

Verified: `uv run pytest -q` -> **835 passed, 8 skipped** (18 in `tests/test_ui_web.py`), ruff
and `uv lock --check` clean. Driven in a real browser against a fake backend: streaming,
abstain -> Tier 3, invented-marker removal, Tamil/Chinese rendering, light and dark.

Run it: `uv run python -m ui.web --docs data/pdfs` (`--port`, `--no-open`, `--index`).

Next: nothing outstanding on the UI. V2 verdict above still stands — `git merge v2`, flip
`retrieve.hybrid.enabled: true`, re-run the eval to confirm 0.939.

## 2026-08-25 19:30 — branch `gemma-4`: Gemma 4 E2B converts nowhere on this machine

Trying a smaller generator, against BLOCKERS #14 (V1 needs 13.5 GB steady) and the Tamil
generation failure below. `google/gemma-4-E2B-it` — Apache-2.0, ungated, published 2026-07-20,
pinned at `3e22461f`. ~5B raw weights that run as ~2B effective.

Done:
- `models/convert.py`: `gemma-4-e2b-it` source + a new `hf_vlm` kind. Gemma 4 is any-to-any and
  its text tower does not stand alone — the per-layer embeddings are a separate graph — so it
  exports through `OVModelForVisualCausalLM` (`image-text-to-text`), not `OVModelForCausalLM`.
- Toolchain bumped for the `gemma4` architecture: transformers 4.53.3 -> 5.5.4, optimum
  1.27 -> 2.3.0, optimum-intel 1.25.2 -> 2.1.0 (it carries the gemma4 OpenVINO exporter),
  nncf 2.18 -> 3.3. **Suite stayed green through the bump: 835 passed, 8 skipped.** No runtime
  module imports transformers — this touches conversion tooling only.
- `configs/gemma4-e2b{,-int8,-fp16}.yaml` so the arm can be run without disturbing `base.yaml`
  or the pinned v1 baseline.

Not done, and not doable here: **BLOCKERS #16** — NNCF computes compression scales by running
OpenVINO ops over each weight, and the Reduce executor has no arm64 implementation. int4, int8
and uncompressed all die on the same node. This machine can run models but cannot build them.

The three failures before that one were mine, not the machine's, and are fixed in the code:
wrong export class (`image-to-text-with-past` is not registered for gemma4), `ratio=0.8` at
8 bits (it is the INT4 share), `group_size=128` at 8 bits (must be -1).

Verified: ruff clean, `uv lock --check` clean, `uv run pytest` -> 835 passed, 8 skipped.

Next: continue on the Intel laptop — BLOCKERS #16 carries the three commands and the two
wiring gaps (`VLMPipeline` in `load_generator`, multi-part IR in `registry.is_converted`).
`main` is untouched; nothing here changes V1.

## 2026-08-25 19:45 — Tamil is answering nothing, recorded as BLOCKERS #17

Ran the real generator on the pinned index: both Tamil golden questions abstain with top scores
*above* tau (0.608, 0.570), and the V1 baseline run scores Tamil groundedness **0.000** across
all 6 questions against en 0.935 / zh 0.900. The pooled 0.844 hides it, and model-side
abstentions are excluded from groundedness while the `abstained` column tracks only the
retrieval gate — so no reported number moves when Tamil fails completely.

Recorded rather than fixed: the parquet holding this is gitignored, so #17 carries the table.
Cross-lingual retrieval itself is fine — an English question answered correctly off a Tamil
source, and Chinese answered in Chinese.

Verified: ruff clean, `uv lock --check` clean, `uv run pytest` -> 835 passed, 8 skipped.

Next: #17 lists the four candidate fixes in order; the fourth is #16's Gemma 4 experiment,
which needs the Intel laptop.

## 2026-08-25 21:14 — eval/run.py, eval/bench.py, tests/test_ui_web.py: Windows portability

On the Intel Windows laptop for #16. Before touching Gemma 4, ran the continuation plan's
steps 2–3 (index fingerprint check, full suite, real-generator UI sanity check) — the suite
broke immediately. This codebase had never run on Windows before.

Done:
- `eval/run.py` and `eval/bench.py` both did `import resource` unconditionally; `resource` is
  POSIX-only, so 4 test modules failed to collect. Consolidated peak-RSS reading into
  `eval.bench.peak_rss_bytes()` (which already had the darwin-vs-linux `ru_maxrss` unit fix)
  with a Windows branch via `GetProcessMemoryInfo`; `eval.run` now imports it instead of
  duplicating the logic. First attempt silently returned 0 — ctypes truncates
  `GetCurrentProcess()`'s pseudo-handle without explicit `argtypes`/`restype`, so the call
  failed `ERROR_INVALID_HANDLE` and was swallowed. Fixed by typing the call explicitly.
- `zoneinfo.ZoneInfo("UTC")` raised `ZoneInfoNotFoundError` inside polars' parquet round-trip —
  Windows ships no IANA tz database. Added `tzdata` as a Windows-only marker dependency via
  `uv add`, so `uv.lock` changed only as that command's side effect (CLAUDE.md rule 5).
- Two `Path.read_text()` calls (`eval/run.py:load_golden`, `tests/test_ui_web.py`) relied on
  the platform default encoding — cp1252 on Windows — and crashed on the em-dashes in
  `eval/golden.jsonl` and `ui/web.py`. Pinned both to `encoding="utf-8"`.

Verified: `uv run ruff format .` and `ruff check --fix .` clean, `uv run pytest -q` exit 0 with
the same 8 skipped as the last non-Windows run (no count regression), `uv lock --check` clean.

Also ran steps 2–3 for real: step 2 confirmed the carried-over index's embedder fingerprint
still matches (`OK BAAI/bge-m3 c3ea306efeb5 2026.3.0`). Step 3's UI query came back an
abstention, not the cited answer the plan expected — recorded as a new item in BLOCKERS.md
rather than tuned around, since it's a one-sample borderline score.

Next: #16 (Gemma 4 download + INT4 convert) — running separately on this machine already.

## 2026-08-25 21:45 — pyproject.toml: add torchvision, root cause of the gemma4 convert failure

The developer's own `--only generator` run got past the trace and NNCF int4/int8 compression
(all four submodels: language model, text + per-layer embeddings, vision embeddings) and wrote
a complete IR to `models/ir/gemma-4-e2b-it-int4/`, then failed at the very last step:
`AutoProcessor.from_pretrained(snapshot).save_pretrained(ir_dir)` in
`models/convert.py:convert_vlm` raised `Gemma4VideoProcessor requires the Torchvision library`.

Root cause, not the trace-check crash the developer's paste showed and BLOCKERS #16 predicted
(`Supported Reduce executor is not found`, arm64-specific) — that step passed clean on a
same-machine re-run, so it looks non-deterministic under load rather than a real blocker here.
The `AutoProcessor` call is already wrapped in `try: ... except (OSError, ValueError)` for the
case where a model genuinely has no processor (`convert.py:452`), but `ImportError` isn't in
that tuple, so a *missing optional dependency* took down the whole conversion instead of
logging the intended warning and continuing.

`torchvision` was never in the gemma4 toolchain bump (BLOCKERS #16 listed transformers,
optimum, optimum-intel, nncf) — it's Gemma4Processor's own hard import for the video half of
its any-to-any support, needed even though this project only sends text. Added
`torchvision>=0.23.0` (`uv add`, resolved to 0.23.0 against the existing `torch<2.9` pin).
Confirmed in isolation: `AutoProcessor.from_pretrained(snapshot)` now returns a
`Gemma4Processor` instead of raising.

Verified: ruff clean, `uv run pytest -q` exit 0 (unaffected — this dependency is conversion-
tooling only, no runtime module imports it), `uv lock --check` clean.

Re-running `--only generator` now to get a complete IR with the processor config VLMPipeline
needs. Left the `except` clause as-is rather than widening it to `ImportError` — the fix is
having the dependency, not swallowing its absence.

## 2026-08-25 22:05 — models/convert.py: gemma-4-e2b-it conversion completes; manifest path bug

The re-run with `torchvision` installed **converted clean**: full multi-part IR (language
model, text + per-layer embeddings, vision embeddings, tokenizer, detokenizer, processor
config) written to `models/ir/gemma-4-e2b-it-int4/`, 625.9s, no `Reduce executor` failure at
all on this hardware (that failure was arm64-specific, per BLOCKERS #16). Manifest updated.

Caught before committing: `_relative_to_manifest` (`models/convert.py:649`) used `str(Path)`
for the manifest's `ir_dir` field. Every existing entry reads `ir/<name>` (written on macOS/
Linux); this run wrote `ir\gemma-4-e2b-it-int4` — `WindowsPath.__str__` uses `\`, which is not
a path separator on POSIX, so the committed manifest would have parsed as one literal
directory named `ir\gemma-4-e2b-it-int4` on the baseline machine. Switched to `.as_posix()`,
added `test_manifest_ir_dir_is_always_forward_slashed` as a regression, hand-corrected the
already-written gemma-4 entry to match.

**Still open, not fixed:** the manifest's `ir_sha256` for gemma-4-e2b-it is `""`. `write_manifest`
hashes `ir_dir / "openvino_model.bin"` (`models/registry.py:IR_BIN_NAME`), which the VLM export
doesn't produce — it writes `openvino_language_model.bin` instead. Same root cause as the
`is_converted`/warm-stage gap BLOCKERS #16 already named; fingerprinting multi-part IR needs a
decision about which part (or a combined hash of all parts) is authoritative, which is part of
the VLMPipeline wiring, not a one-line fix alongside it.

Verified: ruff clean, `uv run pytest -q` exit 0, `uv lock --check` clean.

Next: VLMPipeline wiring in `answer/generate.py:load_generator` + multi-part-aware
`is_converted`/`ir_sha256` in `models/registry.py` — proceeding in an isolated worktree per the
developer, concurrently with the INT8 conversion running in this tree.

## 2026-08-25 23:50 — models/manifest.json: gemma-4-e2b-it INT8 also converts clean

`--only generator` against `configs/gemma4-e2b-int8.yaml` completed the same way the INT4 run
did: full multi-part IR in 847.8s, no `Reduce executor` failure, "warm" stage fails on the same
known gap (`is_converted` assumes single-file IR — being fixed in the concurrent worktree per
BLOCKERS #16). Manifest path separator held correct this time (`.as_posix()` fix from the
previous entry).

One consequence worth noting, not a bug: `models/manifest.json` keys entries by model name, not
by name+precision, so this run's manifest write **replaced** the INT4 record rather than adding
a second one — same behaviour every other model in this manifest already has (one active
precision at a time). Both `models/ir/gemma-4-e2b-it-int4/` (4.3 GB) and `.../int8/` (5.0 GB)
are still intact on disk; only the manifest's pointer moved. Whichever precision a config asks
for that isn't the manifest's current one will fail fast with the existing "config asks for X
but manifest holds Y" check in `models/registry.py`/`answer/generate.py` — re-running
`scripts.setup` for that precision re-points the manifest, cheaply, since the IR itself is
already on disk and conversion is what's slow.

Verified: ruff clean, `uv run pytest -q` exit 0, `uv lock --check` clean.

## 2026-08-26 00:15 — models/registry.py + answer/generate.py: BLOCKERS #16's two wiring gaps

Done, on branch `gemma-4`, in an isolated worktree with no real model weights (Windows
sandbox this time, not the M4 Pro BLOCKERS #1 describes — see the environment note below):

- `models/registry.py` — `ModelEntry` gained `is_vlm`, `vlm_language_model_xml`,
  `vlm_language_model_bin`. §7.3's `hf_vlm` kind (`convert_vlm`) writes a multi-part IR
  (`openvino_language_model.{xml,bin}` plus the vision/text-embedding towers) instead of
  the single `openvino_model.{xml,bin}` pair every other kind produces, so `is_converted`
  read a fully-converted `gemma-4-e2b-it` as not converted. `is_converted` and `ir_sha256`
  now treat the language-model pair as authoritative when `openvino_model.xml` is absent —
  detected from file presence, not a manifest field, so no `MANIFEST_SCHEMA_VERSION` bump
  and `registry.py` still does not import `models/convert.py`'s `SOURCES`/`Kind` table
  (asserted directly in `test_registry.py`).
- `answer/generate.py:load_generator` branches on `entry.is_vlm` and builds
  `genai.VLMPipeline` instead of `genai.LLMPipeline`. **Their constructors are not
  call-compatible** — confirmed against the installed `openvino_genai==2026.3.0` package
  directly (`VLMPipeline(models_path, "CPU", {"a": 1})` raises `TypeError: incompatible
  constructor arguments`, since the text-only overload is `(models_path, device,
  **kwargs)`, no positional `config`). `ov_config` is now passed as `**ov_config` for the
  VLM branch, positionally for `LLMPipeline` as before. `OpenVinoGenerator` is untouched;
  it only ever receives a pipeline instance.

**Not done, and written up in full in BLOCKERS #16 instead of guessed:** whether
`OpenVinoGenerator.stream`'s `self.pipe.generate(prompt, gen_cfg, streamer)` — three
positional arguments — is safe for `VLMPipeline`. Introspecting the real package shows
`VLMPipeline.generate` has nine overloads; every one taking `generation_config`/`streamer`
positionally also requires an `images`/`videos`/`image` argument first, and the one
pure-text overload (`generate(self, prompt, **kwargs)`) documents `generation_config` and
`streamer` as keyword-only. That suggests the current call would raise `TypeError` against
a real `VLMPipeline`, the same failure mode the constructor had. It could not be verified
live — no Gemma 4 weights exist in this worktree (BLOCKERS #16: NNCF can't build them on
arm64; this box can't either, it just has no weights at all), and probing an
uninitialized `VLMPipeline` instance's `.generate` beyond its docstring **segfaults the
interpreter**, so there is no safe way to exercise it here. Left `stream` byte-for-byte
unchanged per this task's own instruction not to force an unverified change onto the
Qwen3 path; BLOCKERS #16 carries the exact one-line fix to try (`generation_config=`,
`streamer=` as keywords — a no-op for `LLMPipeline`, whose only overload names those same
two parameters) and what to check before applying it.

**Environment note:** this worktree's sandbox is Windows (`win32`), not the Apple M4 Pro
BLOCKERS #1 describes — a different machine from the one the rest of this file narrates,
for this one delegated task. Two pre-existing, unrelated Windows-only failures showed up
running the full suite and are not touched: `tests/test_bench.py`, `test_dense.py`,
`test_eval_harness.py`, `test_sweep.py` fail to collect (`eval/run.py` and `eval/bench.py`
import the POSIX-only `resource` module); `tests/test_telemetry.py` (5 tests) and
`tests/test_ui_web.py` (1 test) fail on missing IANA tzdata and a non-UTF8 default file
encoding respectively. None of the five failing/non-collecting files were touched by this
change, and confirmed unchanged in git status throughout.

Verified: `uv run ruff format .` / `ruff check --fix .` clean. `uv run pytest
tests/test_registry.py tests/test_generate.py -q` -> **99 passed** (49 in
`test_registry.py`, 7 of them new multi-part-IR tests; 50 in `test_generate.py`, 4 of them
new `load_generator` tests — that file had *zero* `load_generator` tests before this
change). Full suite minus the five pre-existing/unrelated files above: all green except
the 6 listed above. `uv lock --check` clean.

Next: a live smoke test on the Intel laptop, once `gemma-4-e2b-it` is actually converted
there (BLOCKERS #16's three commands) — confirms or refutes the `.generate()` call-shape
finding, and is the only way to answer #17's open question (does Gemma 4 answer Tamil
better than Qwen3-4B). Nothing else in #16 is outstanding.
