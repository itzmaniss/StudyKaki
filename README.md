# StudyKaki

**Your study buddy for the books you already have.** Ask a question in English, Tamil or
Chinese; get an answer grounded in your own textbooks, with the page it came from. Nothing
ever leaves the laptop.

Built on OpenVINO — OCR, embeddings and generation all run locally on CPU/iGPU. There is no
API key, no account, and no network call at runtime. Turn the wifi off and it still works.

---

## Why it exists

A student's syllabus is a stack of PDFs, often not in English, often scanned. A general
chatbot will answer confidently from the open internet and quietly contradict the book they
are actually examined on. StudyKaki answers **only** from the documents you give it, cites the
page, and says "I couldn't find this in your documents" rather than guessing.

Three properties, in priority order:

1. **Grounded.** Every claim carries a `[n]` that resolves to a real page. Invented citations
   are stripped before the answer is shown, not after.
2. **Offline.** Model downloads happen once, in a setup script. Runtime code that reaches the
   network is a bug.
3. **Multilingual.** English, Tamil and Chinese, including cross-lingual — ask in English,
   retrieve from a Tamil textbook, and get the answer back in English.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (pinned — OpenVINO wheel coverage
lags on 3.12+).

```bash
uv sync                                   # dependencies, from the committed uv.lock
uv run python -m scripts.fetch_corpus     # the evaluation corpus (network)
uv run python -m scripts.setup            # download + convert models to OpenVINO IR (network)
uv run python -m ingest.pipeline data/corpus   # OCR -> chunk -> embed -> index
```

Then run it:

```bash
uv run python -m ui.web \
  --index data/index/<index_id> \
  --docs data/corpus/en --docs data/corpus/ta --docs data/corpus/zh
```

`--index` is required once more than one index exists; `--docs` is what makes citations
clickable. There is a terminal UI too: `uv run python -m ui.app`.

See **[DEMO.md](DEMO.md)** for the demo runbook — verified questions, expected answers, and
the launch gotchas.

## How it works

```
PDF ─ load ─┬─ text layer? ── skip OCR ──┐
            └─ PaddleOCR (per-script) ───┴─ normalize ─ chunk ─ embed ─ index
                                                                          │
question ─ embed ─ dense top-k ─ tau gate ─ top-n context ─ generate ─ cite.verify ─ answer
```

- **Ingest** is cached per stage on `(input_hash, stage_version, config_hash)`, so retuning a
  chunking parameter does not re-OCR the corpus.
- **Provenance** (`doc_id`, `page`, `bbox`) rides on every chunk from OCR onward. Citations
  depend on it and it cannot be retrofitted.
- **The index carries an embedder fingerprint.** Query-time and index-time embeddings must come
  from a byte-identical model; a mismatch hard-fails rather than silently degrading retrieval.
- **Abstain beats hallucinate.** Below `tau` no model runs at all.

Every model runs through OpenVINO, with a mandatory device fallback that logs which device it
actually got — the footer strip in the UI reports what loaded, not what the config asked for.

## Results

54 golden questions, 8 documents, 2,595 chunks, on an Intel Core Ultra 7 255H:

| | groundedness | en (lang match) | ta | zh | median generate |
|---|---|---|---|---|---|
| **gemma-4-e2b-it** int4 (default) | 0.796 | 0.758 (1.000) | 0.833 | 0.900 | **12.6 s** |
| qwen3-4b-instruct int4 | **0.935** | **0.935** (1.000) | **1.000** | 0.900 | 29.5 s |

Retrieval is identical either way: **recall@5 0.898**, **recall@10 0.980**, **MRR@10 0.743**,
**abstain precision 1.000**.

The default trades quality for speed and that trade is written into `configs/base.yaml` with
both rows and a one-line switch. Reproduce with:

```bash
uv run python -m eval.run --retriever dense --groundedness --index data/index/<index_id>
uv run python -m eval.bench      # TTFT, tok/s, peak RSS, per-stage timings
```

## Configuration

`configs/base.yaml` is the single source of truth — models, devices, precisions, chunking,
`tau`, and generation settings. Nothing tunable is hardcoded elsewhere. Both generator options
are listed there with their measured numbers, so switching is an informed two-line edit.

## Layout

```
core/      schema, config, cache, telemetry      ingest/    load, ocr, normalize, chunk, embed, index
models/    convert to OpenVINO IR, registry      retrieve/  dense, fusion, retriever
answer/    prompt, generate, cite                eval/      golden.jsonl, run, bench
ui/        app.py (terminal), web.py (browser)   scripts/   setup, fetch_corpus  ← the only network code
```

## Development

```bash
uv run ruff format . && uv run ruff check --fix .
uv run pytest -q
uv lock --check
```

Tests assert on structure, provenance and error handling — never on model output text, which
drifts. Fixtures stay under 5 MB; no full textbooks in the repo.

## Status and known limitations

V1. Working end to end and measured, with real gaps recorded rather than hidden:

- **Memory.** ~8.5 GB peak. `ARCHITECTURE.md` targets 8–16 GB, so this is tight on the low end
  and the generator is the floor.
- **Tamil is the weakest language.** `ta 0.833` groundedness on the default generator; some
  answers emit malformed citation markers and land ungrounded. `qwen3-4b-instruct` scores
  `ta 1.000` if Tamil matters more than latency.
- **Two tests fail on hardware with a working iGPU**, because `"GPU"` and `"GPU.0"` are
  different strings for one device. Tracked as BLOCKERS #22.
- **Never set `device: NPU` for the VLM generator.** It loads successfully and generates token
  salad; the load-failure fallback cannot catch it. BLOCKERS #20.

`ARCHITECTURE.md` is the contract. `BLOCKERS.md` holds every open question with what was tried
and what decision is needed; `PROGRESS.md` is the running build log.

## Out of scope

Handwriting OCR, PPTX ingestion, HyDE, agentic multi-hop, fine-tuning, installers, more than
three languages, and online document fetching. Interfaces exist so they can be added later.
