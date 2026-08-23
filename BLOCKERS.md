# BLOCKERS

Things I could not resolve. Each says what I tried and what decision I need.

---

## 1. Dev hardware is Apple M4 Pro, not Intel i5 / Iris Xe — no GPU device exists

`ARCHITECTURE.md` targets "Intel Core i5 (11th gen, Iris Xe), 8–16 GB RAM". This machine is
an **Apple M4 Pro, arm64, 48 GB**. Measured, not assumed:

```
openvino 2026.3.0-22451  genai 2026.3.0.0-3277  tokenizers 2026.3.0.0
Core().available_devices == ['CPU']
CPU -> Apple M4 Pro
```

Good news: the pinned OpenVINO stack **installs and imports fine on macOS arm64 / Python 3.11**,
and all three packages are version-aligned. The runtime path is viable.

What breaks:
- `configs/base.yaml` asks for `device: GPU` on `ocr_det`, `ocr_rec`, `embedder`. On this box every
  one of those falls back to CPU. Mandatory device fallback (§7.4) covers it, but the config is
  aspirational here, not descriptive.
- **§8 Day 2 "CPU-vs-GPU comparison numbers" cannot be produced on this machine.** There is no
  second device to compare against. `eval/bench.py` can still emit TTFT / tok/s / peak RSS, but
  the CPU-vs-GPU slide needs the actual Intel laptop.
- Any timing from here is unrepresentative of the target: an M4 Pro is far faster than an i5 with
  Iris Xe, so tok/s and ingest latency measured now will flatter the pitch. **Do not put numbers
  from this machine on a slide claiming i5 performance.**

Tried: enumerating devices via `ov.Core()`; confirmed the GPU plugin is Intel-only, so there is
nothing to fall back *to* other than CPU.

**Need a decision:** (a) is the demo/pitch recorded on the Intel laptop, with this machine used
only for development? If so I keep `device: GPU` in `base.yaml` as target intent and rely on
fallback. Or (b) should `base.yaml` be switched to `CPU` and a separate `configs/intel.yaml`
hold the GPU intent? I have **proceeded with (a)** — config keeps GPU, fallback logs the real
device — because §6 says the file is the source of truth for intent and §7.4 makes fallback
mandatory. Not blocking any code.

---

## 2. `ARCHITECTURE.md` §2 lost its comment alignment to `ruff format`, and I cannot restore it

`uv run ruff format .` rewrites Python code blocks **inside Markdown files**. It reformatted the
fenced `python` schema block in §2: column-aligned trailing comments were collapsed to a single
two-space gap, and the single blank lines between dataclasses became doubled.

**The content is semantically identical — only whitespace inside one fenced code block changed.**
No field, type, or comment text was altered.

This violates prime directive 5 ("never edit `ARCHITECTURE.md` by hand"), so I tried to restore
it byte-for-byte. Both routes are blocked, correctly: `ARCHITECTURE.md` is outside the sandbox
write allowlist, and the unsandboxed restore was denied by the permission classifier. I did not
attempt to circumvent either guard.

Prevented from recurring: `pyproject.toml` now sets
`[tool.ruff] extend-exclude = ["*.md", "data", ".ov_cache"]`, so `ruff format .` no longer
touches Markdown. Verified — a subsequent `ruff format .` reports "18 files left unchanged".

**Need:** run the restore yourself, or grant write access to `ARCHITECTURE.md` once. The exact
original block is recoverable from the top of this session's transcript, or just re-align the
comments by hand — it is cosmetic. Nothing depends on it; `core/schema.py` was written from the
original text before the reformat.

---

## 3. `eval/golden.jsonl` is a labelled placeholder, not a gold set

§5 wants 40–60 real questions (10 cross-lingual, 5 unanswerable, 5 table/caption). Real ones need
the actual corpus, which does not exist in the repo yet. Fabricating 50 plausible-looking
questions against imaginary documents would produce eval numbers that look real and mean nothing —
worse than no numbers, given §0.5.

So `eval/golden.jsonl` currently holds **12 entries, every one tagged `"note": "PLACEHOLDER"`** and
pointing at `PLACEHOLDER_DOC_A` / `PLACEHOLDER_DOC_B`. They exercise every code path in the
harness (cross-lingual, unanswerable, caption) and nothing else. Grep `PLACEHOLDER` to find them.

**Need:** the study PDFs (and which three languages are final). Once a corpus is indexed I can
draft the real 40–60 against actual page numbers.

---

## 4. Sandbox cannot write `data/` from Bash

`data/` is not in the Bash write allowlist, so `mkdir data` fails with "Operation not permitted";
`models/` had the same problem. Created both via the file-write tool instead, which is permitted.

Impact is small but will hit the ingest work: anything running under sandboxed Bash that writes
to `data/cache/…` or `data/index/…` will fail. Mitigation already in place — `paths.data_dir` is
a config key, and `eval/run.py` takes `--out-dir`, so tests use pytest's `tmp_path` and never
touch the real `data/`. That is better design anyway.

**Need (optional):** add `data/` to the sandbox write allowlist via `/sandbox` if you want ingest
runs to work under the default sandbox rather than needing approval each time.

---

## 5. The Tamil corpus has a text layer, and it is mojibake — §3's text-layer rule is unsafe

Corpus fetched via `scripts/fetch_corpus.py` — 5 documents, **1,149 pages**, all from Internet
Archive. Every one reports a native text layer, so on §3's rule (*"If the PDF has a text layer,
extract it with pymupdf and skip OCR"*) **OCR would never run on any of them.**

For the English volumes that is fine — the text extracts cleanly. For the Tamil volumes it is not:

```
p61 of std12_cs_vol1_ta.pdf:
  "x ²ì¢® êó¤ò£ù Þìî¢î¤ô¢ Þ¼ï¢î£ô¢ Üï¢îê¢ ²ì¢® Þ¼î¬ô ªè£í¢ì Üñ¢¹è¢ °ø¤«ð£ô¢..."

any Tamil codepoints (U+0B80–U+0BFF)?  False
codepoints:                            0xb2 0xec 0xa2 0xae 0xea 0xf3   (all Latin-1)
fonts embedded:                        TAM_ELANGO_Abirami, TAMLKamban-Normal
our detect_script() verdict:           latn
```

These are legacy **TAB/TAM (Tamil ASCII)** fonts. The bytes are Latin-1 and only the embedded
font maps them to Tamil glyphs, so the page *displays* correct Tamil while `get_text()` returns
garbage. This is common in pre-Unicode Indic publishing and the 2006 state textbooks are full of it.

**Why this matters more than a bad document.** The failure is completely silent:

1. `ingest/load.py` sets `has_text_layer=True` → OCR skipped
2. `ingest/normalize.py` `detect_script()` returns `latn` for a Tamil book
3. `ingest/chunk.py` therefore counts whitespace tokens, not characters
4. BGE-M3 embeds mojibake, producing meaningless vectors
5. Tamil retrieval returns confident nonsense, and **every test still passes**

Nothing raises. This is the exact "looks like a chunking bug, costs you a day" failure mode
§3.1 warns about, arriving through the text layer instead of the embedder.

Tried: confirmed zero Tamil codepoints across sampled pages; confirmed the embedded font names;
confirmed our own `detect_script()` mislabels it. The English twin (`std12_cs_vol1_en.pdf`)
extracts correct text, so this is specific to the legacy-encoded Tamil, not to the fetch.

**Need a decision — §3 assumes a text layer is trustworthy, and for real multilingual scans it
is not.** Options, cheapest first:

- **(a) Validity check, not presence check.** Trust a text layer only if it decodes to codepoints
  in a plausible script — and force OCR when a document declares one language but its text layer
  contains none of that script. Small change to `ingest/load.py`, benefits every corpus.
- **(b) Per-document `force_ocr` override** in config. Cruder, but explicit and immediate.
- **(c) Transcode TSCII/TAB → Unicode.** Mapping tables exist, but this is real scope and §11
  is unforgiving.

I have **not** implemented any of them — this changes §3 behaviour, which is the contract's call,
not mine. **(a) is what I would build**, since it also protects against the general case of a
bad or partial OCR layer that someone else baked in.

**Silver lining, and it is a large one.** These documents render correct Tamil glyphs, so
rasterising a page and OCRing it produces real Tamil — which finally gives the OCR stage genuine
Tamil input. Better still, the *English* volume is a page-parallel twin of the Tamil one, so it
supplies near ground truth for the §5 cross-lingual questions without hand-labelling. And CLAUDE.md's
open Tamil-confidence question (0.4–0.6 on correct output) can at last be calibrated against real
pages rather than guessed.

**Blocked on:** nothing else. English (471pp) is usable for a baseline right now.
