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

## 3. `eval/golden.jsonl` — RESOLVED for en/zh, Tamil questions still outstanding

**Was:** 12 entries tagged `"note": "PLACEHOLDER"` against imaginary documents, because no corpus
existed. Now `data/corpus/` holds eight real PDFs and the set holds **44 real questions**, every
answer read off the page it is labelled with before being written down. No PLACEHOLDER remains.

Current composition against §5's target (40–60 / 10 cross-lingual / 5 unanswerable / 5 table):

| | have | want |
|---|---|---|
| total | 44 | 40–60 |
| cross-lingual | 8 | 10 |
| unanswerable | 5 | 5 |
| table or figure | 7 | 5 |
| Tamil questions | 0 | — |

**Still needed:** the Tamil half. All three Tamil PDFs reach the index through OCR, so their page
content is not readable until the OCR pass finishes; questions written before then would be
labelled against guessed page numbers, which is the exact failure this blocker was raised for.
Once OCR lands, add ~6 Tamil-language questions and ~4 cross-lingual `en -> digital_electronics_ta`,
which also takes cross-lingual from 8 to the 10 §5 asks for.

**For the developer:** the labels are my reading of the pages, not verified against an answer key.
Worth a spot-check on a handful before trusting the absolute numbers — though V2 gating compares
before/after on the same labels, where a consistent label error largely cancels.

---

## 9. Parallel translations make a single `doc_id` label ambiguous

`GoldQuestion.doc_id` is one document, and `eval/metrics.py:is_relevant` requires
`hit.chunk.doc_id == gold.doc_id`. But four of the eight corpus documents are parallel
translations of the other four: `std12_cs_vol1_{ta,en}`, `std12_cs_vol2_{ta,en}`, and
`itu_wtdc22_{zh,en}` (both 664 pages). A question answerable from either member of a pair has two
correct answers and one label, so a retriever that returns the translated twin scores 0 for
being right in the wrong language.

**RESOLVED** — `GoldQuestion` takes an optional `alt_doc_ids`. §5's shape is untouched: a row with
only `doc_id` behaves exactly as before. Two forms:

```json
{"q": "...", "doc_id": "<primary>", "gold_pages": [42], "alt_doc_ids": ["<twin>"]}
{"q": "...", "doc_id": "<primary>", "gold_pages": [42], "alt_doc_ids": {"<twin>": [45, 46]}}
```

The list form shares `gold_pages` with the twin. The object form gives the twin its own pages, and
is **required where the editions drift**. Measured over the corpus by comparing the numeric tokens
on each page (digits survive the Tamil mojibake, so this works without OCR):

| pair | best offset | agreement | use |
|---|---|---|---|
| `itu_wtdc22_en` ↔ `itu_wtdc22_zh` | 0 | 0.950 | list form |
| `std12_cs_vol2_en` ↔ `std12_cs_vol2_ta` | 0 | 0.780 | list form |
| `std12_cs_vol1_en` ↔ `std12_cs_vol1_ta` | 0 | **0.358** | **object form** — 231pp vs 232pp, drifts to +3 by the back |

An unanswerable question may not carry `alt_doc_ids` (empty `gold_pages` means abstain, so no
document answers it); that and duplicate/malformed entries raise rather than parse loosely.

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

### #5 RESOLVED — text-layer validity check implemented (approved option (a))

`ingest/load.py` now checks whether a text layer *decodes*, not merely whether it exists.
Two independent signals, both measured against the real corpus:

| Signal | Catches | Threshold |
|---|---|---|
| `legacy_encoding_ratio` | wholly mis-decoded 8-bit text (TSCII/TAB, CP1252) | > 0.25 rejects |
| `mixed_script_token_ratio` | *partially* decoded layers — Latin fused inside words | > 0.15 rejects |

Measured (50-page samples):

```
Tamil vol1 (TSCII)      legacy=0.724 fused=0.000 -> REJECTED, OCR
Tamil vol2 (TSCII)      legacy=0.555 fused=0.000 -> REJECTED, OCR
DigitalElec (partial)   legacy=0.000 fused=0.304 -> REJECTED, OCR
English vol1/vol2       legacy≤0.009 fused=0.000 -> trusted
CNNIC / ITU zh / ITU en legacy≤0.002 fused=0.000 -> trusted
```

Two findings worth recording:

1. **A codepoint floor is the wrong test.** The first implementation trusted any text containing
   a codepoint above U+0590. Mis-decoded CP1252 emits `™` (U+2122) and `›` (U+203A), which sit
   *above* that floor, so mojibake read as proof of a real script — exactly backwards. Real
   script blocks are now enumerated explicitly.
2. **`digital_electronics_ta.pdf` is a third failure mode**, not caught by the ratio test at all:
   the layer decodes *partially*, giving real Tamil codepoints with Latin fall-through fused
   inside words (`எzகள}`). The ratio test abstains because real script is present. The
   fused-token signal catches it at 0.304.

**The fused-token test is deliberately restricted to space-delimited scripts.** CJK and Thai are
not written with spaces, so an ordinary CNNIC line like `我国IPv6地址数量` is one token and
would read as fused. Without that exemption the entire Chinese corpus is rejected wholesale —
verified as a test.

Threshold caveat: the fused-token control sample is small (hand-built Tamil with English
technical terms, measuring 0.0) against one real document at 0.304. 0.15 sits well clear of
both rather than splitting the difference, but it deserves re-checking once real Tamil OCR
output exists to compare against.

---

## 6. §6 names two models that cannot be used as written

**`PP-OCRv6_mobile_det` does not exist.** `configs/base.yaml` in §6 names it, but
`PaddlePaddle/PP-OCRv6_*` returns 401 on the HF API — there is no public v6 checkpoint. v5
mobile is the newest released and is what PaddleOCR itself ships. `configs/base.yaml` now
reads `PP-OCRv5_mobile_det`, with the reason inline. §6 names `PP-OCRv5_mobile_rec` for the
recogniser already, so only the detector deviates.

**PaddlePaddle 3.0 broke OpenVINO's Paddle frontend.** §7 assumes OpenVINO reads Paddle
inference models natively. It no longer can: every PaddlePaddle HF repo — v4 mobile as well
as v5 — now ships a PIR program (`inference.json`, `{"magic":"pir"}`) instead of the legacy
protobuf `.pdmodel`. OpenVINO 2026.3 fails on it with the unhelpful `Cannot recognize input
model.` There is no `.pdmodel` left to fall back to.

Resolved by routing PIR through Paddle's own exporter: **PIR → ONNX → OpenVINO IR → INT8**.
The alternative was a community ONNX re-upload from HF; several exist, all with 0 downloads
and no provenance, and §3.1 makes model identity a hard requirement, so the official
converter won despite its cost.

That cost is three new dependencies — `paddle2onnx`, `paddlepaddle`, `setuptools`
(paddle2onnx imports it at runtime and does not declare it). They are **conversion-time only**,
like `torch`, `nncf` and `optimum-intel` already are; nothing under `ingest/`, `retrieve/` or
`answer/` imports them. `paddlepaddle` is ~100 MB, so `uv sync` is heavier now. If that
matters for the Intel laptop, moving all conversion-time deps into a `[dependency-groups]
convert` group is a clean follow-up — I did not do it because §0.1 keeps `nncf` and
`optimum-intel` in the main dependency list and I matched that convention.

Verified working: both models convert, load through `models/registry.py`, and fall back to
CPU with the right shapes — det `[?,3,?,?] -> [?,1,32..,32..]`, rec `[?,3,48,?] -> [?,1..,18385]`.
The 18385-class head is PP-OCRv5's multilingual charset, which is what Tamil needs.

---

## 7. Tamil OCR confidence is 0.90-0.95, not 0.4-0.6 — CLAUDE.md's calibration note was a wrong-head artefact

CLAUDE.md records from a prior run: *"Tamil recognition returns confidence 0.4-0.6 on clean
scans where output is visibly correct; Latin on the same page returns 0.9+"*, and asks for a
per-script threshold decision or 20 labelled pages.

**That question is now answered, and the premise was wrong.** With a dedicated Tamil head
(`ta_PP-OCRv5_mobile_rec`) the same pages read at **0.90-0.95**. Measured on p61 of
`std12_cs_vol1_ta.pdf`:

```
script_hint=None    35 blocks,  0 contain Tamil   e.g. conf=0.67 '@uπg @1 6migu (margin guide) u @lgl'
script_hint='taml'  47 blocks, 38 contain Tamil   e.g. conf=0.93 'சுட்டி சரியானஇடத்தில்இருந்தால் அந்தச் சுட்டி'
```

The 0.4-0.6 band was the **Chinese+English head guessing at Tamil glyphs**, not Tamil being
intrinsically hard to read. Cross-check: the recovered line is the same sentence the mojibake
text layer encodes as `²ì¢® êó¤ò£ù Þìî¢î¤ô¢ Þ¼ï¢î£ô¢ Üï¢îê¢ ²ì¢®`, so the OCR is
independently corroborated by the broken layer it replaced.

**No per-script thresholds are needed.** `confidence_by_script` stays empty and
`min_confidence=0.30` stands, now with evidence rather than as a hedge. Do not add per-script
numbers to work around a problem that was a routing bug.

### Routing needed a hint — text-based head selection cannot bootstrap

`_recognize` chose a dedicated head by running `detect_script` on what the *default* head
produced. That only works if the default head can represent the script at all. It cannot for
Tamil, so `detect_script` saw CJK noise, said `hans`, and never reached the Tamil head — 0 of
35 blocks Tamil, every one confidently wrong. `script_hint` now lets the caller name the
script; it flows through `ocr_page`/`ocr_pages`/`Pipeline` and is part of the OCR cache key,
since two hints are two different results for identical pixels.

---

## 8. `tau` — RESOLVED at 0.45, but the safe window is 0.01 wide

Recalibrated 2026-08-24 against the full 8-document index and the **54**-question golden set.

| | min | median | max |
|---|---|---|---|
| answerable (49) | 0.460 | 0.647 | 0.783 |
| unanswerable (5) | 0.443 | 0.561 | 0.692 |

**The ranges overlap almost completely.** tau is a trade, never a correct value.

| tau | abstains | correct | answerable wrongly declined |
|---|---|---|---|
| 0.35 (original) | 0 | 0/5 | 0 |
| **0.45 (now)** | **2** | **2/5** | **0** |
| 0.47 | 4 | 2/5 | 2 |
| 0.55 | 7 | 2/5 | 5 |

**What changed and why it matters.** At 44 questions the safe window was `[0.45, 0.52]` and 0.50
looked comfortable. Adding ten Tamil questions collapsed it to `[0.45, 0.46]` — 0.50 now wrongly
declines two answerable questions, and `abstain_precision` fell from 1.000 to 0.500 until it was
retuned. The cause is visible in the per-language scores:

| query -> document | top-1 cosine |
|---|---|
| zh -> zh | 0.731–0.783 |
| ta -> ta | 0.570–0.644 |
| **en -> ta (cross-lingual, OCR text)** | **0.460–0.542** |

The hardest-won answers score lowest, so they sit directly on top of the abstain noise floor. Any
threshold tuned without cross-lingual questions in the set will be too aggressive. **A single
global cosine threshold is close to the end of its usefulness here** — the honest fix is either a
per-script threshold or letting the generator decline (§9), and the latter is measurable today
with `--groundedness`.

Three of five unanswerable questions still score above the answerable minimum and cannot be
thresholded out at all: they are phrased in the corpus's own vocabulary ("install Microsoft Excel"
0.570 against a StarOffice textbook, "forecast 2025 netizen numbers" 0.692 against a report that
stops at 2021).

---

## 9. Parallel translations make a single `doc_id` label ambiguous

`GoldQuestion.doc_id` is one document, and `eval/metrics.py:is_relevant` requires
`hit.chunk.doc_id == gold.doc_id`. But four of the eight corpus documents are parallel
translations of the other four: `std12_cs_vol1_{ta,en}`, `std12_cs_vol2_{ta,en}`, and
`itu_wtdc22_{zh,en}` (both 664 pages). A question answerable from either member of a pair has two
correct answers and one label, so a retriever that returns the translated twin scores 0 for
being right in the wrong language.

**RESOLVED** — `GoldQuestion` takes an optional `alt_doc_ids`. §5's shape is untouched: a row with
only `doc_id` behaves exactly as before. Two forms:

```json
{"q": "...", "doc_id": "<primary>", "gold_pages": [42], "alt_doc_ids": ["<twin>"]}
{"q": "...", "doc_id": "<primary>", "gold_pages": [42], "alt_doc_ids": {"<twin>": [45, 46]}}
```

The list form shares `gold_pages` with the twin. The object form gives the twin its own pages, and
is **required where the editions drift**. Measured over the corpus by comparing the numeric tokens
on each page (digits survive the Tamil mojibake, so this works without OCR):

| pair | best offset | agreement | use |
|---|---|---|---|
| `itu_wtdc22_en` ↔ `itu_wtdc22_zh` | 0 | 0.950 | list form |
| `std12_cs_vol2_en` ↔ `std12_cs_vol2_ta` | 0 | 0.780 | list form |
| `std12_cs_vol1_en` ↔ `std12_cs_vol1_ta` | 0 | **0.358** | **object form** — 231pp vs 232pp, drifts to +3 by the back |

An unanswerable question may not carry `alt_doc_ids` (empty `gold_pages` means abstain, so no
document answers it); that and duplicate/malformed entries raise rather than parse loosely.

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

### #5 RESOLVED — text-layer validity check implemented (approved option (a))

`ingest/load.py` now checks whether a text layer *decodes*, not merely whether it exists.
Two independent signals, both measured against the real corpus:

| Signal | Catches | Threshold |
|---|---|---|
| `legacy_encoding_ratio` | wholly mis-decoded 8-bit text (TSCII/TAB, CP1252) | > 0.25 rejects |
| `mixed_script_token_ratio` | *partially* decoded layers — Latin fused inside words | > 0.15 rejects |

Measured (50-page samples):

```
Tamil vol1 (TSCII)      legacy=0.724 fused=0.000 -> REJECTED, OCR
Tamil vol2 (TSCII)      legacy=0.555 fused=0.000 -> REJECTED, OCR
DigitalElec (partial)   legacy=0.000 fused=0.304 -> REJECTED, OCR
English vol1/vol2       legacy≤0.009 fused=0.000 -> trusted
CNNIC / ITU zh / ITU en legacy≤0.002 fused=0.000 -> trusted
```

Two findings worth recording:

1. **A codepoint floor is the wrong test.** The first implementation trusted any text containing
   a codepoint above U+0590. Mis-decoded CP1252 emits `™` (U+2122) and `›` (U+203A), which sit
   *above* that floor, so mojibake read as proof of a real script — exactly backwards. Real
   script blocks are now enumerated explicitly.
2. **`digital_electronics_ta.pdf` is a third failure mode**, not caught by the ratio test at all:
   the layer decodes *partially*, giving real Tamil codepoints with Latin fall-through fused
   inside words (`எzகள}`). The ratio test abstains because real script is present. The
   fused-token signal catches it at 0.304.

**The fused-token test is deliberately restricted to space-delimited scripts.** CJK and Thai are
not written with spaces, so an ordinary CNNIC line like `我国IPv6地址数量` is one token and
would read as fused. Without that exemption the entire Chinese corpus is rejected wholesale —
verified as a test.

Threshold caveat: the fused-token control sample is small (hand-built Tamil with English
technical terms, measuring 0.0) against one real document at 0.304. 0.15 sits well clear of
both rather than splitting the difference, but it deserves re-checking once real Tamil OCR
output exists to compare against.

---

## 6. §6 names two models that cannot be used as written

**`PP-OCRv6_mobile_det` does not exist.** `configs/base.yaml` in §6 names it, but
`PaddlePaddle/PP-OCRv6_*` returns 401 on the HF API — there is no public v6 checkpoint. v5
mobile is the newest released and is what PaddleOCR itself ships. `configs/base.yaml` now
reads `PP-OCRv5_mobile_det`, with the reason inline. §6 names `PP-OCRv5_mobile_rec` for the
recogniser already, so only the detector deviates.

**PaddlePaddle 3.0 broke OpenVINO's Paddle frontend.** §7 assumes OpenVINO reads Paddle
inference models natively. It no longer can: every PaddlePaddle HF repo — v4 mobile as well
as v5 — now ships a PIR program (`inference.json`, `{"magic":"pir"}`) instead of the legacy
protobuf `.pdmodel`. OpenVINO 2026.3 fails on it with the unhelpful `Cannot recognize input
model.` There is no `.pdmodel` left to fall back to.

Resolved by routing PIR through Paddle's own exporter: **PIR → ONNX → OpenVINO IR → INT8**.
The alternative was a community ONNX re-upload from HF; several exist, all with 0 downloads
and no provenance, and §3.1 makes model identity a hard requirement, so the official
converter won despite its cost.

That cost is three new dependencies — `paddle2onnx`, `paddlepaddle`, `setuptools`
(paddle2onnx imports it at runtime and does not declare it). They are **conversion-time only**,
like `torch`, `nncf` and `optimum-intel` already are; nothing under `ingest/`, `retrieve/` or
`answer/` imports them. `paddlepaddle` is ~100 MB, so `uv sync` is heavier now. If that
matters for the Intel laptop, moving all conversion-time deps into a `[dependency-groups]
convert` group is a clean follow-up — I did not do it because §0.1 keeps `nncf` and
`optimum-intel` in the main dependency list and I matched that convention.

Verified working: both models convert, load through `models/registry.py`, and fall back to
CPU with the right shapes — det `[?,3,?,?] -> [?,1,32..,32..]`, rec `[?,3,48,?] -> [?,1..,18385]`.
The 18385-class head is PP-OCRv5's multilingual charset, which is what Tamil needs.

---

## 7. Tamil OCR confidence is 0.90-0.95, not 0.4-0.6 — CLAUDE.md's calibration note was a wrong-head artefact

CLAUDE.md records from a prior run: *"Tamil recognition returns confidence 0.4-0.6 on clean
scans where output is visibly correct; Latin on the same page returns 0.9+"*, and asks for a
per-script threshold decision or 20 labelled pages.

**That question is now answered, and the premise was wrong.** With a dedicated Tamil head
(`ta_PP-OCRv5_mobile_rec`) the same pages read at **0.90-0.95**. Measured on p61 of
`std12_cs_vol1_ta.pdf`:

```
script_hint=None    35 blocks,  0 contain Tamil   e.g. conf=0.67 '@uπg @1 6migu (margin guide) u @lgl'
script_hint='taml'  47 blocks, 38 contain Tamil   e.g. conf=0.93 'சுட்டி சரியானஇடத்தில்இருந்தால் அந்தச் சுட்டி'
```

The 0.4-0.6 band was the **Chinese+English head guessing at Tamil glyphs**, not Tamil being
intrinsically hard to read. Cross-check: the recovered line is the same sentence the mojibake
text layer encodes as `²ì¢® êó¤ò£ù Þìî¢î¤ô¢ Þ¼ï¢î£ô¢ Üï¢îê¢ ²ì¢®`, so the OCR is
independently corroborated by the broken layer it replaced.

**No per-script thresholds are needed.** `confidence_by_script` stays empty and
`min_confidence=0.30` stands, now with evidence rather than as a hedge. Do not add per-script
numbers to work around a problem that was a routing bug.

### Routing needed a hint — text-based head selection cannot bootstrap

`_recognize` chose a dedicated head by running `detect_script` on what the *default* head
produced. That only works if the default head can represent the script at all. It cannot for
Tamil, so `detect_script` saw CJK noise, said `hans`, and never reached the Tamil head — 0 of
35 blocks Tamil, every one confidently wrong. `script_hint` now lets the caller name the
script; it flows through `ocr_page`/`ocr_pages`/`Pipeline` and is part of the OCR cache key,
since two hints are two different results for identical pixels.

---

## 8. `tau` — RESOLVED at 0.50, but the score distributions overlap

Calibrated 2026-08-24 against the full 8-document index and the 44-question golden set.
Measured cosine top-1:

| | min | median | max |
|---|---|---|---|
| answerable (39) | 0.547 | 0.660 | 0.783 |
| unanswerable (5) | 0.443 | 0.561 | 0.692 |

**The ranges overlap (0.547–0.692), so no threshold separates them.** tau is a trade, not a
setting that can be made correct. Sweep:

| tau | abstains | correct | answerable wrongly declined |
|---|---|---|---|
| 0.35 (was) | 0 | 0/5 | 0 |
| **0.50 (now)** | **2** | **2/5** | **0** |
| 0.60 | 10 | 4/5 | 6 |
| 0.70 | 30 | 5/5 | 25 |

0.50 is the best free operating point: two correct abstentions at zero cost. 0.60 catches two
more but wrongly declines six answerable questions, which is the wrong trade for a study aid.
Retrieval metrics are unaffected — recall is scored before the abstain decision.

**Still open, and it is the interesting one:** three of the five unanswerable questions score
*above* the lowest answerable question, so thresholding a single cosine cannot fix them. The
two that get caught are the ones about topics absent from the corpus entirely (2026 World Cup,
Bitcoin price). The three that slip through are phrased in the corpus's own vocabulary —
"install Microsoft Excel" (0.570) against a StarOffice textbook, and a request to forecast 2025
netizen numbers (0.692) against a report that stops at 2021. Those need the *generator* to
decline, not the retriever. §9's abstain path exists for exactly this; worth measuring
`--groundedness` before assuming reranking would help.

---

## 9. Parallel translations make a single `doc_id` label ambiguous

`GoldQuestion.doc_id` is one document, and `eval/metrics.py:is_relevant` requires
`hit.chunk.doc_id == gold.doc_id`. But four of the eight corpus documents are parallel
translations of the other four: `std12_cs_vol1_{ta,en}`, `std12_cs_vol2_{ta,en}`, and
`itu_wtdc22_{zh,en}` (both 664 pages). A question answerable from either member of a pair has two
correct answers and one label, so a retriever that returns the translated twin scores 0 for
being right in the wrong language.

**RESOLVED** — `GoldQuestion` takes an optional `alt_doc_ids`. §5's shape is untouched: a row with
only `doc_id` behaves exactly as before. Two forms:

```json
{"q": "...", "doc_id": "<primary>", "gold_pages": [42], "alt_doc_ids": ["<twin>"]}
{"q": "...", "doc_id": "<primary>", "gold_pages": [42], "alt_doc_ids": {"<twin>": [45, 46]}}
```

The list form shares `gold_pages` with the twin. The object form gives the twin its own pages, and
is **required where the editions drift**. Measured over the corpus by comparing the numeric tokens
on each page (digits survive the Tamil mojibake, so this works without OCR):

| pair | best offset | agreement | use |
|---|---|---|---|
| `itu_wtdc22_en` ↔ `itu_wtdc22_zh` | 0 | 0.950 | list form |
| `std12_cs_vol2_en` ↔ `std12_cs_vol2_ta` | 0 | 0.780 | list form |
| `std12_cs_vol1_en` ↔ `std12_cs_vol1_ta` | 0 | **0.358** | **object form** — 231pp vs 232pp, drifts to +3 by the back |

An unanswerable question may not carry `alt_doc_ids` (empty `gold_pages` means abstain, so no
document answers it); that and duplicate/malformed entries raise rather than parse loosely.

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

### #5 RESOLVED — text-layer validity check implemented (approved option (a))

`ingest/load.py` now checks whether a text layer *decodes*, not merely whether it exists.
Two independent signals, both measured against the real corpus:

| Signal | Catches | Threshold |
|---|---|---|
| `legacy_encoding_ratio` | wholly mis-decoded 8-bit text (TSCII/TAB, CP1252) | > 0.25 rejects |
| `mixed_script_token_ratio` | *partially* decoded layers — Latin fused inside words | > 0.15 rejects |

Measured (50-page samples):

```
Tamil vol1 (TSCII)      legacy=0.724 fused=0.000 -> REJECTED, OCR
Tamil vol2 (TSCII)      legacy=0.555 fused=0.000 -> REJECTED, OCR
DigitalElec (partial)   legacy=0.000 fused=0.304 -> REJECTED, OCR
English vol1/vol2       legacy≤0.009 fused=0.000 -> trusted
CNNIC / ITU zh / ITU en legacy≤0.002 fused=0.000 -> trusted
```

Two findings worth recording:

1. **A codepoint floor is the wrong test.** The first implementation trusted any text containing
   a codepoint above U+0590. Mis-decoded CP1252 emits `™` (U+2122) and `›` (U+203A), which sit
   *above* that floor, so mojibake read as proof of a real script — exactly backwards. Real
   script blocks are now enumerated explicitly.
2. **`digital_electronics_ta.pdf` is a third failure mode**, not caught by the ratio test at all:
   the layer decodes *partially*, giving real Tamil codepoints with Latin fall-through fused
   inside words (`எzகள}`). The ratio test abstains because real script is present. The
   fused-token signal catches it at 0.304.

**The fused-token test is deliberately restricted to space-delimited scripts.** CJK and Thai are
not written with spaces, so an ordinary CNNIC line like `我国IPv6地址数量` is one token and
would read as fused. Without that exemption the entire Chinese corpus is rejected wholesale —
verified as a test.

Threshold caveat: the fused-token control sample is small (hand-built Tamil with English
technical terms, measuring 0.0) against one real document at 0.304. 0.15 sits well clear of
both rather than splitting the difference, but it deserves re-checking once real Tamil OCR
output exists to compare against.

---

## 6. §6 names two models that cannot be used as written

**`PP-OCRv6_mobile_det` does not exist.** `configs/base.yaml` in §6 names it, but
`PaddlePaddle/PP-OCRv6_*` returns 401 on the HF API — there is no public v6 checkpoint. v5
mobile is the newest released and is what PaddleOCR itself ships. `configs/base.yaml` now
reads `PP-OCRv5_mobile_det`, with the reason inline. §6 names `PP-OCRv5_mobile_rec` for the
recogniser already, so only the detector deviates.

**PaddlePaddle 3.0 broke OpenVINO's Paddle frontend.** §7 assumes OpenVINO reads Paddle
inference models natively. It no longer can: every PaddlePaddle HF repo — v4 mobile as well
as v5 — now ships a PIR program (`inference.json`, `{"magic":"pir"}`) instead of the legacy
protobuf `.pdmodel`. OpenVINO 2026.3 fails on it with the unhelpful `Cannot recognize input
model.` There is no `.pdmodel` left to fall back to.

Resolved by routing PIR through Paddle's own exporter: **PIR → ONNX → OpenVINO IR → INT8**.
The alternative was a community ONNX re-upload from HF; several exist, all with 0 downloads
and no provenance, and §3.1 makes model identity a hard requirement, so the official
converter won despite its cost.

That cost is three new dependencies — `paddle2onnx`, `paddlepaddle`, `setuptools`
(paddle2onnx imports it at runtime and does not declare it). They are **conversion-time only**,
like `torch`, `nncf` and `optimum-intel` already are; nothing under `ingest/`, `retrieve/` or
`answer/` imports them. `paddlepaddle` is ~100 MB, so `uv sync` is heavier now. If that
matters for the Intel laptop, moving all conversion-time deps into a `[dependency-groups]
convert` group is a clean follow-up — I did not do it because §0.1 keeps `nncf` and
`optimum-intel` in the main dependency list and I matched that convention.

Verified working: both models convert, load through `models/registry.py`, and fall back to
CPU with the right shapes — det `[?,3,?,?] -> [?,1,32..,32..]`, rec `[?,3,48,?] -> [?,1..,18385]`.
The 18385-class head is PP-OCRv5's multilingual charset, which is what Tamil needs.

---

## 7. Tamil OCR confidence is 0.90-0.95, not 0.4-0.6 — CLAUDE.md's calibration note was a wrong-head artefact

CLAUDE.md records from a prior run: *"Tamil recognition returns confidence 0.4-0.6 on clean
scans where output is visibly correct; Latin on the same page returns 0.9+"*, and asks for a
per-script threshold decision or 20 labelled pages.

**That question is now answered, and the premise was wrong.** With a dedicated Tamil head
(`ta_PP-OCRv5_mobile_rec`) the same pages read at **0.90-0.95**. Measured on p61 of
`std12_cs_vol1_ta.pdf`:

```
script_hint=None    35 blocks,  0 contain Tamil   e.g. conf=0.67 '@uπg @1 6migu (margin guide) u @lgl'
script_hint='taml'  47 blocks, 38 contain Tamil   e.g. conf=0.93 'சுட்டி சரியானஇடத்தில்இருந்தால் அந்தச் சுட்டி'
```

The 0.4-0.6 band was the **Chinese+English head guessing at Tamil glyphs**, not Tamil being
intrinsically hard to read. Cross-check: the recovered line is the same sentence the mojibake
text layer encodes as `²ì¢® êó¤ò£ù Þìî¢î¤ô¢ Þ¼ï¢î£ô¢ Üï¢îê¢ ²ì¢®`, so the OCR is
independently corroborated by the broken layer it replaced.

**No per-script thresholds are needed.** `confidence_by_script` stays empty and
`min_confidence=0.30` stands, now with evidence rather than as a hedge. Do not add per-script
numbers to work around a problem that was a routing bug.

### Routing needed a hint — text-based head selection cannot bootstrap

`_recognize` chose a dedicated head by running `detect_script` on what the *default* head
produced. That only works if the default head can represent the script at all. It cannot for
Tamil, so `detect_script` saw CJK noise, said `hans`, and never reached the Tamil head — 0 of
35 blocks Tamil, every one confidently wrong. `script_hint` now lets the caller name the
script; it flows through `ocr_page`/`ocr_pages`/`Pipeline` and is part of the OCR cache key,
since two hints are two different results for identical pixels.

---

## 8. `tau = 0.35` does not abstain on this corpus

§6 sets `retrieve.tau: 0.35`. On 15 OCR'd Tamil pages (52 chunks):

```
Q: "What colour is the table background?"  top=0.767  -> correct Tamil chunk (cross-lingual, works)
Q: "How do I file my tax return?"          top=0.475  -> DID NOT ABSTAIN
```

A tax question against a computer-science textbook must abstain, and 0.475 clears 0.35
comfortably. On the earlier synthetic corpus the equivalent out-of-domain question scored
0.269 and abstained correctly, so this is corpus-dependent, not a code fault: BGE-M3 cosine
scores sit higher against a small, noisy, OCR'd corpus than against clean synthetic text.

**Not fixing this by guessing a new number.** §0.5 says no feature ships without an eval
number, and `tau` is exactly what the 5 unanswerable golden questions exist to calibrate
(§5). Raising it blind would trade false answers for false abstentions with nothing to show
which is worse.

**Need:** the golden set. Once 40-60 questions exist against this corpus, `abstain_precision`
picks `tau` directly. Until then the demo will answer questions it should decline.

---

## 10. Two golden labels were wrong, and the way I built them explains how

Found by the developer reviewing the check sheet, 2026-08-24. Both replaced.

| was | why it failed |
|---|---|
| "What is the difference between hard formatting and soft formatting?" `vol1_en` p[39,43] | Lifted from the book's own *"III. Answer the following"* exercise list on p43. The phrase occurs exactly twice in the whole volume — p41 says two types exist, p43 asks the question — and it is **never defined**. The page numbers were wrong too: the bullet is on p41, not p39. |
| "What does the address-of operator & do?" `vol2_en` p[48] | p48 carries the heading *"The '&' operator:"* but the prose under it describes what happens when you declare `int num1=10`, not what `&` does. The program below prints `&i`, so it is inferable, but the page never states it. |

Replaced with facts the page states outright: default margins (`vol1_en` p59) and the seven kinds
of basic statement (`vol2_en` p62). Both retrieve at rank 1.

**Root cause, which matters more than the two labels.** I wrote the set from a 300-character
per-page digest. That is enough to see what a page is *about*, and not enough to confirm it
*answers* a question — so a page discussing formatting styles looked like it covered hard vs soft
formatting. A page-level topic match is not an answer.

**The failure was silent.** Both questions *passed* before the fix: they scored recall@1 = 1.0
against pages that do not contain the answer. Correcting them did not move a single headline
number. A bad label does not show up as a bad score, which is exactly why the numbers cannot
validate the labels and a human review was the only way to catch this.

Rate: 2 of 44 (4.5%) failed review. A sweep for the same error class — gold pages dominated by
`Exercises` / `Answer the following` / `Fill in the blanks` furniture — found only these, so it is
not believed to be systemic. The remaining 42 have not been independently checked.

**Available if wanted:** the hard/soft formatting question is an unusually good *unanswerable*
candidate — the exact terms appear in the corpus, so retrieval scores high, but nothing answers
it. That is the failure mode #8 is still open on. Not added, because whether p41's "there are two
types" counts as an answer is arguable, and the 5 existing unanswerables are unambiguous.

---

## 11. The Tamil corpus is unanswerable end-to-end: prompt overflows OpenVINO's CPU MatMul

Found 2026-08-24 running `eval/run.py --groundedness`. It failed on question 45 — the **first
Tamil question** — with:

```
GenerationError: qwen3-4b-instruct: generation failed on CPU:
[CPU] MatMul node 'Multiply_156855' could not create a primitive descriptor for the matmul primitive
```

Not a model context limit; Qwen3-4B-Instruct-2507 advertises far more. It is an OpenVINO INT4
CPU limit on this build, and **Tamil reaches it first because Tamil tokenizes roughly twice as
densely as English**:

| language | chars/token | prompt chars (5 chunks) | prompt tokens | generates? |
|---|---|---|---|---|
| zh | 1.62 | 3,303 | 2,042 | yes |
| en | 2.33 | 10,956 | 4,704 | yes |
| **ta** | **1.10** | **11,395** | **10,367** | **no** |

Bisected on the failing question: 1 chunk (3,037 tok) works, 2 (5,419) works, 3 (6,537) works,
5 (10,367) fails. The ceiling is somewhere in 6.5k–10.4k.

**This is a V1 correctness bug, not a V2 concern.** With the shipped `n_context: 5`, *every*
Tamil question fails to generate — a third of the corpus and the entire multilingual claim.
Retrieval is unaffected and its numbers stand; only generation breaks.

It also hid behind the retrieval-only eval: `recall@5 = 0.898` looked healthy while the system
could not answer a single Tamil question. Nothing in the harness exercised generation until
`--groundedness` was wired.

Worth noting what the model does when the prompt *does* fit: at 1 chunk it answered
`ஜார்ஜ் பூல் [1]` — correct, and correctly cited. So the capability is there; only the budget is
missing.

**Fix — RESOLVED 2026-08-24 in `ae14943`.** `generate.max_prompt_tokens` (default 6000) had been
added to `core/config.py` and `configs/base.yaml` on 2026-08-24, but nothing read it: the key
existed and the fix did not, so every Tamil question still failed. Now wired end to end.

`retrieve.n_context` is a request, not a guarantee. `answer/prompt.py:fit_context` drops blocks
from the tail until the prompt fits the budget, so the highest-scoring context survives, and
`answer/generate.py` logs `generate.context_trimmed` when it does. A lone oversized block is
attempted rather than abstained.

Sizing needs a *real* token count, and that is the part worth remembering:
`ingest.chunk.count_tokens` counts whitespace words, which undercounts Tamil subwords roughly
sevenfold. Using it as the budget would have reproduced the crash while looking correct — it is
the same blindness that let `recall@5 = 0.898` coexist with zero answerable Tamil questions. So
the budget uses `OpenVinoGenerator.count_tokens` (the pipeline's own tokenizer, exact) and falls
back to `answer.prompt.estimate_tokens`, a script-weighted character rate calibrated on the table
above with a 15% safety margin. Checked against all three measured prompts it over-counts by
~15% and never under; undercounting is the only failure mode that matters here.

Chunk text is never truncated to fit. That would hold the block count up at the cost of §4
provenance — a citation pointing at a page whose text the model was never shown.

**For the developer:** the budget of 6000 is set below the observed 6.5k–10.4k failure window,
not measured against it. The bisection only bracketed the ceiling; nobody has found the exact
edge. If Tamil answers look thin, 6000 is the first number to raise.

---

## 12. Hybrid retrieval cannot answer — RRF scores are not similarities — RESOLVED (149d6f6)

**Resolved with option 1 below.** `retrieve/retriever.py:abstains_for` asks the retriever for
an `abstain_top_score` and compares *that* against tau; `HybridRetriever` returns the dense
cosine. A retriever that exposes nothing keeps V1 behaviour bit-identical, so nothing changed
for dense. `answer/generate.py` and `eval/run.py` both go through it.

Consequence, accepted deliberately: a chunk only the lexical arm found cannot rescue a query
dense wanted to abstain on. That is the price of keeping tau's calibration, and it is the right
trade for a study tool where a confident wrong answer is the worst outcome (§0.6).

The original problem follows.

RRF scores come from ranks, not cosine. Best possible for one list is `1/(60+1)` ≈ 0.016;
with two arms the top hit lands near 0.03. `cfg.retrieve.tau` is 0.45. So
`retriever.abstains(hits, tau)` returns True for **every** query, and the whole system
abstains the moment the hybrid arm is switched on.

This does not affect the sweep: `recall@k` and `mrr@10` are pure rank metrics and never read
`score`. It affects `abstain_precision`, and everything downstream of generation.

**Tried:** returning cosine on fused hits instead. Does not work — a chunk found only by the
lexical arm has no cosine score, and the `Retriever` protocol says hits are sorted by
descending score, which fused-by-RRF/scored-by-cosine would violate.

**Need a decision between:**
1. Abstain on the dense arm's top score, ignoring lexical. `HybridRetriever.last_dense_top_score`
   already exposes it; `answer/generate.py` would consult the retriever rather than the hits.
   Keeps tau's calibration (BLOCKERS #8 — the safe window is 0.01 wide) but means a
   lexical-only hit can never rescue a query dense wanted to abstain on.
2. Recalibrate a separate `tau_rrf` against the golden set. Honest, but #8 showed tau is
   knife-edge on this corpus and there is no reason to think an RRF threshold is less so.

Option 1 is cheaper and preserves a calibration that took real work. Not implemented — it
changes the V1 answer path, which should not happen on the strength of a retrieval number
alone.

Blocked file: `answer/generate.py:_run` (the `abstains(hits, tau)` call).

---

## 13. Rerank costs 5-11s per query, not §10's 1-2s — and TextRerankPipeline is unusable on arm64

Two separate findings from wiring the §10 rerank arm.

### 13a. `openvino_genai.TextRerankPipeline` cannot be used on this machine — RESOLVED, worked around

It loads without error and then throws on **every** `rerank()` call:

```
Node UnigramTokenizer_789260 of type Reference
Tensor data with element type f16, is not representable as pointer to f32
```

Same arm64 trap `ingest/embed.py` already documents: openvino_tokenizers' custom ops are
reference implementations that read weights as f32, and the CPU plugin defaults
`INFERENCE_PRECISION_HINT` to f16 here (BLOCKERS #1 — this box is an M4 Pro, not the Intel
target). The embedder fixes it by pinning the hint on the tokenizer compile.

**TextRerankPipeline gives you no way to do that.** Tried: `**kwargs` on the constructor
(reaches the model compile, not the tokenizer), and `ov.Core().set_property("CPU", ...)`
before construction (GenAI uses its own Core). Both still fail.

Worked around by driving the model directly — `core.compile_model` on the cross-encoder plus
our own pair tokenizer compiled with the f32 hint, which is the pattern `ingest/embed.py`
already uses. This also required re-converting the tokenizer with `number_of_inputs=2`, since
a cross-encoder scores (query, passage) pairs rather than one sequence.

**This is very likely dev-machine-only.** On the Intel target the CPU default is f32 and
`TextRerankPipeline` should work as §10 assumes. The direct path works on both, so it stays.

### 13b. The latency budget in §10 is off by 10x for our chunk size — NEEDS A DECISION

§10 says "20 cross-encoder passes on CPU is ~1-2s". Measured here, bge-reranker-v2-m3 INT8:

| batch | seq len | wall |
|---|---|---|
| 20 pairs | 613 tok | 11.5 s |
| 10 pairs | 613 tok | 5.6 s |
| 5 pairs | 613 tok | 2.8 s |

~550 ms per pair, linear in batch. On real corpus chunks a full top-20 rerank measured
**29.3 s**. Tokenization is not the cost (187 ms); the model is.

The estimate assumed short passages. Our chunks are `target_tokens: 400`, so each pair is
~600 tokens — roughly 10x the work §10 priced in.

`top_n` is already cut from 20 to 10, which is free: `v1.json`'s rank distribution puts the
correct chunk at rank 11-20 exactly zero times. That gets it to ~5.6 s. Still 3x over budget.

§10's own mitigation is "put it on iGPU" — **there is no iGPU here** (BLOCKERS #1).

**Decided: `top_n` is 10** (`ac7635f`, `configs/base.yaml`). Free on quality — the rank
distribution puts the correct chunk at rank 11-20 exactly zero times — and halves the cost.
Measured in the sweep at 3-9 s per query depending on chunk length, so the latency question
below still stands even at 10.

**Still needs a decision, once the quality number lands:**
1. Ship rerank only if the recall gain justifies the added latency. For a study tool where
   TTFT is the pitch, that is likely a no.
2. Re-chunk smaller (`target_tokens` 200) so pairs are ~300 tokens. Halves rerank cost but
   **invalidates the index and every cached stage**, and moves the dense baseline too — so it
   is not a rerank-only decision.
3. Cut `top_n` to 5 (~2.8 s). The rank distribution says rank 2-5 holds 14 of 49 correct
   chunks, so this keeps most of the available headroom.
4. Drop rerank; take hybrid's +0.041 for ~1 ms and spend the time elsewhere.

Measured on an M4 Pro. The Intel target is slower, so these are optimistic.

### 13c. Rerank is 3x slower on Tamil — same root cause as #11

From the groundedness run at `top_n: 10`, 45 rerank calls before it was killed:

| queries | mean rerank |
|---|---|
| 1-40 (en / zh) | 6.6 s |
| 41-45 (first Tamil) | **19.5 s** |
| worst single call | **42.5 s** |

Same cause as BLOCKERS #11: Tamil tokenizes at ~1.10 chars/token against English's 2.33, so an
identical 400-token chunk becomes a far longer sequence for the cross-encoder. Cost per query is
therefore a function of the *language of the retrieved chunks*, not of the query.

This matters more than the mean suggests. A third of the corpus is Tamil, and §5 pitches TTFT.
A student asking a Tamil question waits ~20 s before the first token, worst case 42 s — on top
of generation. The English mean of 6.6 s hides that completely.

If rerank ships, the honest options are a per-script `top_n` (rerank fewer candidates when the
candidates are Tamil), or truncating the passage fed to the cross-encoder — which is safe here
in a way it is not for generation, since the reranker only produces a score and never a citation.

---

## 14. V1 does not fit its own target machine — 13.5 GB steady, 24 GB peak

ARCHITECTURE §0 targets an Intel i5 with **8-16 GB RAM**. Measured here (M4 Pro, 48 GB):

| stage | peak RSS | steady RSS |
|---|---|---|
| generator loaded | 24 GB | 6.5 GB |
| after first generation | 24 GB | **13.5 GB** |
| + embedder + reranker | | **~17 GB** |

The 24 GB is a transient spike while `LLMPipeline` compiles, not the working set —
`resource.ru_maxrss` reports only peaks, which makes this easy to misread.

**This is V1, not V2.** Even with every V2 arm off, the generator alone is 13.5 GB steady with a
24 GB compile spike. On a 16 GB target it will swap or be killed; on 8 GB it cannot load.

Tried, both ineffective:
- `max_position_embeddings` 262144 -> 8192 — steady 13.52 -> 14.11 GB, i.e. nothing. GenAI does
  not size the KV cache from it.
- `KV_CACHE_PRECISION: u8` — no improvement.

So there is no configuration fix. The options are structural:
1. **Smaller generator.** Qwen3-1.7B INT4 would roughly quarter this. Costs answer quality.
2. **Never hold two large models at once.** Load, use, release around each stage rather than
   keeping embedder + reranker + generator resident. Helps ~2-4 GB and stops compile spikes
   stacking; does not fix the 13.5 GB floor.
3. **Restate the target hardware.** If the demo laptop has 32 GB, say so in §0 and move on.

**Need a decision on which.** Nothing else in the build is blocked by this — but every latency
and memory number produced on this machine is optimistic, and this is the one that flatters the
pitch most, because 48 GB hides it completely.

---

## 15. `groundedness` cannot tell a correct answer from word salad

§10 gates V2 on "recall@5 or groundedness". Measured on question 45 (Tamil) under rerank+hybrid:

```
answer:       ஜந்த்து இருார் ஜார் பூலியார் இயற்று இருார். [2]
groundedness: 1.00
```

That is not Tamil. It is a malformed string with a valid citation marker attached, and it scores
a perfect 1.00 — because `groundedness` measures *what fraction of emitted citation markers
resolve to retrieved context*, which is exactly what it was built to measure (`eval/metrics.py`).
It was never a fluency or correctness metric.

BLOCKERS #11 recorded the same question at 1 chunk answering `ஜார்ஜ் பூல் [1]` — correct. So more
context produced worse prose at unchanged groundedness.

**Consequence: half of §10's gating rule is blind on Tamil.** An arm that degrades Tamil
generation while keeping citations valid scores as an improvement. Any V2 groundedness number on
this corpus should be read as "citations resolve", never as "answers are good".

**Need:** either a second metric with teeth on generation quality (chrF against a reference
answer would do, and the golden set already carries the page the answer is on), or an explicit
decision that Tamil answer quality is judged by eye before the demo. I have not built either —
inventing a quality metric the architecture does not ask for is §11 scope creep, and this is a
measurement design decision, not a coding one.

---

## 16. No generator IR can be built on this machine — NNCF's reduce ops have no arm64 executor

Found 2026-08-25 converting `google/gemma-4-E2B-it` (branch `gemma-4`). Every precision fails
in the same place, inside NNCF's weight compression:

```
[CPU] ReduceMin node 'ReduceMin_2233941'   (int4, group_size=128, ratio=0.8)
[CPU] ReduceMin node 'ReduceMin_2167653'   (int8, group_size=-1, ratio=1.0)
[CPU] ReduceMax node 'ReduceMax_1602448'   (fp16, no quantization_config at all)
Supported Reduce executor is not found
    src/plugins/intel_cpu/src/nodes/executors/reduce_list.hpp:70
```

NNCF computes compression scales by **running OpenVINO ops over each weight** rather than in
numpy, so conversion needs a working CPU plugin, not just a working exporter. On arm64
(BLOCKERS #1 — this is an M4 Pro) the Reduce executor for these shapes is not implemented.

It is **not** a bit-width problem and **not** a Gemma 4 problem:
- the three small sub-models (1, 115, 1 layers) compress fine at int8_sym per-channel;
- the large language model fails at 4-bit, at 8-bit, and with compression switched off entirely.

`qwen3-4b-instruct-int4` predates this and is already on disk, which is why it was never hit.

**Consequence for the demo:** this machine can run models but cannot *produce* them. Any new
generator — Gemma 4, a smaller Qwen, anything answering BLOCKERS #14's memory problem — has to
be converted on the Intel laptop, where these executors exist. That is a sibling of #13a
(`TextRerankPipeline` unusable on arm64) and #1 (no Intel GPU here): three separate places
where the dev machine is not the target machine.

**To continue on the Intel laptop** (nothing here is blocked on a decision, only on hardware):

```bash
git checkout gemma-4
uv sync                                                   # transformers 5.5, optimum-intel 2.1
uv run python -m scripts.setup --config configs/gemma4-e2b.yaml --only generator
```

That downloads 9.6 GB and converts INT4; `configs/gemma4-e2b-int8.yaml` does 8-bit. The HF
cache does not travel in git, so the download repeats there.

**Then the wiring that is still missing**: `answer/generate.py:load_generator` builds an
`openvino_genai.LLMPipeline`. Gemma 4 E2B is any-to-any and exports as several IRs, so it loads
through **`VLMPipeline`** instead — genai 2026.3 supports it (`visual_language/gemma4/classes.cpp`,
`gemma4_unified`, `gemma4_mtp_embeddings`). `VLMPipeline.generate` takes the same
`(prompt, config, streamer)` shape, so `OpenVinoGenerator.stream` should carry over unchanged.
`models/registry.py` also assumes one `openvino_model.xml` per entry, which a multi-part VLM
export does not produce — `is_converted` and `ir_sha256` will need to look at the language
model part instead.

**Open question this was meant to answer, still open:** whether Gemma 4 E2B answers Tamil
better than Qwen3-4B, which abstains on 2 of 2 Tamil questions and scores groundedness 0.000
across all 6 — the evidence is BLOCKERS #17.

---

## 17. Tamil answers nothing, and every reported metric hides it

Found 2026-08-25, running the real generator on the pinned index and then re-reading the V1
baseline run. This is the evidence BLOCKERS #16's experiment was trying to act on.

**Live, `qwen3-4b-instruct-int4`, both Tamil golden questions:**

```
பூலியன் இயற்கணிதத்தை உருவாக்கியவர் யார்?   -> "I couldn't find this in your documents."  top 0.608
ஒரு தர்க்க வாயிலுக்கு எத்தனை வெளியீடுகள்...  -> "I couldn't find this in your documents."  top 0.570
```

Both top scores are **above `tau = 0.45`**, so retrieval did not abstain — the model did, on
context that answers the question. English and Chinese are fine on the same run (zh answered
in zh; an English question answered in English off a *Tamil* source, so cross-lingual retrieval
works).

**The V1 baseline run agrees** (`data/eval/runs/20260824T150936Z_dense.parquet`, the run
`eval/baselines/v1.json` pins — the parquet is gitignored, so the numbers are recorded here):

| lang | questions | scored | mean groundedness |
|---|---|---|---|
| en | 37 | 31 | 0.935 |
| zh | 11 | 10 | 0.900 |
| ta | 6 | 4 | **0.000** |

**Why no metric caught it.** `groundedness = 0.844` is the pooled mean, where 41 good en/zh
rows drown 6 Tamil zeros. Worse, the two missing Tamil rows are *structurally* invisible:
`eval/run.py:200` returns `None` for a model-side abstention (excluded from the average), while
the `abstained` column at `run.py:134` records only the **retrieval** gate. A question where
retrieval succeeded and generation refused therefore contributes a clean `recall@5 = 1.0` and
nothing else. Tamil could fail on all six and every headline number would look healthy. Same
blind spot as #15, one layer deeper.

**Two smaller defects this exposed, both unfixed:**
1. `ABSTAIN_MESSAGE` (`answer/prompt.py:23`) and `TIER3_DISCLAIMER` (`:27`) are hardcoded
   English, so both §9 failure paths speak English regardless of the question's language —
   even though rule 1 of both system prompts orders same-language answers.
2. `ui/app.py:121` and `ui/web.py`'s card both label *any* abstention "nothing above the score
   threshold". At top score 0.608 that is simply false. Retrieval abstention and generation
   abstention need different words.

**Need a decision on the order of work:**
1. Per-language metric breakdown in `eval/run.py`, and count model-abstentions separately from
   retrieval-abstentions. Cheapest, and nothing else is measurable until it exists.
2. An `answer_lang_match` column — `detect_script(answer)` vs `detect_script(question)`, using
   `ingest/normalize.py`'s existing detector. Turns "answers come back in the asking language"
   into a number.
3. Localise the two fixed strings. §9 mandates the disclaimer's *content*, not that it be
   English — but it is architecture-adjacent wording, so it wants an explicit yes.
4. The root cause: whether Tamil generation fails because of the 6000-token prompt budget
   biting hardest on Tamil's ~1.1 chars/token (12 of 54 questions were trimmed), or because
   Qwen3-4B INT4 is simply weak in Tamil. **#16's Gemma 4 experiment is the test for the
   second half of that**, and it is blocked on x86 hardware, not on a decision.

---

## 18. This machine's top score for the golden English smoke question sits just under tau

Found 2026-08-25, running step 3 of the gemma-4 continuation plan (real generator, pinned
index, `"Who formalised boolean algebra?"`) on the Intel Windows laptop. The plan expected a
cited English answer; this machine abstains instead:

```
top 0.445, tau 0.45 -> abstained. Same 0.445 across two repeated runs (deterministic, not jitter).
```

Step 2 first confirmed the index's embedder fingerprint is bit-identical to config
(`OK BAAI/bge-m3 c3ea306efeb5 2026.3.0` — same `ir_sha256` as the manifest, so this is not a
stale or rebuilt index). The corpus and index are the ones carried over, unmodified.

**Most likely cause, not confirmed:** int8-quantized kernel execution differs slightly by CPU
microarchitecture (AVX512 vs AVX2 reduction order, etc.), even from a byte-identical IR. This
machine's `bge-m3` embedder fell back to CPU (`model.device_unavailable requested=GPU
available=['CPU', 'GPU.0', 'GPU.1']` — `select_device` in `models/registry.py:253` requires an
exact string match and this machine enumerates two indexed GPUs rather than the target's single
`Iris Xe`, so `"GPU" != "GPU.0"` and it never matches on this hardware regardless of driver
health). If the index was built or last queried on a machine where the embedder ran on GPU or a
different CPU, a few thousandths of cosine similarity drifting across a `tau = 0.45` cutoff set
from one baseline machine is exactly what an unlucky borderline question looks like.

**Not tuned around:** moving `tau` on the strength of one question on one machine is guessing,
which CLAUDE.md rule 4 rules out. `configs/base.yaml`'s `tau: 0.45` is presumably backed by the
V1 eval sweep on the baseline machine — that number stays authoritative until there's a reason
tied to a real eval run, not a single smoke query, to revisit it.

**Need a decision:** is cross-machine embedding-score drift near `tau` expected and tolerable
(then this is just a note, not a defect), or does it call for a numerically stable quantization
path / a margin around `tau` / re-running the eval sweep on this machine before trusting any
number from it? Blocked file: none — this is a finding, not a code change waiting on approval.
