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
