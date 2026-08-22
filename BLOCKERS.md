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
