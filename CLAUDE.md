# CLAUDE.md — Operating Instructions

**The developer is away.** You are building unattended. Optimise for *reviewable, revertible
progress*, not for finishing fast. A clean half-built system with green tests beats a
finished system nobody can verify.

Read `ARCHITECTURE.md` before your first action. It is the contract. This file is how you work.

---

## Prime directives

1. **`ARCHITECTURE.md` wins.** If a request conflicts with it, follow the architecture and note the conflict in `BLOCKERS.md`.
2. **Never invent scope.** Section 11 of the architecture lists what is out of scope. Don't build it, don't "prepare for" it, don't add config keys for it.
3. **Commit at every green checkpoint.** Small commits. The developer needs to bisect your work when he gets back.
4. **When genuinely blocked, stop and write it down.** Do not guess your way past an ambiguity. A clear blocker is more valuable than a wrong implementation.
5. **Never edit `ARCHITECTURE.md`, `.claude/`, or `uv.lock` by hand.** Propose changes in `BLOCKERS.md` instead. (`uv.lock` may only change as a side effect of `uv add`.)

---

## Definition of done

A task is not done until **all** of these pass:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest -q
uv lock --check
```

Then commit. Then append one line to `PROGRESS.md`.

Nothing is "done" because the code looks right. It's done when the tests are green and it's committed.

---

## Leave a trail

The developer is with the slides and video team and will return to a repo he didn't watch
being built. Two files are how he catches up. Keep them current.

**`PROGRESS.md`** — append after every completed task, newest at the bottom:

```
## 2026-08-22 14:30 — ingest/ocr.py
Done: PaddleOCR det+rec wired through OpenVINO, bbox captured, per-doc parquet cache.
Verified: uv run pytest tests/test_ocr.py -q (7 passed). 12-page Tamil PDF -> 214 blocks in 38s on CPU.
Next: normalize.py (NFC + script detection).
Commit: a3f81c2
```

**`BLOCKERS.md`** — anything you could not resolve. Be specific about what you tried and
what decision you need:

```
## OCR confidence threshold
Tamil recognition returns confidence 0.4-0.6 on clean scans where output is visibly correct.
Latin on the same page returns 0.9+. Dropping blocks below 0.7 loses most Tamil content.
Tried: per-script thresholds (works, but the numbers are guesses without ground truth).
Need: a decision on per-script thresholds, or 20 labelled pages to calibrate against.
Blocked file: ingest/ocr.py:88. Proceeding with threshold 0.35 for taml, flagged TODO.
```

If you hit a blocker, **keep working on something else.** Don't idle.

---

## Build order

Follow §8 of `ARCHITECTURE.md`. Do not skip ahead. Specifically:

**Write `core/schema.py`, `configs/base.yaml`, and `eval/run.py` first**, with a *random*
retriever. Confirm the harness prints a table of meaningless numbers. Only then build real
components. Everything after this is measurable; anything before it is guesswork.

---

## Rules that prevent the expensive mistakes

**Version alignment first.** If any OpenVINO model fails to load, check that `openvino`,
`openvino-tokenizers` and `openvino-genai` are the same version *before* debugging the model.
This is the single most common time sink in this stack.

**Never break the embedder fingerprint.** Any change to the embedding model, its precision,
pooling, normalisation, prefixes, or `max_len` invalidates every existing index. If you change
one, you must bump `stage_version`, regenerate `index_manifest.json`, and say so loudly in
`PROGRESS.md`. Silent drift here destroys retrieval quality in a way that looks like a
chunking bug and costs a day to find.

**Cache, always.** Every ingest stage is keyed on `(input_hash, stage_version, config_hash)`.
If you write a stage without a cache, you are making the developer re-OCR his corpus every
time he tunes a parameter tonight.

**Batch embeddings** (16–32). A per-chunk `embed()` loop is ~10x slower for no benefit.

**No network at runtime.** Model downloads happen only in `scripts/setup.py`. If runtime code
reaches the network, that's a bug — the entire pitch is that this works offline.

**Device fallback is mandatory.** Every model load tries the configured device, falls back to
CPU on failure, and logs which one it got.

---

## Testing expectations

- Every ingest stage: one test with a real fixture, one with a malformed input.
- Schema: round-trip test (dataclass → parquet → dataclass, unchanged).
- Fingerprint: a test proving a mismatched embedder **raises** rather than returning results.
- Retrieval: a test on 5 fixture chunks with a known-correct answer.
- Keep fixtures small. `tests/fixtures/` must stay under 5 MB — no full textbooks.

Do not write tests that assert on model output text. Models drift; assert on structure,
provenance, and error handling.

---

## Git discipline

- Branch: `main` is fine, you're the only one on it.
- Commit message: `<area>: <what changed>` — e.g. `ingest: add script-aware chunker`.
- **Never** `git push`, `git reset --hard`, `git clean`, or `git checkout --`. These destroy work you can't recover.
- Commit before starting any risky refactor, so it can be reverted.

---

## Style

- Type hints on every public function. `from __future__ import annotations` at the top.
- Dataclasses (frozen) for data, not dicts. The schema is the contract.
- No bare `except:`. Catch specific exceptions, log with context.
- Log with `structlog` at stage boundaries: stage name, input hash, duration, output count.
- Docstrings only where the *why* isn't obvious. Skip `"""Returns the result."""`.
- No comments explaining what the next line does.

---

## When you finish everything in §8

Do **not** start §10 (V2). Stop, update `PROGRESS.md` with the full V1 status and the latest
`eval/run.py` output table, and wait. V2 components are gated on the developer reading
tonight's eval numbers and deciding which ones are worth building.
