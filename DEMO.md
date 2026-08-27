# StudyKaki — demo runbook

Everything needed to put this on camera. Numbers here were measured on the shipping config
(`configs/base.yaml`: `gemma-4-e2b-it` int4, `language_reminder: true`) — re-check them after
any change to the embedder, chunking, `tau`, or the generator.

## Run it

The UI *is* the backend — `ui/web.py` loads the retriever and the generator and serves the
page from one process. One command, nothing else to start:

```bash
uv run python -m ui.web \
  --index data/index/73677e039adbe1178b006ad2f3027bcb6fbca934ca7257d9be64ed221f5dfd3c \
  --docs data/corpus/en --docs data/corpus/ta --docs data/corpus/zh
```

Then <http://127.0.0.1:8000> — it opens your browser on its own (add `--no-open` to stop that).

- `--index` is **required**: two indexes exist on disk and it refuses to guess.
- `--docs` is what makes citations clickable and the badge read `8 docs · 2,595 chunks`.
  Without it you get `0 docs` and dead citation links.
- Use **forward slashes** even on Windows — backslashes get mangled through PowerShell.
- Nothing reaches the network. It binds `127.0.0.1`, and the page loads no CDN font or script,
  so it genuinely works with the wifi off. That is the pitch — feel free to turn wifi off on camera.

## Before rolling

**Warm it up.** The first query pays a one-off cost: ~30 s first, ~12 s by the third. Ask
anything off-camera before you start.

## The three chips on the page

All three are verbatim `eval/golden.jsonl` questions and all clear `tau = 0.45`.

| # | question | top score | answer |
|---|---|---|---|
| 1 | `In which year did the personal computer first appear?` | 0.647 | "The personal computer (PC) first appeared in 1975 [1]." → `std12_cs_vol1_en.pdf` pp.67-68 |
| 2 | `பூலியன் இயற்கணிதத்தை உருவாக்கியவர் யார்?` | 0.608 | ஜார்ஜ் பூல், 1840s [1] → `digital_electronics_ta.pdf` pp.87-90 |
| 3 | `我国IPv6地址数量是多少？` | 0.781 | "截至2021年12月，我国IPv6地址数量为63052块/32 [1]。" → `cnnic_internet_report.pdf` pp.17-19 |

## The Tamil question to use

```
SOP முறையில் கார்ணா வரைபடத்தில் எந்த மதிப்புகளைத் தொகுக்க வேண்டும்?
```

> SOP முறையில் K வரைபடத்தில் குறியிடப்பட்டுள்ள 1' தர்க்க மதிப்புகளை ஒன்றாகத் தொகுக்க வேண்டும் [1].
>
> `[1] digital_electronics_ta.pdf pp.163-165` · 25 s

Correct, one clean sentence, no preamble, and the marker resolves to a real page you can open.
The only one of the six Tamil golden questions with no blemish.

## Tamil questions to avoid on camera

All six were driven through the live server. Three are safe, three are not:

| question | verdict |
|---|---|
| `SOP முறையில் கார்ணா வரைபடத்தில்...` | **safe** — correct, cited, clean |
| `தர்க்க சுற்றுகளில் HIGH மற்றும் LOW...` | usable — correct (5V/0V) and cited, but renders `HGH`, missing the I |
| `பூலியன் இயற்கணிதத்தை உருவாக்கியவர் யார்?` | usable — correct and cited, but opens with `தேற்றம் 1: <question>? பதில்:`, mimicking the textbook Q&A layout on the retrieved page |
| `அடிப்படை தர்க்க வாயில்கள் எத்தனை வகைப்படும்?` | **avoid** — content right (AND/OR/NOT) but writes `[3.5]`, a malformed marker, so the answer shows **zero citations** |
| `கார்ணா வரைபடத்தில் சாய்வாக நகர்வது...` | **avoid** — content right but writes `(2)`, again zero citations, and the text is slightly garbled |
| `ஒரு தர்க்க வாயிலுக்கு எத்தனை வெளியீடுகள்...` | **avoid — factually wrong.** Says a logic gate has two outputs. It has one. |

Only 3 of 6 Tamil questions produced a valid citation and one was outright wrong, which is what
`ta` groundedness `0.833` looks like up close. Tamil is where the current generator is most
likely to embarrass you. If a Tamil segment is central to the pitch, `qwen3-4b-instruct` scored
`ta 1.000` — the swap is two lines in `configs/base.yaml` plus
`uv run python -m models.convert --only generator`, and it costs ~2.3x on latency.

## What the footer strip says

It reports what actually loaded, not what the config asked for (§7.4) — `generator`,
`embedder`, `retrieval`, `device`. If it says a device you did not expect, believe the strip.

## Generator trade, if it comes up

Measured over all 54 golden questions, identical retrieval, same machine:

| | groundedness | en (lang_match) | ta | zh | median generate |
|---|---|---|---|---|---|
| **gemma-4-e2b-it** (shipping) | 0.796 | 0.758 (1.000) | 0.833 | 0.900 | **12.6 s** |
| qwen3-4b-instruct | **0.935** | **0.935** (1.000) | **1.000** | 0.900 | 29.5 s |

Retrieval is the same either way: recall@5 `0.898`, MRR@10 `0.743`, abstain precision `1.000`.
