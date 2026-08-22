"""Citation verification — ARCHITECTURE.md §4, §0.6.

The model will invent citation markers. §4 is explicit: *strip any `[n]` the model invented*.
A marker is valid only when `n` indexes a block that was actually in the context we handed
the model, so `[7]` against five context blocks is a fabrication and never reaches the user.

**Numbering contract, shared with `answer/prompt.py`:** block `[n]` is `context_hits[n - 1]`,
i.e. 1-indexed *position in the list passed to the prompt builder* — not `Retrieved.rank`.
Both modules must be given the same list, in the same order, or citations point at the wrong page.

Markers inside code spans and fenced code blocks are left untouched: `arr[1]` is array
indexing, not a citation, and rewriting it would corrupt the answer. Such a marker never
renders as a citation either, so it cannot mislead anyone.
"""

from __future__ import annotations

import re

from core.schema import Retrieved

# `[1]`, `[01]`, `[1, 2]`, `[1,2]` — models emit all of these. Anything non-numeric
# (`[abc]`, `[^1]`, a markdown link label) is not a citation marker and is left alone.
_MARKER_RE = re.compile(r"\[[ \t]*(?P<nums>\d+(?:[ \t]*,[ \t]*\d+)*)[ \t]*\]")
_NUM_RE = re.compile(r"\d+")

# Placeholder standing where a dropped marker was, so whitespace can be repaired without
# touching text we did not modify (markdown hard line breaks are two trailing spaces).
_SENT = "\x00"
_CLOSERS = r"\s,.;:!?)\]}»”’、。，．！？：；）】」』"
_DROP_AT_LINE_START = re.compile(r"(?m)^(?P<ind>[ \t]*)" + _SENT + r"[ \t]*")
_DROP_HUGGING = re.compile(r"[ \t]*" + _SENT + r"[ \t]*(?=[" + _CLOSERS + _SENT + r"]|$)", re.M)
_DROP_REST = re.compile(r"[ \t]*" + _SENT + r"[ \t]*")


def verify(answer_text: str, context_hits: list[Retrieved]) -> tuple[str, list[Retrieved]]:
    """Drop invented citation markers and report the citations actually referenced.

    Returns `(clean_text, used_citations)`. Every `[n]` surviving in `clean_text` is
    canonical (one number per bracket) and indexes a real context block, so a renderer
    can trust `\\[(\\d+)\\]` without re-validating. `used_citations` is deduplicated and
    ordered by block number, so it reads as a relevance-ordered source list.

    Text containing only valid canonical markers is returned unchanged.
    """
    hits = _check_hits(context_hits)
    if not isinstance(answer_text, str):
        raise TypeError(f"answer_text must be str, got {type(answer_text).__name__}")

    # The sentinel is an internal marker; a model emitting a raw NUL would corrupt repair.
    text = answer_text.replace(_SENT, "")

    used: dict[int, Retrieved] = {}
    parts = [
        seg if protected else _rewrite(seg, hits, used) for seg, protected in _split_protected(text)
    ]
    return "".join(parts), [used[n] for n in sorted(used)]


def find_markers(text: str) -> list[int]:
    """Every citation number written outside code, valid or invented, in order of appearance."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    found: list[int] = []
    for seg, protected in _split_protected(text):
        if protected:
            continue
        for m in _MARKER_RE.finditer(seg):
            found.extend(int(x) for x in _NUM_RE.findall(m.group("nums")))
    return found


def has_citation_markers(text: str) -> bool:
    """§9 Tier 3 must emit none of these, ever."""
    return bool(find_markers(text))


def strip_all_markers(text: str) -> str:
    """Remove every citation marker. Used by the Tier 3 path, which may cite nothing."""
    return verify(text, [])[0]


def _check_hits(context_hits: list[Retrieved]) -> list[Retrieved]:
    try:
        hits = list(context_hits)
    except TypeError as exc:
        raise TypeError(
            f"context_hits must be a sequence of Retrieved, got {type(context_hits).__name__}"
        ) from exc
    for i, hit in enumerate(hits):
        if not isinstance(hit, Retrieved):
            raise TypeError(f"context_hits[{i}] must be Retrieved, got {type(hit).__name__}")
    return hits


def _rewrite(segment: str, hits: list[Retrieved], used: dict[int, Retrieved]) -> str:
    n = len(hits)

    def repl(m: re.Match[str]) -> str:
        kept: list[int] = []
        for raw in _NUM_RE.findall(m.group("nums")):
            num = int(raw)
            if 1 <= num <= n and num not in kept:
                kept.append(num)
        if not kept:
            return _SENT
        for num in kept:
            used.setdefault(num, hits[num - 1])
        return "".join(f"[{num}]" for num in kept)

    out = _MARKER_RE.sub(repl, segment)
    if _SENT not in out:
        return out
    out = _DROP_AT_LINE_START.sub(r"\g<ind>", out)
    out = _DROP_HUGGING.sub("", out)
    return _DROP_REST.sub(" ", out)


def _split_protected(text: str) -> list[tuple[str, bool]]:
    """Segment into (text, is_code) runs so code is copied through byte-for-byte."""
    segments: list[tuple[str, bool]] = []
    prev = 0
    for start, end in _protected_spans(text):
        if start > prev:
            segments.append((text[prev:start], False))
        segments.append((text[start:end], True))
        prev = end
    if prev < len(text):
        segments.append((text[prev:], False))
    return segments


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "`~":
            i += 1
            continue
        run = _run_length(text, i, ch)
        if run >= 3 and _at_line_start(text, i):
            end = _close_fence(text, i + run, ch, run)
            spans.append((i, end))
            i = end
            continue
        if ch == "`":
            end = _close_inline(text, i + run, run)
            if end is not None:
                spans.append((i, end))
                i = end
                continue
        i += run
    return spans


def _run_length(text: str, start: int, ch: str) -> int:
    i = start
    while i < len(text) and text[i] == ch:
        i += 1
    return i - start


def _at_line_start(text: str, i: int) -> bool:
    return not text[text.rfind("\n", 0, i) + 1 : i].strip()


def _close_fence(text: str, pos: int, ch: str, run: int) -> int:
    closer = re.compile(r"^[ \t]*" + re.escape(ch) + "{" + str(run) + r",}[ \t]*$", re.M)
    m = closer.search(text, pos)
    return m.end() if m else len(text)


def _close_inline(text: str, pos: int, run: int) -> int | None:
    i = pos
    while i < len(text):
        if text[i] != "`":
            i += 1
            continue
        found = _run_length(text, i, "`")
        if found == run:
            return i + found
        i += found
    return None
