"""Conditional query rewrite — §10 (V2, rewrite arm).

§10: trigger **only** when the query is short (< `trigger_max_tokens`) or carries an
unresolved pronoun. Rewriting everything costs a 2-4s generation round-trip per query and
§10 warns it can destroy BGE-M3's cross-lingual signal.

Expect no win on the current golden set: every question there is a well-formed standalone,
so the trigger almost never fires. That is a legitimate result, not a bug — this arm is for
conversational follow-ups the golden set does not contain.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog

from core.config import Config
from core.schema import Retrieved
from ingest.normalize import detect_script
from retrieve.lexical import tokenize
from retrieve.retriever import Retriever

log = structlog.get_logger("retrieve.rewrite")

#: Pronouns/deictics that usually point at earlier turns. English only — the corpus languages
#: that matter here (ta, zh) drop pronouns rather than stranding them, so a word list would
#: mostly produce false positives.
_DANGLING = frozenset(
    {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
        "him",
        "one",
        "ones",
        "former",
        "latter",
        "same",
        "above",
        "below",
    }
)

REWRITE_INSTRUCTION = """\
Rewrite the question as a standalone search query. Keep the original language. Keep every \
proper noun and number exactly as written. Reply with the query only, nothing else.

Question: {q}
Query:"""


def has_dangling_reference(query: str) -> bool:
    return bool(_DANGLING & set(tokenize(query, detect_script(query))))


def should_rewrite(query: str, *, trigger_max_tokens: int) -> bool:
    """§10 trigger: short query, or one pointing at something it never names."""
    terms = tokenize(query, detect_script(query))
    return len(terms) < trigger_max_tokens or has_dangling_reference(query)


class RewritingRetriever:
    """Wraps any retriever; rewrites the query first, but only when §10 says to."""

    @classmethod
    def open(cls, cfg: Config, inner: Retriever) -> RewritingRetriever:
        from answer.generate import GenerationSettings, load_generator

        gen = load_generator(cfg)
        settings = GenerationSettings.from_config(cfg)

        def rewrite(q: str) -> str:
            return "".join(gen.stream(REWRITE_INSTRUCTION.format(q=q), settings))

        return cls(inner, cfg, rewrite)

    def __init__(
        self,
        inner: Retriever,
        cfg: Config,
        rewriter: Callable[[str], str] | None = None,
    ) -> None:
        self.inner = inner
        self.cfg = cfg
        self.rewriter = rewriter
        self.last_rewrite: str | None = None

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        self.last_rewrite = None
        max_tokens = self.cfg.retrieve.rewrite.trigger_max_tokens

        if self.rewriter is None or not should_rewrite(query, trigger_max_tokens=max_tokens):
            return self.inner.retrieve(query, k)

        started = time.perf_counter()
        rewritten = _clean(self.rewriter(query))
        # An empty or runaway rewrite means the model ignored the instruction. Original wins:
        # a bad rewrite is worse than no rewrite, and §10 gates this arm on measurement.
        if not rewritten or len(rewritten) > 4 * len(query) + 80:
            log.warning("rewrite.rejected", original=query, rewritten=rewritten[:120])
            return self.inner.retrieve(query, k)

        self.last_rewrite = rewritten
        log.info(
            "rewrite.applied",
            original=query,
            rewritten=rewritten,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return self.inner.retrieve(rewritten, k)


def _clean(text: str) -> str:
    """First non-empty line, unquoted. Models like to add a preamble or wrap in quotes."""
    for line in text.strip().splitlines():
        stripped = line.strip().strip("\"'“”").strip()
        if stripped:
            return stripped
    return ""
