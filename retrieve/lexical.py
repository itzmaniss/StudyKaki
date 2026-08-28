"""BM25 lexical retrieval — ARCHITECTURE.md §10 (V2, hybrid arm).

The work here is per-script tokenization, not BM25. Okapi is forty lines of specified
arithmetic; what counts as a term in Tamil, Chinese and English at once is what decides
whether the arm ships.

Dense embeds meaning and carries cross-lingual retrieval, but it is weak where students are
precise: formulas, proper nouns, section numbers. `ஜார்ஜ் பூல்` and "George Boole" are one
person to BGE-M3 and two strings to BM25, and a student searching `2.4.1` wants the string.
The failure modes are near-disjoint, so §10 fuses rather than replaces.

Tokenization follows the chunk's recorded script — `ingest/normalize.py` already decided,
and a second opinion at query time could disagree with what is baked into the index.

No BM25 dependency: neither `rank_bm25` nor `bm25s` is worth it for 2595 chunks (§0.3).
Standard `k1=1.5, b=0.75`.

Cross-lingual is not this arm's job — a Tamil query scores ~0 against English chunks by
construction. That is what `dense.py` and `fusion.py` are for.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import structlog

from core.config import Config
from core.schema import Retrieved
from ingest.normalize import detect_script
from retrieve.dense import DenseIndex, default_index_dir, load_index

log = structlog.get_logger("retrieve.lexical")

#: Okapi BM25. `k1` damps term-frequency saturation, `b` controls length normalisation.
BM25_K1 = 1.5
BM25_B = 0.75

#: Non-whitespace-delimited scripts. §10 names jieba/fugashi/pythainlp; none is a hard
#: dependency — see `_segment`.
NON_SEGMENTING_SCRIPTS = frozenset({"hans", "hant", "han", "jpan", "kana", "kore", "hang", "thai"})

#: Stripped from token edges; internal dots survive so "2.4.1" stays one term. Not a
#: `\w` regex — `\w` drops combining marks, shattering ஜார்ஜ் into ['ஜ','ர','ஜ'].
_EDGE_PUNCT = " \t\n\r.,;:!?()[]{}<>\"'“”‘’«»—–-_/\\|@#$%^&*+=~`…।॥"


def _segment_cjk(text: str) -> list[str]:
    """Segment a non-whitespace script, preferring a real segmenter if installed.

    Character bigrams are the fallback, not a placeholder: a standard choice for BM25 over
    Chinese that lands close to a dictionary segmenter, since a two-character window catches
    most compounds. jieba is better, but the corpus must stay searchable without it.
    """
    try:
        import jieba
    except ImportError:
        chars = [ch for ch in text if not ch.isspace()]
        if len(chars) < 2:
            return chars
        return ["".join(pair) for pair in zip(chars, chars[1:], strict=False)]
    return [tok for tok in jieba.cut(text) if tok.strip()]


def tokenize(text: str, script: str) -> list[str]:
    """Terms for `text`, segmented according to `script` (§10).

    Casefolded, not lowercased — the Unicode-correct operation, and it differs for scripts
    this corpus meets. Tamil and Devanagari have no case.
    """
    if not text:
        return []
    if script in NON_SEGMENTING_SCRIPTS:
        return [tok.casefold() for tok in _segment_cjk(text)]
    return [t for raw in text.split() if (t := raw.strip(_EDGE_PUNCT).casefold())]


class BM25Index:
    """Okapi BM25 over a fixed corpus, addressed by row to align with `chunks.parquet`.

    Row `i` is row `i` of the dense index — the same join `vectors.npy` relies on, so both
    scores always describe one chunk.
    """

    def __init__(
        self, documents: list[list[str]], *, k1: float = BM25_K1, b: float = BM25_B
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")

        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_len = np.asarray([len(d) for d in documents], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        # term -> (rows, tfs). Parallel arrays, so scoring is numpy scatter-add per term.
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw: dict[str, list[tuple[int, int]]] = {}
        for row, terms in enumerate(documents):
            for term, tf in Counter(terms).items():
                raw.setdefault(term, []).append((row, tf))
        for term, pairs in raw.items():
            rows = np.asarray([r for r, _ in pairs], dtype=np.int32)
            freqs = np.asarray([f for _, f in pairs], dtype=np.float32)
            self.postings[term] = (rows, freqs)

        self.idf = {term: self._idf(len(rows)) for term, (rows, _) in self.postings.items()}

    def _idf(self, n_containing: int) -> float:
        """Lucene's BM25 IDF: always positive, so a term in every document scores ~0, not < 0."""
        return math.log(1.0 + (self.n_docs - n_containing + 0.5) / (n_containing + 0.5))

    def scores(self, query_terms: list[str]) -> np.ndarray:
        """BM25 score per row. Rows matching no query term score exactly 0.0."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avg_len == 0.0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avg_len)
        for term in set(query_terms):
            posting = self.postings.get(term)
            if posting is None:
                continue
            rows, freqs = posting
            out[rows] += self.idf[term] * (freqs * (self.k1 + 1.0)) / (freqs + norm[rows])
        return out


class LexicalRetriever:
    """`retrieve.retriever.Retriever` over BM25 (§10).

    Same index directory as `DenseRetriever`, so both arms rank the same corpus —
    `fusion.py` dedups on `chunk_id`, which is only sound if they saw the same chunks.
    """

    @classmethod
    def open(
        cls,
        cfg: Config,
        index_path: str | Path | None = None,
    ) -> LexicalRetriever:
        index = load_index(index_path if index_path is not None else default_index_dir(cfg))
        return cls(index, cfg)

    def __init__(self, index: DenseIndex, cfg: Config) -> None:
        self.index = index
        self.cfg = cfg
        started = time.perf_counter()
        frame = index.frame
        texts = frame["text"].to_list()
        scripts = frame["script"].to_list()
        documents = [tokenize(t or "", s or "latn") for t, s in zip(texts, scripts, strict=True)]
        self.bm25 = BM25Index(documents)
        log.info(
            "lexical.built",
            index_id=index.index_id,
            n_chunks=len(documents),
            n_terms=len(self.bm25.postings),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        """Top-`k` chunks by BM25, rank 1 = best (§4).

        Rows at exactly 0.0 share no term with the query and are dropped, not padded to `k`:
        an unmatched chunk is not a weak match, and RRF would give it unearned rank credit.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not query.strip():
            raise ValueError("query must not be empty")

        started = time.perf_counter()
        terms = tokenize(query, detect_script(query))
        scores = self.bm25.scores(terms)

        matched = int(np.count_nonzero(scores))
        if matched == 0:
            log.info("retrieve.lexical", n_hits=0, n_query_terms=len(terms), matched=0)
            return []

        top = np.argsort(-scores, kind="stable")[: min(k, matched)]
        hits = [
            Retrieved(chunk=self.index.chunk_at(int(row)), score=float(scores[row]), rank=rank)
            for rank, row in enumerate(top, start=1)
        ]
        log.info(
            "retrieve.lexical",
            index_id=self.index.index_id,
            k=k,
            n_hits=len(hits),
            n_query_terms=len(terms),
            matched=matched,
            top_score=round(hits[0].score, 4),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return hits
