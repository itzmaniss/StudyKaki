"""BM25 lexical retrieval — ARCHITECTURE.md §10 (V2, hybrid arm).

§10 is explicit that **the work here is per-script tokenization, not BM25**. BM25 Okapi is
forty lines of well-specified arithmetic; deciding what counts as a term in Tamil, Chinese
and English at once is the part that decides whether the arm is worth shipping.

**Why this exists at all.** Dense retrieval embeds meaning and is what makes cross-lingual
retrieval work, but it is weak exactly where students are precise: formulas, proper nouns,
section numbers, technical vocabulary. `ஜார்ஜ் பூல்` and "George Boole" are the same person
to BGE-M3 and different strings to BM25 — and a student searching for `2.4.1` wants the
string. The two failure modes are close to disjoint, which is why §10 fuses rather than
replaces.

**Tokenization is driven by the chunk's own recorded script**, not by re-detecting it here.
`ingest/normalize.py` already decided, `core/schema.Chunk` carries the answer, and a second
opinion at query time could disagree with the one baked into the index.

**No BM25 dependency.** `rank_bm25` and `bm25s` both exist; neither is worth a dependency for
2595 chunks, and §0.3 keeps the runtime surface small. The scoring below is Okapi BM25 with
the standard `k1=1.5, b=0.75`.

**Cross-lingual is not this arm's job.** A Tamil query scores ~0 against English chunks here,
by construction — there is no shared vocabulary. That is not a bug to fix with translation;
it is why `retrieve/dense.py` exists and why `fusion.py` combines them.
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

#: Scripts that do not delimit words with whitespace. §10 names jieba (zh), fugashi (ja) and
#: pythainlp (th); none is a hard dependency here — see `_segment`.
NON_SEGMENTING_SCRIPTS = frozenset({"hans", "hant", "han", "jpan", "kana", "kore", "hang", "thai"})

#: Punctuation stripped from token edges. Internal dots survive so "2.4.1" stays one term.
#: Do NOT switch this to a `\w`-based regex: `\w` excludes combining marks (category Mn), so
#: Tamil words shatter at every vowel sign — ஜார்ஜ் becomes ['ஜ','ர','ஜ'].
_EDGE_PUNCT = " \t\n\r.,;:!?()[]{}<>\"'“”‘’«»—–-_/\\|@#$%^&*+=~`…।॥"


def _segment_cjk(text: str) -> list[str]:
    """Segment a non-whitespace script, preferring a real segmenter when one is installed.

    Character bigrams are the fallback, not a placeholder: for BM25 over Chinese they are a
    long-standing standard and land close to a dictionary segmenter, because a two-character
    window captures most of the language's compounds. jieba would be better and §10 names it,
    but the corpus must remain searchable on a machine that does not have it.
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

    Casefolded rather than lowercased: casefold is the Unicode-correct operation and differs
    for scripts this corpus will meet. Tamil and Devanagari have no case and are unaffected.
    """
    if not text:
        return []
    if script in NON_SEGMENTING_SCRIPTS:
        return [tok.casefold() for tok in _segment_cjk(text)]
    return [t for raw in text.split() if (t := raw.strip(_EDGE_PUNCT).casefold())]


class BM25Index:
    """Okapi BM25 over a fixed corpus, addressed by row so it aligns with `chunks.parquet`.

    Row `i` here is row `i` of the dense index — the same join that `vectors.npy` relies on
    (`retrieve/dense.py`), so a chunk's dense and lexical scores always describe one chunk.
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

        # term -> (rows, term frequencies). Two parallel arrays rather than a list of pairs:
        # scoring a query is then pure numpy scatter-add per term.
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
    """Satisfies `retrieve.retriever.Retriever` over BM25 (§10).

    Built from the same index directory as `DenseRetriever` so the two arms are guaranteed to
    be ranking the same corpus — `fusion.py` deduplicates on `chunk_id`, which is only sound
    if both arms saw the same chunks.
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

        Rows scoring exactly 0.0 share no term with the query and are dropped rather than
        padded out to `k`: an unmatched chunk is not a weak match, and passing it to RRF would
        hand it rank-based credit it never earned.
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
