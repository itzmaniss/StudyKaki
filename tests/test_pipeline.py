"""Ingest orchestration — ARCHITECTURE.md §3, §0 non-negotiable 1.

The pipeline's job is *what runs, on what key, and when a model loads* — so that is what is
asserted here. The stages themselves are tested in their own files; nothing below re-checks
chunking or OCR quality.

Documents are real PDFs built with pymupdf (`tests/test_ingest_load.py` builds the same
born-digital fixture the loader is tested against), the embedder is a deterministic fake, and
the OCR engine is a recording fake. That combination is what makes "OCR never ran" and "the
embedder was never asked to embed anything" observable, which is the whole point: a resumed
ingest that silently re-OCRs the corpus still produces a correct index, just hours later.

No test touches `data/`, and none asserts on embedding values or model text.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pymupdf
import pytest
import structlog

from core.cache import StageCache
from core.config import DEFAULT_CONFIG, PathsConfig, load_config
from core.schema import Block
from ingest.load import CorruptDocumentError, DocumentLoadError
from ingest.ocr import OcrParams
from ingest.pipeline import (
    CONFIG_INDEPENDENT,
    DEFAULT_OCR_DPI,
    Pipeline,
    ingest_paths,
    main,
    pdf_paths,
)
from models.registry import embedder_fingerprint
from retrieve.dense import DenseRetriever, load_index
from tests.test_ingest_load import build_text_pdf
from tests.test_registry import EMBEDDING, write_manifest

DIM = EMBEDDING["dim"]
A4 = (595, 842)

SCAN_LINES = [
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "The light reactions occur in the thylakoid membrane and release oxygen as a by-product.",
    "The Calvin cycle fixes carbon dioxide using the ATP and NADPH the light reactions made.",
]


def build_scanned_pdf(pages: int = 2) -> bytes:
    """Pages with no text layer at all — `load.py` must send these to OCR (§3)."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=A4[0], height=A4[1])
        page.draw_rect(pymupdf.Rect(60, 60, 535, 400), color=(0, 0, 0), width=2)
    return doc.tobytes()


def deterministic_vector(text: str) -> np.ndarray:
    """A vector that depends only on the text, so a cache round trip is checkable."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """The two members of `ingest.embed.Embedder` the pipeline uses, plus a call counter.

    Counting calls is how "the embedder never ran" becomes an assertion rather than a hope.
    """

    def __init__(self, fingerprint: dict) -> None:
        self.fingerprint = fingerprint
        self.calls = 0
        self.texts: list[str] = []

    def embed_passages(self, texts, **kw):
        self.calls += 1
        self.texts.extend(texts)
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([deterministic_vector(t) for t in texts])


class FakeEngine:
    """Recording stand-in for `ingest.ocr.OcrEngine` — det + rec without an IR on disk."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.params = OcrParams()
        self.pages_read: list[int] = []
        self.lines = SCAN_LINES if lines is None else lines

    @property
    def fingerprint(self) -> str:
        return "fake-ocr-engine/1"

    def read_page(self, image, doc_id: str) -> list[Block]:
        self.pages_read.append(image.page)
        return [
            Block(
                block_id=f"{doc_id[:8]}-p{image.page}-{i}",
                doc_id=doc_id,
                page=image.page,
                bbox=(0.1, 0.1 + 0.2 * i, 0.9, 0.25 + 0.2 * i),
                kind="paragraph",
                reading_order=i,
                script="latn",
                text=text,
                ocr_confidence=0.88,
            )
            for i, text in enumerate(self.lines)
        ]


class FakeEncoder:
    """Query-side stand-in, so a pipeline-built index can be searched end to end."""

    def __init__(self, text: str) -> None:
        self.vector = deterministic_vector(text)

    def __call__(self, texts):
        return np.asarray([self.vector], dtype=np.float32)


@pytest.fixture
def cfg(tmp_path):
    base = load_config(DEFAULT_CONFIG)
    return base.model_copy(
        update={
            "paths": PathsConfig(data_dir=tmp_path / "data", ov_cache_dir=tmp_path / "ov_cache")
        }
    )


@pytest.fixture
def models_manifest(tmp_path):
    return write_manifest(tmp_path / "models", embedding=EMBEDDING)


@pytest.fixture
def fingerprint(cfg, models_manifest):
    return embedder_fingerprint(cfg, manifest_path=models_manifest)


@pytest.fixture
def embedder(fingerprint):
    return FakeEmbedder(fingerprint)


@pytest.fixture
def cache(tmp_path):
    return StageCache(tmp_path / "cache")


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "corpus" / "biology.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_text_pdf())
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "corpus" / "scanned.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_scanned_pdf())
    return path


def make_pipeline(cfg, cache, embedder, tmp_path, **kw):
    kw.setdefault("engine", FakeEngine())
    return Pipeline(cfg, cache=cache, embedder=embedder, index_root=tmp_path / "index", **kw)


# --- the spine: a real PDF becomes an index dense.py can open -------------------------------


def test_a_born_digital_pdf_ends_up_as_a_readable_index(cfg, cache, embedder, tmp_path, text_pdf):
    result = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])

    assert result.n_documents == 1
    assert result.n_chunks > 0
    index = load_index(result.path)
    assert index.n_vectors == result.n_chunks
    assert index.dim == DIM


def test_chunks_keep_the_provenance_citations_depend_on(cfg, cache, embedder, tmp_path, text_pdf):
    result = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])
    doc_id = hashlib.sha256(text_pdf.read_bytes()).hexdigest()

    index = load_index(result.path)
    for row in range(index.n_vectors):
        chunk = index.chunk_at(row)
        assert chunk.doc_id == doc_id
        assert chunk.page_start >= 1
        assert chunk.page_end >= chunk.page_start
        assert chunk.block_ids
        assert all(0.0 <= v <= 1.0 for v in chunk.bbox_union)


def test_the_index_manifest_describes_the_embedder_that_actually_ran(
    cfg, cache, embedder, tmp_path, text_pdf, fingerprint
):
    """§3.1 — the fingerprint comes from the embedder object, not from a second lookup."""
    result = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])
    assert result.index.manifest["embedder"] == fingerprint
    assert result.index.manifest["chunk_config_hash"] == cfg.chunk_config_hash
    assert result.index.manifest["n_vectors"] == result.n_chunks


def test_a_query_against_the_built_index_returns_a_cited_chunk(
    cfg, cache, embedder, tmp_path, text_pdf, models_manifest
):
    result = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])
    index = load_index(result.path)
    target = index.chunk_at(0)

    retriever = DenseRetriever(
        index, cfg, FakeEncoder(target.text), models_manifest_path=models_manifest
    )
    hits = retriever.retrieve("what happens in the thylakoid membrane?", k=3)
    assert hits[0].chunk.chunk_id == target.chunk_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].chunk.page_start >= 1


# --- OCR runs only when there is no text layer (§3) -----------------------------------------


def test_ocr_never_runs_on_a_document_with_a_text_layer(cfg, cache, embedder, tmp_path, text_pdf):
    engine = FakeEngine()
    pipeline = make_pipeline(cfg, cache, embedder, tmp_path, engine=engine)
    ingest = pipeline.ingest_document(text_pdf.read_bytes(), text_pdf.name)

    assert ingest.document.has_text_layer is True
    assert ingest.ocr_used is False
    assert engine.pages_read == []


def test_a_scanned_document_goes_through_ocr(cfg, cache, embedder, tmp_path, scanned_pdf):
    engine = FakeEngine()
    pipeline = make_pipeline(cfg, cache, embedder, tmp_path, engine=engine)
    ingest = pipeline.ingest_document(scanned_pdf.read_bytes(), scanned_pdf.name)

    assert ingest.document.has_text_layer is False
    assert ingest.ocr_used is True
    assert engine.pages_read == [1, 2]
    assert ingest.n_blocks == 2 * len(SCAN_LINES)
    assert ingest.chunks
    assert {c.page_start for c in ingest.chunks} <= {1, 2}


def test_an_ocr_engine_is_never_built_for_a_born_digital_corpus(
    cfg, cache, embedder, tmp_path, text_pdf
):
    """`engine()` would import and compile two IRs; a text-layer corpus must not call it."""
    pipeline = Pipeline(
        cfg, cache=cache, embedder=embedder, index_root=tmp_path / "index", engine=None
    )
    result = pipeline.ingest_paths([text_pdf])
    assert result.n_ocr_documents == 0
    assert pipeline._engine is None


# --- resumability: the second run recomputes nothing (§0 non-negotiable 1) -------------------


def test_a_second_run_hits_the_cache_at_every_stage(cfg, cache, embedder, tmp_path, text_pdf):
    make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])

    second = FakeEmbedder(embedder.fingerprint)
    with structlog.testing.capture_logs() as events:
        result = make_pipeline(cfg, cache, second, tmp_path).ingest_paths([text_pdf])

    cached = {e["stage"] for e in events if e.get("event") == "ingest.stage" and e.get("cached")}
    assert {"load", "normalize", "chunk", "embed", "index"} <= cached
    assert second.calls == 0, "a resumed run re-embedded chunks it had already embedded"
    assert result.index.reused is True


def test_a_resumed_scan_does_not_re_ocr_pages(cfg, cache, embedder, tmp_path, scanned_pdf):
    make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([scanned_pdf])

    engine = FakeEngine()
    make_pipeline(cfg, cache, embedder, tmp_path, engine=engine).ingest_paths([scanned_pdf])
    assert engine.pages_read == [], "OCR is the slowest stage; a cached page must never re-run"


def test_adding_a_document_only_embeds_the_new_one(
    cfg, cache, embedder, tmp_path, text_pdf, scanned_pdf
):
    """Embedding is keyed per document, so a growing corpus does not re-embed itself."""
    first = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])

    second = FakeEmbedder(embedder.fingerprint)
    grown = make_pipeline(cfg, cache, second, tmp_path).ingest_paths([text_pdf, scanned_pdf])

    assert grown.n_documents == 2
    assert grown.n_chunks > first.n_chunks
    assert second.texts, "the new document was never embedded"
    assert not any(t in second.texts for t in _texts_of(first))


def _texts_of(result):
    index = load_index(result.path)
    return [index.chunk_at(row).text for row in range(index.n_vectors)]


def test_the_cache_can_be_turned_off_deliberately(cfg, embedder, tmp_path, text_pdf):
    disabled = StageCache(tmp_path / "cache", enabled=False)
    pipeline = make_pipeline(cfg, disabled, embedder, tmp_path)
    pipeline.ingest_paths([text_pdf])

    second = FakeEmbedder(embedder.fingerprint)
    make_pipeline(cfg, disabled, second, tmp_path).ingest_paths([text_pdf])
    assert second.calls > 0
    assert not (tmp_path / "cache").exists()


# --- the corpus is a set, not a command line ------------------------------------------------


def test_path_order_does_not_change_the_index(
    cfg, cache, embedder, tmp_path, text_pdf, scanned_pdf
):
    forward = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf, scanned_pdf])
    backward = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([scanned_pdf, text_pdf])
    assert forward.index.index_id == backward.index.index_id


def test_the_same_bytes_twice_are_one_document(cfg, cache, embedder, tmp_path, text_pdf):
    copy = tmp_path / "corpus" / "biology-copy.pdf"
    copy.write_bytes(text_pdf.read_bytes())

    result = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf, copy])
    assert result.n_documents == 1
    ids = [load_index(result.path).chunk_at(r).chunk_id for r in range(result.n_chunks)]
    assert len(set(ids)) == len(ids)


def test_a_changed_chunk_config_produces_a_different_index(
    cfg, cache, embedder, tmp_path, text_pdf
):
    base = make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([text_pdf])

    retuned = cfg.model_copy(update={"chunk": cfg.chunk.model_copy(update={"target_tokens": 120})})
    other = make_pipeline(retuned, cache, embedder, tmp_path).ingest_paths([text_pdf])
    assert other.index.index_id != base.index.index_id
    assert other.index.manifest["chunk_config_hash"] != base.index.manifest["chunk_config_hash"]


# --- malformed input ------------------------------------------------------------------------


def test_a_corrupt_pdf_fails_loudly_and_writes_no_index(cfg, cache, embedder, tmp_path):
    broken = tmp_path / "corpus" / "broken.pdf"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"%PDF-1.7\nnot actually a pdf\n%%EOF")

    with pytest.raises(CorruptDocumentError):
        make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([broken])
    assert not (tmp_path / "index").exists()


def test_a_missing_path_fails_before_anything_is_loaded(cfg, cache, embedder, tmp_path):
    with pytest.raises(DocumentLoadError, match="not a file"):
        make_pipeline(cfg, cache, embedder, tmp_path).ingest_paths([tmp_path / "nope.pdf"])


def test_a_document_with_no_recognisable_text_still_produces_a_readable_index(
    cfg, cache, embedder, tmp_path, scanned_pdf
):
    """A scan OCR finds nothing in is an empty index, not a crash — every query abstains."""
    pipeline = make_pipeline(cfg, cache, embedder, tmp_path, engine=FakeEngine(lines=[]))
    result = pipeline.ingest_paths([scanned_pdf])

    assert result.n_chunks == 0
    assert load_index(result.path).n_vectors == 0


def test_a_zero_dpi_is_refused(cfg, cache, embedder, tmp_path):
    with pytest.raises(ValueError, match="dpi must be positive"):
        make_pipeline(cfg, cache, embedder, tmp_path, dpi=0)


# --- plumbing -------------------------------------------------------------------------------


def test_stages_that_no_config_key_can_change_are_not_keyed_on_the_config(cfg):
    """Retuning `retrieve.tau` must not invalidate the OCR cache (§0 non-negotiable 1)."""
    assert cfg.config_hash != CONFIG_INDEPENDENT
    assert CONFIG_INDEPENDENT


def test_directories_expand_to_the_pdfs_beneath_them(tmp_path, text_pdf, scanned_pdf):
    found = pdf_paths([text_pdf.parent])
    assert found == sorted([text_pdf, scanned_pdf])


def test_files_are_taken_as_given(text_pdf):
    assert pdf_paths([str(text_pdf)]) == [text_pdf]


def test_the_cli_reports_an_empty_corpus_instead_of_building_an_empty_index(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 1
    assert "no PDFs" in capsys.readouterr().out


def test_the_module_level_helper_builds_an_index(cfg, cache, embedder, tmp_path, text_pdf):
    result = ingest_paths(
        [text_pdf],
        cfg,
        cache=cache,
        embedder=embedder,
        engine=FakeEngine(),
        index_root=tmp_path / "index",
    )
    assert result.n_documents == 1
    assert result.path.is_dir()
    assert DEFAULT_OCR_DPI == 200


# --- the committed BGE-M3 INT8 IR, when it is actually on disk ------------------------------


@pytest.mark.skipif(
    os.environ.get("INTEL2026_REAL_MODELS") != "1",
    reason="set INTEL2026_REAL_MODELS=1 to run against the 544 MB BGE-M3 IR (§7.3)",
)
def test_an_index_built_by_the_real_embedder_verifies_against_it(cfg, cache, tmp_path, text_pdf):
    """The §3.1 loop closed with real weights: what `build_index` stamped into the manifest
    is what `verify_fingerprint` recomputes at query time. A stub IR cannot prove that the
    recorded `ir_sha256` and `ov_version` are the ones the runtime actually reports."""
    from ingest.embed import Embedder
    from retrieve.dense import query_encoder

    emb = Embedder.load(cfg)
    result = Pipeline(cfg, cache=cache, embedder=emb, index_root=tmp_path / "index").ingest_paths(
        [text_pdf]
    )

    index = load_index(result.path)
    assert index.dim == emb.dim
    assert index.n_vectors == result.n_chunks
    assert index.embedder["hf_id"] == "BAAI/bge-m3"

    retriever = DenseRetriever(index, cfg, query_encoder(cfg, embedder=emb))
    hits = retriever.retrieve("What does RuBisCO catalyse?", k=3)
    assert len(hits) == min(3, index.n_vectors)
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert all(h.chunk.page_start >= 1 for h in hits)
