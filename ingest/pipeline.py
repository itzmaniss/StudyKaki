"""Ingest orchestration — ARCHITECTURE.md §3, §0 non-negotiable 1, §8.

    load -> (ocr, only when there is no text layer) -> normalize -> chunk -> embed -> index

Every stage here is one already-tested function from `ingest/`. This module owns three things
and nothing else: **what runs**, **what it is keyed on**, and **when a model gets loaded.**

**OCR is skipped whenever the PDF has a text layer** (§3 `load` — "biggest single speedup
available"). `load.py` decides; the pipeline only obeys, because a born-digital textbook that
gets rasterised and re-recognised costs minutes per document and *loses* the exact glyphs it
already had.

**Nothing is keyed on `cfg.config_hash`.** `load`, `ocr` and `normalize` produce the same
output whatever is in `configs/base.yaml` — their models and parameters already ride in their
own input hashes — so they key on a constant, and only `chunk` keys on `chunk_config_hash`.
Keying the whole pipeline on the whole config would re-OCR the corpus every time the developer
retunes `retrieve.tau` tonight, which is exactly the cost §0 non-negotiable 1 exists to prevent.

**Embedding is per document, never per corpus.** `embed_chunks` keys on the hash of the texts
it is handed, so one corpus-wide call means adding the fifty-first PDF re-embeds the other
fifty. Per document, a resumed run pays only for what changed.

**Models are loaded lazily and at most once.** The OCR engine is built only if some document
lacks a text layer; the embedder compiles only on an embed cache miss. A fully cached re-run
of the pipeline therefore touches no IR at all — and nothing here touches the network (§0.3).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from core.cache import StageCache, hash_bytes, stage_timer
from core.config import Config, load_config
from core.schema import Chunk, Document
from ingest.chunk import chunk_blocks
from ingest.embed import DEFAULT_BATCH_SIZE, embed_chunks
from ingest.index import BuiltIndex, build_index
from ingest.load import DocumentLoadError, doc_id_for, load_pdf, render_pages
from ingest.normalize import normalize_blocks
from ingest.ocr import OcrEngine, OcrParams, ocr_pages
from models.registry import embedder_fingerprint

log = structlog.get_logger(__name__)

STAGE_VERSION = "pipeline/1"

#: Rasterisation DPI for documents that have to go through OCR. 200 is PaddleOCR mobile's
#: comfortable range for body text at textbook point sizes.
DEFAULT_OCR_DPI = 200

#: See the module docstring: stages whose output no `configs/base.yaml` key can change.
CONFIG_INDEPENDENT = "none"

PDF_SUFFIX = ".pdf"


@dataclass(frozen=True)
class DocumentIngest:
    """One document, all the way to vectors. Not written anywhere — the index is."""

    document: Document
    chunks: list[Chunk]
    vectors: np.ndarray
    ocr_used: bool
    n_blocks: int

    @property
    def doc_id(self) -> str:
        return self.document.doc_id


@dataclass(frozen=True)
class IngestResult:
    """What a corpus run produced. Deliberately holds no vectors: they are on disk already,
    and keeping a second copy alive doubles peak RSS on an 8 GB machine for nothing."""

    index: BuiltIndex
    documents: list[Document]
    n_chunks: int
    n_ocr_documents: int

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def path(self) -> Path:
        return self.index.path


class _LazyEmbedder:
    """Stands in for `ingest.embed.Embedder` until something actually needs the weights.

    `embed_chunks` touches exactly two members of an embedder: `.fingerprint`, to build the
    cache key, and `.embed_passages`, only on a miss. The fingerprint is a hash of the IR on
    disk plus a JSON block, so it can be produced without `compile_model`. That is the whole
    point: a resumed ingest whose vectors are all cached must not spend 20 s compiling a
    544 MB IR to discover it has nothing to embed.
    """

    def __init__(self, cfg: Config, *, manifest_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self.manifest_path = manifest_path
        self._fingerprint: dict[str, Any] | None = None
        self._loaded: Any = None

    @property
    def fingerprint(self) -> dict[str, Any]:
        if self._fingerprint is None:
            self._fingerprint = embedder_fingerprint(self.cfg, manifest_path=self.manifest_path)
        return self._fingerprint

    @property
    def dim(self) -> int:
        return int(self.fingerprint["dim"])

    def embed_passages(self, texts: Sequence[str], **kw: Any) -> np.ndarray:
        if not texts:  # a blank document must not cost a model compile
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embedder().embed_passages(texts, **kw)

    def _embedder(self) -> Any:
        if self._loaded is None:
            from ingest.embed import Embedder

            self._loaded = Embedder.load(self.cfg, manifest_path=self.manifest_path)
            log.info(
                "pipeline.embedder_loaded", model=self._loaded.name, device=self._loaded.device
            )
        return self._loaded


class Pipeline:
    """Ingest, resumable at every stage. Reusable across documents; not thread-safe.

    `cache`, `embedder` and `engine` are injectable so the orchestration can be tested without
    a 544 MB IR on disk — and so `data/` is never written by a test.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        cache: StageCache | None = None,
        embedder: Any | None = None,
        engine: OcrEngine | None = None,
        manifest_path: str | Path | None = None,
        ocr_params: OcrParams | None = None,
        dpi: int = DEFAULT_OCR_DPI,
        batch_size: int = DEFAULT_BATCH_SIZE,
        index_root: str | Path | None = None,
        script_hint: str | None = None,
    ) -> None:
        if dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")
        self.cfg = cfg
        # Caching is a non-negotiable (§0.1), so it is the default rather than an opt-in.
        # `StageCache(root, enabled=False)` is how a caller turns it off deliberately.
        self.cache = StageCache.from_config(cfg) if cache is None else cache
        self.manifest_path = manifest_path
        self.ocr_params = ocr_params
        self.dpi = dpi
        self.batch_size = batch_size
        self.index_root = index_root
        # Which recognition head leads on a scanned document. Text-based head routing cannot
        # bootstrap itself when the default charset cannot represent the script at all — a
        # Tamil page read by the Chinese+English head yields confident nonsense that
        # `detect_script` labels `hans`, so the Tamil head is never reached. See ingest/ocr.py.
        self.script_hint = script_hint
        self._embedder = (
            embedder if embedder is not None else _LazyEmbedder(cfg, manifest_path=manifest_path)
        )
        self._engine = engine

    @property
    def fingerprint(self) -> dict[str, Any]:
        """The embedder that will vectorise this corpus — and therefore the §3.1 block that
        goes into `index_manifest.json`. Taken from the embedder itself, not recomputed, so
        the manifest can never describe a different model than the one that ran."""
        return dict(self._embedder.fingerprint)

    def engine(self) -> OcrEngine:
        """The OCR engine, built on first use. A born-digital corpus never builds one."""
        if self._engine is None:
            from ingest.ocr import build_engine

            self._engine = build_engine(
                self.cfg, params=self.ocr_params, manifest_path=self.manifest_path
            )
        return self._engine

    def ingest_document(
        self, data: bytes, filename: str, script_hint: str | None = None
    ) -> DocumentIngest:
        """One document, load -> vectors. Nothing is indexed here: an index is a corpus."""
        doc_id = doc_id_for(data)
        with stage_timer("pipeline.document", doc_id) as span:
            span.extra["filename"] = filename
            loaded = load_pdf(data, filename, cache=self.cache, config_hash=CONFIG_INDEPENDENT)
            document = loaded.document
            span.extra["n_pages"] = document.n_pages
            span.extra["has_text_layer"] = document.has_text_layer

            if document.has_text_layer:
                blocks = loaded.blocks
            else:
                blocks = ocr_pages(
                    render_pages(data, dpi=self.dpi),
                    doc_id=doc_id,
                    engine=self.engine(),
                    cache=self.cache,
                    config_hash=CONFIG_INDEPENDENT,
                    script_hint=script_hint or self.script_hint,
                )
            n_blocks = len(blocks)
            span.extra["n_blocks"] = n_blocks

            blocks = normalize_blocks(blocks, cache=self.cache, config_hash=CONFIG_INDEPENDENT)
            chunks = chunk_blocks(
                blocks,
                self.cfg.chunk,
                cache=self.cache,
                config_hash=self.cfg.chunk_config_hash,
            )
            vectors = embed_chunks(
                chunks,
                self.cfg,
                embedder=self._embedder,
                cache=self.cache,
                batch_size=self.batch_size,
                manifest_path=self.manifest_path,
            )
            span.n_out = len(chunks)

        return DocumentIngest(
            document=document,
            chunks=chunks,
            vectors=vectors,
            ocr_used=not document.has_text_layer,
            n_blocks=n_blocks,
        )

    def ingest_paths(self, paths: Sequence[str | Path]) -> IngestResult:
        """Every document, then one index over all of them.

        Documents are ordered by `doc_id` before indexing, so the index is a function of the
        corpus and not of the order the files were named on the command line — the same set
        of PDFs lands on the same `index_id` and rewrites nothing.
        """
        resolved = [Path(p) for p in paths]
        with stage_timer("pipeline", _corpus_hash(resolved)) as span:
            span.extra["n_paths"] = len(resolved)
            ingests = self._ingest_each(resolved)
            ingests.sort(key=lambda d: d.doc_id)

            chunks = [c for d in ingests for c in d.chunks]
            vectors = _stack([d.vectors for d in ingests], int(self.fingerprint["dim"]))
            index = build_index(
                chunks,
                vectors,
                self.cfg,
                root=self.index_root,
                fingerprint=self.fingerprint,
            )
            span.n_out = len(chunks)
            span.extra["index_id"] = index.index_id[:19]

        result = IngestResult(
            index=index,
            documents=[d.document for d in ingests],
            n_chunks=len(chunks),
            n_ocr_documents=sum(1 for d in ingests if d.ocr_used),
        )
        log.info(
            "pipeline.done",
            index_id=index.index_id[:19],
            path=str(index.path),
            n_documents=result.n_documents,
            n_chunks=result.n_chunks,
            n_ocr_documents=result.n_ocr_documents,
            index_reused=index.reused,
        )
        return result

    def _ingest_each(self, paths: Sequence[Path]) -> list[DocumentIngest]:
        """Load each path once. Two paths with identical bytes are one document (§2: `doc_id`
        is the sha256 of the file), and indexing it twice would duplicate every citation."""
        ingests: list[DocumentIngest] = []
        seen: set[str] = set()
        for path in paths:
            data = _read(path)
            doc_id = doc_id_for(data)
            if doc_id in seen:
                log.info("pipeline.duplicate_skipped", path=str(path), doc_id=doc_id[:16])
                continue
            seen.add(doc_id)
            ingests.append(self.ingest_document(data, path.name))
        return ingests


def ingest_paths(paths: Sequence[str | Path], cfg: Config, **kw: Any) -> IngestResult:
    """Ingest a corpus and write its index. The one call `eval/run.py` and the UI need."""
    return Pipeline(cfg, **kw).ingest_paths(paths)


def ingest_document(data: bytes, filename: str, cfg: Config, **kw: Any) -> DocumentIngest:
    return Pipeline(cfg, **kw).ingest_document(data, filename)


def _read(path: Path) -> bytes:
    if not path.is_file():
        raise DocumentLoadError(f"not a file: {path}")
    return path.read_bytes()


def _corpus_hash(paths: Sequence[Path]) -> str:
    return hash_bytes("\x1f".join(sorted(str(p) for p in paths)).encode())


def _stack(parts: Sequence[np.ndarray], dim: int) -> np.ndarray:
    """One `[n_chunks, dim]` matrix in document order, or an empty one of the right width."""
    if not parts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(parts), dtype=np.float32)


def pdf_paths(targets: Sequence[str | Path]) -> list[Path]:
    """Files as given, directories expanded to the PDFs beneath them, sorted for determinism."""
    out: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob(f"*{PDF_SUFFIX}") if p.is_file()))
        else:
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest PDFs into a flat index (ARCHITECTURE.md §3)")
    ap.add_argument("paths", nargs="+", type=Path, help="PDF files, or directories of PDFs")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dpi", type=int, default=DEFAULT_OCR_DPI)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the stage cache and recompute everything (slow; for debugging drift)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    targets = pdf_paths(args.paths)
    if not targets:
        print("no PDFs found in the given paths")
        return 1

    cache = StageCache.from_config(cfg)
    if args.no_cache:
        cache = StageCache(cache.root, enabled=False)

    result = ingest_paths(targets, cfg, cache=cache, dpi=args.dpi, batch_size=args.batch_size)
    print(
        f"{result.n_documents} document(s), {result.n_chunks} chunks "
        f"({result.n_ocr_documents} needed OCR)\n"
        f"index {result.index.index_id}\nwrote {result.path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
