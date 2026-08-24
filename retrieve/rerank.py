"""Cross-encoder rerank — §10 (V2, largest expected precision gain).

Dense/hybrid retrieves k=20, the cross-encoder scores each (query, chunk) pair properly, and
the top `n` survive. Bi-encoders embed query and chunk separately; a cross-encoder reads them
together, which is why it is more accurate and why it costs 20 forward passes per query.

Ceiling on this corpus is known and small: `eval/baselines/v1.json` records recall@20 ==
recall@10 == 0.980 and the correct chunk at rank 11-20 exactly zero times, so a *perfect*
reranker over top-20 buys +0.082 recall@5 and nothing more. §10's `top_n` exists to spend
less for most of that.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog

from core.config import Config
from core.schema import Retrieved
from ingest.embed import TOKENIZER_OV_CONFIG
from models.registry import IR_XML_NAME
from retrieve.retriever import Retriever

log = structlog.get_logger("retrieve.rerank")

CPU = "CPU"
TOKENIZER_XML_NAME = "openvino_tokenizer.xml"


class RerankError(RuntimeError):
    """Reranking failed. Always names the model and device."""


def load_reranker(
    cfg: Config,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[Any, Any, str, str]:
    """Compile the cross-encoder and its pair tokenizer, with the §7.4 device fallback.

    Not `openvino_genai.TextRerankPipeline`: that compiles its own tokenizer with no way to
    pass a precision hint, so on arm64 it loads fine and then dies on every call with
    "element type f16, is not representable as pointer to f32" (BLOCKERS #1, #13). Driving
    the model directly is the same fix ingest/embed.py already uses for the embedder.
    """
    import openvino as ov
    import openvino_tokenizers as ovt

    from models.registry import ModelNotFound, load_manifest, ov_version, select_device, spec_for

    _, spec = spec_for("reranker", cfg)
    entry = load_manifest(manifest_path).by_name(spec.name)

    if not entry.is_converted:
        raise ModelNotFound(
            f"{entry.name}: IR missing at {entry.ir_dir} (§7.3) — run "
            f"`uv run python -m models.convert --only reranker`"
        )
    if entry.ov_version and entry.ov_version != ov_version():
        log.warning(
            "rerank.ov_version_drift",
            model=entry.name,
            ir_built_with=entry.ov_version,
            runtime=ov_version(),
        )

    core = ov.Core()
    try:
        core.add_extension(str(ovt._ext_path))
    except (RuntimeError, AttributeError) as e:
        log.warning("rerank.tokenizer_extension_not_registered", error=str(e))

    tok_xml = entry.ir_dir / TOKENIZER_XML_NAME
    if not tok_xml.is_file():
        raise RerankError(f"{entry.name}: pair tokenizer missing at {tok_xml} — re-run convert")
    tokenizer = core.compile_model(str(tok_xml), CPU, TOKENIZER_OV_CONFIG)

    requested = select_device(spec.device)
    last: Exception | None = None
    for device in dict.fromkeys((requested, CPU)):
        try:
            model = core.compile_model(str(entry.ir_dir / IR_XML_NAME), device)
        except (RuntimeError, OSError) as e:
            last = e
            log.warning("rerank.load_failed", model=entry.name, device=device, error=str(e))
            continue
        log.info(
            "rerank.loaded",
            model=entry.name,
            requested_device=spec.device,
            device=device,
            fell_back=device != spec.device,
        )
        return model, tokenizer, entry.name, device

    raise RerankError(
        f"{entry.name}: cross-encoder would not load on {requested} or {CPU} — last error: {last}"
    )


class RerankingRetriever:
    """Wraps a retriever; rescores its top `top_n` with the cross-encoder."""

    @classmethod
    def open(
        cls,
        cfg: Config,
        inner: Retriever,
        *,
        models_manifest_path: str | Path | None = None,
    ) -> RerankingRetriever:
        model, tokenizer, name, device = load_reranker(cfg, manifest_path=models_manifest_path)
        return cls(inner, cfg, model, tokenizer, name=name, device=device)

    def __init__(
        self,
        inner: Retriever,
        cfg: Config,
        model: Any,
        tokenizer: Any,
        *,
        name: str = "reranker",
        device: str = CPU,
    ) -> None:
        self.inner = inner
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = device

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        top_n = self.cfg.retrieve.rerank.top_n
        # Ask the inner arm for the full k. Reranking a truncated list cannot recover a chunk
        # the first stage never returned, so the candidate pool is what bounds quality.
        hits = self.inner.retrieve(query, k)
        if len(hits) < 2:
            return hits

        head, tail = hits[:top_n], hits[top_n:]
        started = time.perf_counter()
        try:
            scores = self._score(query, [h.chunk.text for h in head])
        except (RuntimeError, OSError) as e:
            # §0.5: a degraded ranking beats no answer, but never silently.
            log.warning("rerank.failed_passthrough", model=self.name, error=str(e))
            return hits

        order = sorted(range(len(head)), key=lambda i: -scores[i])
        reranked = [
            Retrieved(chunk=head[i].chunk, score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]
        # Un-reranked tail keeps its relative order below everything rescored.
        for offset, hit in enumerate(tail, start=len(reranked) + 1):
            reranked.append(Retrieved(chunk=hit.chunk, score=hit.score, rank=offset))

        log.info(
            "retrieve.rerank",
            model=self.name,
            device=self.device,
            n_scored=len(head),
            top_n=top_n,
            moved=sum(1 for new, old in enumerate(order) if new != old),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return reranked

    def _score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance logit per (query, text) pair. Higher is better; not a probability."""
        import numpy as np

        encoded = self.tokenizer([np.array([query] * len(texts)), np.array(texts)])
        feed = {name: encoded[name] for name in ("input_ids", "attention_mask")}
        logits = self.model(feed)[self.model.output(0)]
        return [float(v) for v in np.asarray(logits).reshape(len(texts), -1)[:, 0]]
