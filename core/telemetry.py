"""Per-query telemetry — ARCHITECTURE.md §0.1 (`data/traces/<date>.parquet`), §9 (answer tier).

A trace is the row that explains one answer: which stages ran and how long each took, which
device every model actually landed on *after* fallback (§7.4), what the retriever's best score
was, whether we abstained (§0.6), and which tier the user was shown (§9 requires the tier to be
visible on every answer — it is recorded here so the UI never has to guess).

`trace_id` is the join key: the same string is carried on `Answer.trace_id`, so any rendered
answer can be traced back to the run that produced it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from core.schema import Answer, Retrieved

log = structlog.get_logger("telemetry")

TIER_LOCAL_INDEX = 1
TIER_ONLINE = 2
TIER_PARAMETRIC = 3

TIER_LABELS: dict[int, str] = {
    TIER_LOCAL_INDEX: "local index",
    TIER_ONLINE: "online documents",
    TIER_PARAMETRIC: "general knowledge",
}

# Written explicitly rather than inferred: a trace with no stages or no devices would otherwise
# infer as List(Null) and fail to concat with the rest of the day's file.
TRACE_SCHEMA: dict[str, Any] = {
    "trace_id": pl.Utf8,
    "started_at": pl.Datetime("us", "UTC"),
    "query": pl.Utf8,
    "lang": pl.Utf8,
    "config_hash": pl.Utf8,
    "tier": pl.Int8,
    "abstained": pl.Boolean,
    "top_score": pl.Float64,
    "n_retrieved": pl.Int32,
    "n_citations": pl.Int32,
    "prompt_tokens": pl.Int32,
    "completion_tokens": pl.Int32,
    "ttft_ms": pl.Float64,
    "total_ms": pl.Float64,
    "stages": pl.List(pl.Struct({"stage": pl.Utf8, "duration_ms": pl.Float64})),
    "devices": pl.List(pl.Struct({"model": pl.Utf8, "requested": pl.Utf8, "device": pl.Utf8})),
}


def new_trace_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class StageTiming:
    stage: str
    duration_ms: float

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage name must not be empty")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {self.duration_ms}")


@dataclass(frozen=True)
class DeviceUse:
    """`requested` is what `configs/base.yaml` asked for, `device` is what the load actually got.

    Keeping both is the only way a silent CPU fallback (§7.4) is visible after the fact.
    """

    model: str
    requested: str
    device: str

    @property
    def fell_back(self) -> bool:
        return self.requested != self.device


@dataclass(frozen=True)
class QueryTrace:
    trace_id: str
    started_at: datetime
    query: str
    lang: str
    config_hash: str
    tier: int
    abstained: bool
    top_score: float | None
    n_retrieved: int
    n_citations: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    total_ms: float
    stages: tuple[StageTiming, ...] = ()
    devices: tuple[DeviceUse, ...] = ()

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
        if self.tier not in TIER_LABELS:
            raise ValueError(
                f"unknown answer tier {self.tier!r}, expected one of {sorted(TIER_LABELS)}"
            )
        if self.abstained and self.n_citations:
            raise ValueError("an abstained answer must carry no citations")
        for name in ("n_retrieved", "n_citations", "prompt_tokens", "completion_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.total_ms < 0:
            raise ValueError(f"total_ms must be >= 0, got {self.total_ms}")

    @property
    def tier_label(self) -> str:
        return TIER_LABELS[self.tier]

    @property
    def stage_ms(self) -> dict[str, float]:
        return {s.stage: s.duration_ms for s in self.stages}

    def to_row(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "query": self.query,
            "lang": self.lang,
            "config_hash": self.config_hash,
            "tier": self.tier,
            "abstained": self.abstained,
            "top_score": self.top_score,
            "n_retrieved": self.n_retrieved,
            "n_citations": self.n_citations,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "stages": [{"stage": s.stage, "duration_ms": s.duration_ms} for s in self.stages],
            "devices": [
                {"model": d.model, "requested": d.requested, "device": d.device}
                for d in self.devices
            ],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> QueryTrace:
        started = row["started_at"]
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        return cls(
            trace_id=row["trace_id"],
            started_at=started,
            query=row["query"],
            lang=row["lang"],
            config_hash=row["config_hash"],
            tier=int(row["tier"]),
            abstained=bool(row["abstained"]),
            top_score=row["top_score"],
            n_retrieved=int(row["n_retrieved"]),
            n_citations=int(row["n_citations"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            ttft_ms=row["ttft_ms"],
            total_ms=float(row["total_ms"]),
            stages=tuple(
                StageTiming(stage=s["stage"], duration_ms=s["duration_ms"])
                for s in row.get("stages") or ()
            ),
            devices=tuple(
                DeviceUse(model=d["model"], requested=d["requested"], device=d["device"])
                for d in row.get("devices") or ()
            ),
        )


class TraceRecorder:
    """Accumulates one query's telemetry, then freezes it into a `QueryTrace`.

    Mutable by necessity — it is written to as the query moves through the pipeline — but
    nothing outside this class ever sees the mutable form.
    """

    def __init__(
        self,
        query: str,
        *,
        lang: str = "unknown",
        config_hash: str = "",
        trace_id: str | None = None,
        clock: Any = time.perf_counter,
        now: Any = None,
    ) -> None:
        self.trace_id = trace_id or new_trace_id()
        self.query = query
        self.lang = lang
        self.config_hash = config_hash
        self._clock = clock
        self._now = now or (lambda: datetime.now(UTC))
        self.started_at: datetime = self._now()
        self._t0 = self._clock()
        self._stages: list[StageTiming] = []
        self._devices: list[DeviceUse] = []
        self._top_score: float | None = None
        self._n_retrieved = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._ttft_ms: float | None = None
        self._finished = False

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = self._clock()
        try:
            yield
        except Exception as e:
            elapsed = (self._clock() - t0) * 1000.0
            self._stages.append(StageTiming(stage=name, duration_ms=elapsed))
            log.warning(
                "stage.failed",
                stage=name,
                trace_id=self.trace_id,
                duration_ms=round(elapsed, 2),
                error=type(e).__name__,
            )
            raise
        elapsed = (self._clock() - t0) * 1000.0
        self._stages.append(StageTiming(stage=name, duration_ms=elapsed))
        log.info(
            "stage.done",
            stage=name,
            trace_id=self.trace_id,
            config_hash=self.config_hash,
            duration_ms=round(elapsed, 2),
        )

    def record_device(self, model: str, requested: str, device: str) -> None:
        use = DeviceUse(model=model, requested=requested, device=device)
        self._devices.append(use)
        log.info(
            "device.resolved",
            model=model,
            requested=requested,
            device=device,
            fell_back=use.fell_back,
            trace_id=self.trace_id,
        )

    def record_retrieval(self, hits: Sequence[Retrieved]) -> None:
        self._n_retrieved = len(hits)
        self._top_score = max((h.score for h in hits), default=None)

    def record_tokens(self, *, prompt: int = 0, completion: int = 0) -> None:
        self._prompt_tokens += prompt
        self._completion_tokens += completion

    def mark_first_token(self) -> None:
        """TTFT is measured from query submission, not from generation start — that is the
        number the user actually feels."""
        if self._ttft_ms is None:
            self._ttft_ms = (self._clock() - self._t0) * 1000.0

    def finish(
        self,
        answer: Answer | None = None,
        *,
        tier: int = TIER_LOCAL_INDEX,
        abstained: bool | None = None,
    ) -> QueryTrace:
        if self._finished:
            raise RuntimeError(f"trace {self.trace_id} already finished")
        if answer is not None and answer.trace_id != self.trace_id:
            raise ValueError(
                f"answer.trace_id {answer.trace_id!r} does not match recorder {self.trace_id!r} "
                "— the trace would not join back to the answer"
            )
        self._finished = True
        total_ms = (self._clock() - self._t0) * 1000.0
        if abstained is None:
            abstained = answer.abstained if answer is not None else False
        n_citations = len(answer.citations) if answer is not None else 0
        trace = QueryTrace(
            trace_id=self.trace_id,
            started_at=self.started_at,
            query=self.query,
            lang=self.lang,
            config_hash=self.config_hash,
            tier=tier,
            abstained=abstained,
            top_score=self._top_score,
            n_retrieved=self._n_retrieved,
            n_citations=n_citations,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            ttft_ms=self._ttft_ms,
            total_ms=total_ms,
            stages=tuple(self._stages),
            devices=tuple(self._devices),
        )
        log.info(
            "query.done",
            trace_id=self.trace_id,
            tier=trace.tier_label,
            abstained=trace.abstained,
            top_score=trace.top_score,
            n_retrieved=trace.n_retrieved,
            total_ms=round(total_ms, 2),
            n_stages=len(trace.stages),
        )
        return trace


def traces_root(data_dir: Path) -> Path:
    return data_dir / "traces"


def trace_path(root: Path, when: datetime | None = None) -> Path:
    """One parquet per UTC day, per the storage layout in §0.1."""
    day = (when or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d")
    return root / f"{day}.parquet"


def to_frame(traces: Sequence[QueryTrace]) -> pl.DataFrame:
    return pl.DataFrame([t.to_row() for t in traces], schema=TRACE_SCHEMA)


def write_traces(traces: Sequence[QueryTrace], root: Path, when: datetime | None = None) -> Path:
    """Append to the day's file. `root` is injectable so tests never touch the real `data/`."""
    if not traces:
        raise ValueError("nothing to write")
    root.mkdir(parents=True, exist_ok=True)
    path = trace_path(root, when or traces[0].started_at)
    df = to_frame(traces)
    if path.exists():
        df = pl.concat([pl.read_parquet(path), df], how="vertical")
    df.write_parquet(path, compression="zstd")
    log.info(
        "traces.written", stage="telemetry", path=str(path), n_traces=len(traces), n_rows=df.height
    )
    return path


def read_traces(path: Path) -> list[QueryTrace]:
    if not path.exists():
        raise FileNotFoundError(f"no trace file at {path}")
    return [QueryTrace.from_row(row) for row in pl.read_parquet(path).to_dicts()]
