"""Performance bench — ARCHITECTURE.md §5.

Prints TTFT, tok/s, peak RSS and per-stage latency **per device that actually exists on this
host**. §5 asks for "CPU and GPU"; this file does not fabricate the second column. A device
OpenVINO does not report is printed as `unavailable`, never as a zero, and the header says
which machine produced the numbers — an M4 Pro flatters an i5 badly enough to make a slide
dishonest (BLOCKERS.md #1).

    uv run python -m eval.bench --config configs/base.yaml
    uv run python -m eval.bench --traces data/traces/2026-08-22.parquet
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import structlog

from core.config import Config, load_config
from core.telemetry import StageTiming, read_traces

log = structlog.get_logger("bench")

TARGET_DESC = "Intel Core i5 11th gen / Iris Xe (ARCHITECTURE.md §0)"

BENCH_PROMPT = "Explain photosynthesis in three sentences."

NOT_WIRED = "not measured — generation probe disabled (--probe none)"
PENDING_INGEST = (
    "not measured — pass --traces data/traces/<date>.parquet once a pipeline has run (§0.1)"
)

UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GenerationSample:
    """One generate() call's timings.

    `resolved_device` is where the model *actually* ran after fallback (§7.4) — it is not
    always the device that was asked for, and a bench that hides that is lying.
    """

    ttft_ms: float
    tokens_per_s: float
    n_tokens: int
    resolved_device: str = ""

    def __post_init__(self) -> None:
        if self.n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {self.n_tokens}")
        if self.ttft_ms < 0 or self.tokens_per_s < 0:
            raise ValueError("generation timings must be >= 0")


@dataclass(frozen=True)
class IngestResult:
    stages: tuple[StageTiming, ...]
    source: str


class GenerationProbe(Protocol):
    def __call__(self, device: str) -> GenerationSample:
        """Run one generation on `device`. Raise if the device or the model is unusable."""
        ...


class IngestProbe(Protocol):
    def __call__(self) -> IngestResult:
        """Report per-stage latency and say where the numbers came from."""
        ...


def _ru_maxrss_to_bytes(ru_maxrss: int, platform_name: str) -> int:
    """macOS reports `ru_maxrss` in **bytes**, Linux in **kilobytes**.

    Assuming either one universally is a silent 1024x error, in the direction that makes the
    pitch look best on exactly one of the two platforms.
    """
    return int(ru_maxrss) if platform_name == "darwin" else int(ru_maxrss) * 1024


def peak_rss_bytes() -> int:
    return _ru_maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)


@dataclass(frozen=True)
class HostInfo:
    system: str
    release: str
    machine: str
    cpu: str
    python: str
    ov_version: str

    @property
    def matches_target(self) -> bool:
        """Intel x86-64 only. arm64 numbers are not comparable to the target machine."""
        return self.machine.lower() in {"x86_64", "amd64"}

    def as_header(self) -> str:
        lines = [
            f"host: {self.system} {self.release} {self.machine} — {self.cpu}",
            f"python {self.python}  openvino {self.ov_version}",
        ]
        if not self.matches_target:
            lines.append(
                f"WARNING: this is NOT the target machine ({TARGET_DESC}). "
                "Do not present these numbers as target-hardware performance."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class DeviceStatus:
    name: str
    available: bool
    full_name: str
    note: str = ""

    def as_line(self) -> str:
        state = self.full_name if self.available else UNAVAILABLE
        return f"  {self.name:<6} {state}" + (f"  ({self.note})" if self.note else "")


@dataclass(frozen=True)
class BenchRow:
    device: str
    status: str
    ttft_ms: float | None = None
    tokens_per_s: float | None = None
    n_tokens: int | None = None
    peak_rss_bytes: int | None = None
    resolved_device: str | None = None
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.status == "measured"

    @property
    def fell_back(self) -> bool:
        return bool(self.resolved_device) and self.resolved_device != self.device

    def cells(self) -> list[str]:
        def num(v: float | None, fmt: str) -> str:
            return "-" if v is None else format(v, fmt)

        rss = "-" if self.peak_rss_bytes is None else f"{self.peak_rss_bytes / 1e6:.1f}"
        return [
            self.device,
            self.resolved_device or "-",
            num(self.ttft_ms, ".1f"),
            num(self.tokens_per_s, ".2f"),
            rss,
            self.status,
        ]


BENCH_COLUMNS = ["device", "ran_on", "TTFT_ms", "tok/s", "peak_RSS_MB", "status"]

BENCH_SCHEMA: dict[str, Any] = {
    "started_at": pl.Datetime("us", "UTC"),
    "device": pl.Utf8,
    "resolved_device": pl.Utf8,
    "status": pl.Utf8,
    "available": pl.Boolean,
    "ttft_ms": pl.Float64,
    "tokens_per_s": pl.Float64,
    "n_tokens": pl.Int32,
    "peak_rss_bytes": pl.Int64,
    "detail": pl.Utf8,
    "host_system": pl.Utf8,
    "host_machine": pl.Utf8,
    "host_cpu": pl.Utf8,
    "matches_target": pl.Boolean,
    "ov_version": pl.Utf8,
    "config_hash": pl.Utf8,
    "ingest_source": pl.Utf8,
    "stages": pl.List(pl.Struct({"stage": pl.Utf8, "duration_ms": pl.Float64})),
}


@dataclass(frozen=True)
class BenchReport:
    host: HostInfo
    config_hash: str
    devices: tuple[DeviceStatus, ...]
    rows: tuple[BenchRow, ...]
    ingest: tuple[StageTiming, ...] = ()
    ingest_source: str = PENDING_INGEST
    started_at: datetime | None = None

    @property
    def any_measured(self) -> bool:
        return any(r.measured for r in self.rows)

    def as_table(self) -> str:
        body = [r.cells() for r in self.rows]
        widths = [
            max(len(c), *(len(row[i]) for row in body)) if body else len(c)
            for i, c in enumerate(BENCH_COLUMNS)
        ]
        head = "  ".join(c.ljust(w) for c, w in zip(BENCH_COLUMNS, widths, strict=True))
        lines = [
            self.host.as_header(),
            f"config_hash={self.config_hash}",
            "",
            "devices:",
            *(d.as_line() for d in self.devices),
            "",
            head,
            "-" * len(head),
            *("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in body),
        ]

        for row in self.rows:
            if row.fell_back:
                lines.append(f"note: {row.device} fell back to {row.resolved_device} (§7.4)")
        details = {r.detail for r in self.rows if r.detail}
        lines += [f"note: {d}" for d in sorted(details)]

        lines += ["", f"per-stage ingest latency: {self.ingest_source}"]
        lines += [f"  {s.stage:<14} {s.duration_ms:>10.1f} ms" for s in self.ingest]

        missing = [d.name for d in self.devices if not d.available]
        if missing:
            lines += [
                "",
                f"{', '.join(missing)} not present on this host — the §5 CPU-vs-GPU comparison "
                "needs the Intel laptop (BLOCKERS.md #1).",
            ]
        return "\n".join(lines)

    def to_frame(self) -> pl.DataFrame:
        stamp = self.started_at or datetime.now(UTC)
        available = {d.name: d.available for d in self.devices}
        stages = [{"stage": s.stage, "duration_ms": s.duration_ms} for s in self.ingest]
        rows = [
            {
                "started_at": stamp,
                "device": r.device,
                "resolved_device": r.resolved_device,
                "status": r.status,
                "available": available.get(r.device, False),
                "ttft_ms": r.ttft_ms,
                "tokens_per_s": r.tokens_per_s,
                "n_tokens": r.n_tokens,
                "peak_rss_bytes": r.peak_rss_bytes,
                "detail": r.detail,
                "host_system": self.host.system,
                "host_machine": self.host.machine,
                "host_cpu": self.host.cpu,
                "matches_target": self.host.matches_target,
                "ov_version": self.host.ov_version,
                "config_hash": self.config_hash,
                "ingest_source": self.ingest_source,
                "stages": stages,
            }
            for r in self.rows
        ]
        return pl.DataFrame(rows, schema=BENCH_SCHEMA)


def discover_devices() -> tuple[tuple[str, str], ...]:
    """`(name, full_name)` exactly as OpenVINO reports them.

    Imported lazily: `import openvino` costs a second, and this module must stay importable
    (and its table renderable) on a machine where the runtime is broken.
    """
    try:
        import openvino as ov
    except ImportError as e:
        log.warning("openvino.import_failed", error=str(e))
        return ()
    core = ov.Core()
    out = []
    for name in core.available_devices:
        try:
            full = str(core.get_property(name, "FULL_DEVICE_NAME"))
        except RuntimeError as e:
            log.warning("device.property_failed", device=name, error=str(e))
            full = name
        out.append((name, full))
    return tuple(out)


def ov_version() -> str:
    try:
        import openvino as ov
    except ImportError:
        return UNAVAILABLE
    return str(ov.__version__)


def host_info(discovered: tuple[tuple[str, str], ...]) -> HostInfo:
    cpu = next((full for name, full in discovered if name == "CPU"), platform.processor() or "?")
    return HostInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        cpu=cpu,
        python=platform.python_version(),
        ov_version=ov_version(),
    )


def requested_devices(cfg: Config) -> list[str]:
    """Devices `configs/base.yaml` asks for. They are intent, not a promise (BLOCKERS.md #1)."""
    seen: dict[str, None] = {}
    for spec in cfg.models.model_dump().values():
        seen.setdefault(spec["device"], None)
    return list(seen)


def device_statuses(
    discovered: tuple[tuple[str, str], ...], requested: list[str]
) -> tuple[DeviceStatus, ...]:
    """Union of what exists and what the config wants, so a missing GPU is *visible*."""
    found = dict(discovered)
    out = [DeviceStatus(name=n, available=True, full_name=f) for n, f in discovered]
    out += [
        DeviceStatus(
            name=name,
            available=False,
            full_name="",
            note="requested by configs/base.yaml; every model using it falls back to CPU",
        )
        for name in requested
        if name not in found
    ]
    return tuple(out)


def bench_device(device: str, available: bool, probe: GenerationProbe | None) -> BenchRow:
    if not available:
        return BenchRow(
            device=device,
            status=UNAVAILABLE,
            detail=f"{device} is not present on this host — nothing was measured on it",
        )
    if probe is None:
        return BenchRow(device=device, status="pending", detail=NOT_WIRED)
    try:
        sample = probe(device)
    except (RuntimeError, OSError, ValueError, ImportError, NotImplementedError) as e:
        log.warning("bench.probe_failed", device=device, error=f"{type(e).__name__}: {e}")
        return BenchRow(device=device, status="failed", detail=f"{device}: {type(e).__name__}: {e}")
    log.info(
        "bench.device_done",
        stage="bench",
        device=device,
        resolved_device=sample.resolved_device or device,
        ttft_ms=round(sample.ttft_ms, 2),
        tokens_per_s=round(sample.tokens_per_s, 2),
        n_tokens=sample.n_tokens,
    )
    return BenchRow(
        device=device,
        status="measured",
        ttft_ms=sample.ttft_ms,
        tokens_per_s=sample.tokens_per_s,
        n_tokens=sample.n_tokens,
        resolved_device=sample.resolved_device or device,
        # Process high-water mark sampled after this device's run — monotonic, so it reads as
        # "peak so far", not "peak attributable to this device".
        peak_rss_bytes=peak_rss_bytes(),
    )


def run_bench(
    cfg: Config,
    *,
    discovered: tuple[tuple[str, str], ...] | None = None,
    generation_probe: GenerationProbe | None = None,
    ingest_probe: IngestProbe | None = None,
) -> BenchReport:
    found = discover_devices() if discovered is None else discovered
    statuses = device_statuses(found, requested_devices(cfg))
    rows = tuple(bench_device(d.name, d.available, generation_probe) for d in statuses)

    ingest: tuple[StageTiming, ...] = ()
    ingest_source = PENDING_INGEST
    if ingest_probe is not None:
        try:
            result = ingest_probe()
            ingest, ingest_source = tuple(result.stages), result.source
        except (RuntimeError, OSError, ValueError, ImportError, NotImplementedError) as e:
            ingest_source = f"failed — {type(e).__name__}: {e}"
            log.warning("bench.ingest_failed", error=ingest_source)

    return BenchReport(
        host=host_info(found),
        config_hash=cfg.config_hash,
        devices=statuses,
        rows=rows,
        ingest=ingest,
        ingest_source=ingest_source,
        started_at=datetime.now(UTC),
    )


def _mean_metric(value: Any) -> float:
    """`PerfMetrics` getters return a MeanStdPair; older builds return a bare float."""
    return float(getattr(value, "mean", value))


def _sample_from_metrics(
    metrics: Any,
    *,
    resolved_device: str,
    elapsed_s: float,
    budget: int,
) -> GenerationSample:
    """Prefer GenAI's own perf counters — they separate prefill from decode, which a wall
    clock around `generate()` cannot. The wall-clock path is the fallback for a build that
    reports no metrics, and it can only claim TTFT == total time."""
    if metrics is None:
        return GenerationSample(
            ttft_ms=elapsed_s * 1000.0,
            tokens_per_s=budget / elapsed_s if elapsed_s > 0 else 0.0,
            n_tokens=budget,
            resolved_device=resolved_device,
        )
    return GenerationSample(
        ttft_ms=_mean_metric(metrics.get_ttft()),
        tokens_per_s=_mean_metric(metrics.get_throughput()),
        n_tokens=int(metrics.get_num_generated_tokens()),
        resolved_device=resolved_device,
    )


def registry_generation_probe(
    cfg: Config,
    *,
    prompt: str = BENCH_PROMPT,
    max_new_tokens: int | None = None,
    manifest_path: str | Path | None = None,
) -> GenerationProbe:
    """Real TTFT / tok/s from the configured generator, through OpenVINO GenAI.

    Greedy, not sampled: tok/s is unaffected by sampling but run-to-run variance is, and a
    bench that moves under a fixed config is not a bench. If the model has not been converted
    yet the probe raises, and `bench_device` reports that failure verbatim — §0.5 makes a
    fabricated number worse than a missing one.
    """
    from models.registry import ModelNotFound, load_manifest, select_device

    budget = max_new_tokens if max_new_tokens is not None else cfg.generate.max_new_tokens
    wanted = cfg.models.generator.name

    def probe(device: str) -> GenerationSample:
        # Resolved by config name, not by role: §6 makes `configs/base.yaml` the source of
        # truth, and `by_name` names the missing model and the command that produces it.
        entry = load_manifest(manifest_path).by_name(wanted)
        if not entry.is_converted:
            raise ModelNotFound(
                f"{entry.name}: IR missing at {entry.ir_dir} — run "
                f"`uv run python -m scripts.setup` before benching generation"
            )
        import openvino_genai as genai

        resolved = select_device(device)
        pipe = genai.LLMPipeline(str(entry.ir_dir), resolved)
        gen_cfg = genai.GenerationConfig()
        gen_cfg.max_new_tokens = budget
        gen_cfg.do_sample = False

        t0 = time.perf_counter()
        result = pipe.generate(prompt, gen_cfg)
        elapsed_s = time.perf_counter() - t0

        return _sample_from_metrics(
            getattr(result, "perf_metrics", None),
            resolved_device=resolved,
            elapsed_s=elapsed_s,
            budget=budget,
        )

    return probe


def traces_ingest_probe(path: Path) -> IngestProbe:
    """Per-stage latency from recorded traces (`core/telemetry.py`).

    Deliberately not an import of `ingest/pipeline.py`: the traces file already holds real
    stage timings from real runs, so bench stays decoupled from the pipeline's API and
    reports what actually happened rather than a synthetic re-run.
    """

    def probe() -> IngestResult:
        traces = read_traces(path)
        if not traces:
            raise ValueError(f"{path} holds no traces")
        buckets: dict[str, list[float]] = {}
        for trace in traces:
            for stage in trace.stages:
                buckets.setdefault(stage.stage, []).append(stage.duration_ms)
        if not buckets:
            raise ValueError(f"{path} holds {len(traces)} traces but none recorded a stage")
        stages = tuple(
            StageTiming(stage=name, duration_ms=sum(v) / len(v)) for name, v in buckets.items()
        )
        return IngestResult(stages=stages, source=f"mean over {len(traces)} traces in {path.name}")

    return probe


def _write_run(df: pl.DataFrame, cfg: Config, out_dir: Path | None) -> Path:
    target = out_dir or cfg.resolve(cfg.paths.data_dir) / "eval" / "runs"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"{stamp}_bench.parquet"
    df.write_parquet(path, compression="zstd")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Performance bench (ARCHITECTURE.md §5)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--traces", type=Path, default=None, help="traces parquet for stage latency")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument(
        "--probe",
        choices=["registry", "none"],
        default="registry",
        help="'none' enumerates devices without loading a model",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    probe = (
        registry_generation_probe(cfg, max_new_tokens=args.max_new_tokens)
        if args.probe == "registry"
        else None
    )
    report = run_bench(
        cfg,
        generation_probe=probe,
        ingest_probe=traces_ingest_probe(args.traces) if args.traces else None,
    )
    print(report.as_table())
    if not report.any_measured:
        print("\n^ no generation measured — the notes above say exactly what is missing.")

    try:
        path = _write_run(report.to_frame(), cfg, args.out_dir)
        print(f"wrote {path}")
    except OSError as e:
        print(f"could not write bench parquet ({e}); the table above is still valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
