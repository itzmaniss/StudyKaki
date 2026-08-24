"""A missing device must never look like a slow one.

Nothing here asserts on a duration: every timing comes from a fake probe. What is asserted is
that unavailable devices carry `None` rather than `0`, that RSS units are handled per-platform,
that a fallback stays visible, and that the report survives parquet."""

from __future__ import annotations

import json

import polars as pl
import pytest

from core.config import load_config
from core.telemetry import StageTiming, TraceRecorder, write_traces
from eval.bench import (
    NOT_WIRED,
    PENDING_INGEST,
    BenchReport,
    BenchRow,
    DeviceStatus,
    GenerationSample,
    HostInfo,
    IngestResult,
    _ru_maxrss_to_bytes,
    _sample_from_metrics,
    bench_device,
    device_statuses,
    host_info,
    main,
    peak_rss_bytes,
    registry_generation_probe,
    requested_devices,
    run_bench,
    traces_ingest_probe,
)

CPU_ONLY = (("CPU", "Apple M4 Pro"),)
CPU_AND_GPU = (("CPU", "11th Gen Intel Core i5-1135G7"), ("GPU", "Intel Iris Xe Graphics"))


def sample(device: str) -> GenerationSample:
    return GenerationSample(ttft_ms=910.0, tokens_per_s=12.5, n_tokens=128, resolved_device=device)


def falling_back_probe(device: str) -> GenerationSample:
    return GenerationSample(ttft_ms=1.0, tokens_per_s=1.0, n_tokens=1, resolved_device="CPU")


def exploding_probe(device: str) -> GenerationSample:
    raise RuntimeError(f"no IR for {device}")


def a_host(machine: str = "arm64") -> HostInfo:
    return HostInfo(
        system="Darwin",
        release="25.5.0",
        machine=machine,
        cpu="Apple M4 Pro",
        python="3.11.14",
        ov_version="2026.3.0",
    )


class TestPeakRSS:
    def test_macos_reports_bytes(self):
        assert _ru_maxrss_to_bytes(62_947_328, "darwin") == 62_947_328

    def test_linux_reports_kilobytes(self):
        assert _ru_maxrss_to_bytes(61_472, "linux") == 61_472 * 1024

    def test_the_two_platforms_do_not_agree(self):
        """Guards the classic 1024x bug: treating one platform's unit as the other's."""
        assert _ru_maxrss_to_bytes(1000, "linux") == 1024 * _ru_maxrss_to_bytes(1000, "darwin")

    def test_live_reading_is_physically_plausible(self):
        """A wrong unit lands 1024x outside this range on either platform."""
        assert 10e6 < peak_rss_bytes() < 200e9


class TestDeviceEnumeration:
    def test_discovered_device_is_available(self):
        (cpu,) = device_statuses(CPU_ONLY, ["CPU"])
        assert cpu.available and cpu.full_name == "Apple M4 Pro"

    def test_requested_but_missing_device_is_reported_unavailable(self):
        statuses = device_statuses(CPU_ONLY, ["GPU", "CPU"])
        gpu = next(d for d in statuses if d.name == "GPU")
        assert not gpu.available
        assert "configs/base.yaml" in gpu.note

    def test_no_duplicate_when_requested_device_exists(self):
        names = [d.name for d in device_statuses(CPU_AND_GPU, ["GPU", "CPU"])]
        assert names == ["CPU", "GPU"]

    def test_config_devices_are_the_requested_set(self):
        assert set(requested_devices(load_config())) == {"CPU", "GPU"}

    def test_host_info_names_the_cpu_openvino_reported(self):
        assert host_info(CPU_ONLY).cpu == "Apple M4 Pro"

    def test_arm64_is_not_the_target_machine(self):
        assert not a_host("arm64").matches_target
        assert a_host("x86_64").matches_target

    def test_header_warns_when_off_target(self):
        assert "NOT the target machine" in a_host("arm64").as_header()
        assert "NOT the target machine" not in a_host("x86_64").as_header()


class TestBenchRow:
    def test_unavailable_device_measures_nothing(self):
        row = bench_device("GPU", available=False, probe=sample)
        assert row.status == "unavailable"
        assert (row.ttft_ms, row.tokens_per_s, row.n_tokens, row.peak_rss_bytes) == (
            None,
            None,
            None,
            None,
        ), "a missing device must be None, never 0 — a zero reads as a measurement"
        assert not row.measured

    def test_disabled_probe_is_pending_not_zero(self):
        row = bench_device("CPU", available=True, probe=None)
        assert row.status == "pending"
        assert row.ttft_ms is None and row.tokens_per_s is None
        assert row.detail == NOT_WIRED

    def test_failing_probe_is_reported_not_swallowed(self):
        row = bench_device("CPU", available=True, probe=exploding_probe)
        assert row.status == "failed"
        assert row.ttft_ms is None
        assert "RuntimeError" in row.detail and "no IR" in row.detail

    def test_successful_probe_carries_its_numbers(self):
        row = bench_device("CPU", available=True, probe=sample)
        assert row.measured
        assert row.ttft_ms == pytest.approx(910.0)
        assert row.tokens_per_s == pytest.approx(12.5)
        assert row.peak_rss_bytes and row.peak_rss_bytes > 0

    def test_fallback_is_recorded_against_the_requested_device(self):
        row = bench_device("GPU", available=True, probe=falling_back_probe)
        assert row.resolved_device == "CPU"
        assert row.fell_back

    def test_no_fallback_when_probe_stays_put(self):
        assert not bench_device("CPU", available=True, probe=sample).fell_back

    def test_negative_sample_rejected(self):
        with pytest.raises(ValueError):
            GenerationSample(ttft_ms=-1.0, tokens_per_s=1.0, n_tokens=1)
        with pytest.raises(ValueError, match="n_tokens"):
            GenerationSample(ttft_ms=1.0, tokens_per_s=1.0, n_tokens=-1)


class TestReport:
    def report(self, **over) -> BenchReport:
        return run_bench(load_config(), discovered=CPU_ONLY, **over)

    def test_gpu_absence_is_stated_in_the_table(self):
        table = self.report(generation_probe=sample).as_table()
        assert "unavailable" in table
        assert "CPU-vs-GPU" in table and "BLOCKERS.md" in table

    def test_unavailable_row_prints_dashes_not_numbers(self):
        rows = [
            line
            for line in self.report(generation_probe=sample).as_table().splitlines()
            if line.startswith("GPU ")
        ]
        assert rows, "GPU row missing from the table"
        assert "0" not in rows[-1].replace("GPU", "")

    def test_measured_row_reaches_the_table(self):
        table = self.report(generation_probe=sample).as_table()
        assert "12.50" in table and "910.0" in table

    def test_fallback_note_is_printed(self):
        """A GPU that exists but hands the work to CPU (§7.4) must not read as a GPU number."""
        report = run_bench(
            load_config(), discovered=CPU_AND_GPU, generation_probe=falling_back_probe
        )
        assert "GPU fell back to CPU" in report.as_table()

    def test_any_measured_is_false_without_a_probe(self):
        assert not self.report().any_measured

    def test_ingest_pending_is_explicit(self):
        report = self.report()
        assert report.ingest_source == PENDING_INGEST
        assert "--traces" in report.as_table()

    def test_ingest_probe_stages_are_rendered(self):
        report = self.report(
            ingest_probe=lambda: IngestResult((StageTiming("ocr", 38000.0),), "fixture")
        )
        assert report.ingest_source == "fixture"
        assert "ocr" in report.as_table()

    def test_failing_ingest_probe_does_not_kill_the_run(self):
        report = self.report(ingest_probe=lambda: (_ for _ in ()).throw(OSError("no traces")))
        assert report.ingest_source.startswith("failed")
        assert report.rows, "device rows must still be reported"

    def test_frame_round_trips_through_parquet(self, tmp_path):
        report = self.report(
            generation_probe=sample,
            ingest_probe=lambda: IngestResult((StageTiming("ocr", 1.0),), "fixture"),
        )
        path = tmp_path / "bench.parquet"
        report.to_frame().write_parquet(path, compression="zstd")
        df = pl.read_parquet(path)
        assert df.height == len(report.rows)
        gpu = df.filter(pl.col("device") == "GPU").to_dicts()[0]
        assert gpu["available"] is False
        assert gpu["ttft_ms"] is None, "an absent device must be null in the parquet too"
        assert df["stages"][0][0]["stage"] == "ocr"

    def test_frame_schema_holds_with_no_stages(self, tmp_path):
        """An empty stage list would otherwise infer as List(Null) and fail to write."""
        path = tmp_path / "bench.parquet"
        self.report().to_frame().write_parquet(path, compression="zstd")
        assert pl.read_parquet(path)["stages"].dtype == pl.List(
            pl.Struct({"stage": pl.Utf8, "duration_ms": pl.Float64})
        )

    def test_host_columns_record_the_hardware_caveat(self):
        row = self.report().to_frame().to_dicts()[0]
        assert row["matches_target"] is a_host(row["host_machine"]).matches_target
        assert row["config_hash"].startswith("sha256:")

    def test_empty_row_set_still_renders(self):
        report = BenchReport(host=a_host(), config_hash="sha256:x", devices=(), rows=())
        assert "device" in report.as_table()

    def test_row_cells_are_all_strings(self):
        assert all(isinstance(c, str) for c in BenchRow("CPU", "pending").cells())

    def test_device_status_line_shows_unavailable(self):
        assert "unavailable" in DeviceStatus("GPU", False, "").as_line()


class Ticks:
    """Monotonic fake clock — real durations would make these assertions flake."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.1
        return self.t


class TestTracesIngestProbe:
    def traces(self, tmp_path, n=2):
        for i in range(n):
            rec = TraceRecorder(f"q{i}", clock=Ticks())
            with rec.stage("ocr"):
                pass
            with rec.stage("embed"):
                pass
            path = write_traces([rec.finish()], tmp_path)
        return path

    def test_stage_means_come_from_recorded_traces(self, tmp_path):
        result = traces_ingest_probe(self.traces(tmp_path))()
        assert {s.stage for s in result.stages} == {"ocr", "embed"}
        assert all(s.duration_ms > 0 for s in result.stages)

    def test_source_names_the_file_and_sample_size(self, tmp_path):
        result = traces_ingest_probe(self.traces(tmp_path, n=3))()
        assert "3 traces" in result.source and ".parquet" in result.source

    def test_missing_traces_file_is_reported_not_zeroed(self, tmp_path):
        report = run_bench(
            load_config(),
            discovered=CPU_ONLY,
            ingest_probe=traces_ingest_probe(tmp_path / "absent.parquet"),
        )
        assert report.ingest_source.startswith("failed")
        assert report.ingest == ()

    def test_traces_without_stages_are_rejected(self, tmp_path):
        path = write_traces([TraceRecorder("q").finish()], tmp_path)
        with pytest.raises(ValueError, match="none recorded a stage"):
            traces_ingest_probe(path)()


class MeanStd:
    """Stands in for GenAI's MeanStdPair, which is what the PerfMetrics getters return."""

    def __init__(self, mean: float) -> None:
        self.mean = mean
        self.std = 0.0


class FakeMetrics:
    def __init__(self, ttft, throughput, n_tokens=128) -> None:
        self._ttft, self._throughput, self._n = ttft, throughput, n_tokens

    def get_ttft(self):
        return self._ttft

    def get_throughput(self):
        return self._throughput

    def get_num_generated_tokens(self):
        return self._n


class TestPerfMetrics:
    def test_mean_std_pair_is_unwrapped(self):
        sample = _sample_from_metrics(
            FakeMetrics(MeanStd(880.0), MeanStd(11.25)),
            resolved_device="CPU",
            elapsed_s=9.0,
            budget=512,
        )
        assert sample.ttft_ms == pytest.approx(880.0)
        assert sample.tokens_per_s == pytest.approx(11.25)
        assert sample.n_tokens == 128

    def test_bare_floats_are_accepted(self):
        sample = _sample_from_metrics(
            FakeMetrics(880.0, 11.25), resolved_device="CPU", elapsed_s=9.0, budget=512
        )
        assert sample.ttft_ms == pytest.approx(880.0)

    def test_wall_clock_fallback_when_no_metrics(self):
        sample = _sample_from_metrics(None, resolved_device="CPU", elapsed_s=8.0, budget=128)
        assert sample.tokens_per_s == pytest.approx(16.0)
        assert sample.ttft_ms == pytest.approx(8000.0)

    def test_zero_elapsed_does_not_divide_by_zero(self):
        sample = _sample_from_metrics(None, resolved_device="CPU", elapsed_s=0.0, budget=128)
        assert sample.tokens_per_s == 0.0


class TestRegistryProbe:
    def test_missing_manifest_fails_loudly_with_a_next_step(self, tmp_path):
        probe = registry_generation_probe(load_config(), manifest_path=tmp_path / "manifest.json")
        row = bench_device("CPU", available=True, probe=probe)
        assert row.status == "failed"
        assert row.ttft_ms is None and row.tokens_per_s is None
        assert "scripts.setup" in row.detail or "convert.py" in row.detail

    def test_probe_looks_for_the_generator_the_config_names(self, tmp_path):
        """The failure must name the model `configs/base.yaml` asked for, not a role."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "ov_version": "x", "models": {}}))
        probe = registry_generation_probe(load_config(), manifest_path=manifest)
        row = bench_device("CPU", available=True, probe=probe)
        assert row.status == "failed"
        assert load_config().models.generator.name in row.detail


class TestCLI:
    def test_main_writes_a_run_and_exits_zero(self, tmp_path, capsys):
        assert main(["--out-dir", str(tmp_path), "--probe", "none"]) == 0
        (written,) = list(tmp_path.glob("*_bench.parquet"))
        assert pl.read_parquet(written).height >= 1
        out = capsys.readouterr().out
        assert "devices:" in out and "peak_RSS_MB" in out

    def test_cli_says_so_when_nothing_was_measured(self, tmp_path, capsys):
        """§0.5: a missing number is reported, never filled in with a plausible one.

        Driven through `--probe none` rather than the default. The default probe loads the
        real generator, so this test used to pass only on a machine where the IR happened to
        be absent — and once it was converted it started doing a 512-token CPU generation
        inside the unit suite. The missing-model path itself is covered against an injected
        manifest in `TestGenerationProbe`.
        """
        assert main(["--out-dir", str(tmp_path), "--probe", "none"]) == 0
        out = capsys.readouterr().out
        assert "no generation measured" in out
        assert "12.50" not in out
