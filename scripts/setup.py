"""One-time offline setup: download checkpoints, convert to OpenVINO IR, warm the cache.

**This is the only module in the repo permitted to touch the network** (ARCHITECTURE.md
§0.3). `models/convert.py` reads the local Hugging Face cache and refuses to download;
everything under `ingest/`, `retrieve/` and `answer/` must work with the network off.
If you find a runtime import reaching out, that is a bug, not a feature.

    uv run python -m scripts.setup                 # everything in configs/base.yaml
    uv run python -m scripts.setup --only embedder
    uv run python -m scripts.setup --skip-warm     # CI / low-disk

Each model is independent: one failure is reported and the rest still run, because a
missing generator should not stop you from indexing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from core.config import Config, load_config
from models.convert import (
    IR_ROOT,
    ConversionError,
    convert,
    snapshot_dir,
    source_for,
    write_manifest,
)
from models.registry import (
    MANIFEST_PATH,
    ROLES,
    RegistryError,
    load_model,
    ov_version,
    spec_for,
)

log = structlog.get_logger("scripts.setup")


@dataclass
class StepResult:
    role: str
    name: str
    status: str
    device: str = "-"
    detail: str = ""


def download(name: str, cfg: Config) -> Path:
    """Fetch a checkpoint into the local HF cache, pinned to its commit SHA (§3.1 rule 1)."""
    _, spec = spec_for(name, cfg)
    src = source_for(spec.name)
    log.info("setup.download", model=src.name, hf_id=src.hf_id, revision=src.hf_revision[:8])
    return snapshot_dir(src, allow_download=True)


def warm_cache(name: str, cfg: Config, manifest_path: Path) -> str:
    """Compile once so the first real query does not pay for it (§7.2).

    Returns the device actually obtained, which is also the honest answer to
    "did the GPU intent in base.yaml survive contact with this machine?" (§7.4).
    """
    loaded = load_model(name, cfg, manifest_path=manifest_path)
    log.info(
        "setup.warmed",
        model=loaded.name,
        requested_device=loaded.requested_device,
        device=loaded.device,
    )
    return loaded.device


def run(
    targets: list[str],
    cfg: Config,
    *,
    manifest_path: Path = MANIFEST_PATH,
    out_root: Path = IR_ROOT,
    do_download: bool = True,
    do_convert: bool = True,
    do_warm: bool = True,
    overwrite: bool = False,
) -> list[StepResult]:
    results: list[StepResult] = []
    entries: dict[str, dict[str, Any]] = {}

    for target in targets:
        try:
            role, spec = spec_for(target, cfg)
        except RegistryError as e:
            results.append(StepResult(role=target, name="?", status="unknown", detail=str(e)))
            continue

        result = StepResult(role=role, name=spec.name, status="ok")
        try:
            if do_download:
                download(target, cfg)
            if do_convert:
                entries[spec.name] = convert(target, cfg, out_root=out_root, overwrite=overwrite)
        except (ConversionError, RegistryError, OSError, RuntimeError, ImportError) as e:
            result.status = "FAILED"
            result.detail = str(e).splitlines()[0][:200]
            log.error("setup.failed", model=spec.name, stage="convert", error=str(e))
            results.append(result)
            continue
        results.append(result)

    if entries:
        write_manifest(entries, manifest_path)

    if do_warm:
        for result in results:
            if result.status != "ok":
                continue
            try:
                result.device = warm_cache(result.role, cfg, manifest_path)
            except (RegistryError, RuntimeError) as e:
                result.status = "converted, not warmed"
                result.detail = str(e).splitlines()[0][:200]
                log.error("setup.failed", model=result.name, stage="warm", error=str(e))

    return results


def format_report(results: list[StepResult], cfg: Config) -> str:
    header = ("role", "model", "precision", "requested", "device", "status")
    rows = [header]
    for r in results:
        try:
            spec = getattr(cfg.models, r.role)
            precision, requested = spec.precision, spec.device
        except AttributeError:
            precision = requested = "-"
        rows.append((r.role, r.name, precision, requested, r.device, r.status))

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    details = [f"  {r.name}: {r.detail}" for r in results if r.detail]
    if details:
        lines.extend(["", "details:", *details])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download + convert + warm (ARCHITECTURE.md §7)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--only", action="append", default=None, help=f"role or name; {ROLES}")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out-root", type=Path, default=IR_ROOT)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="re-convert even if IR exists")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    targets = list(args.only or ROLES)

    print(f"openvino {ov_version()} — setting up {', '.join(targets)}")
    results = run(
        targets,
        cfg,
        manifest_path=args.manifest,
        out_root=args.out_root,
        do_download=not args.skip_download,
        do_convert=not args.skip_convert,
        do_warm=not args.skip_warm,
        overwrite=args.overwrite,
    )
    print()
    print(format_report(results, cfg))

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print(f"\n{len(failed)} of {len(results)} models did not complete.")
        return 1
    print(f"\nmanifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
