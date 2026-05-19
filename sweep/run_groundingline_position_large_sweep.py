#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from run_grounded_material_sweep import (
    ROOT,
    estimate_case_reflection,
    load_text,
    prepare_binaries,
    save_compact_gather,
    save_text,
    write_summary,
)


FREQUENCY_HZ = 10.0
DELTA_X_VALUES_M = [0.0, 100.0, 250.0, 500.0]

ARRAY_X_START_M = -2000.0
ARRAY_X_END_M = 2000.0
ARRAY_DX_M = 6.28
ARRAY_Z_SURFACE_M = 1499.0
NETWORK = "GL"
SOURCE_X_BASE_M = -10000.0


def case_id(delta_x_m: float) -> str:
    return f"dx{int(round(delta_x_m)):04d}m"


def replace_source(source_text: str, source_x_m: float, frequency_hz: float) -> str:
    updated: list[str] = []
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("xs"):
            updated.append(f"xs                              = {source_x_m:.6f}d0")
        elif stripped.startswith("f0"):
            updated.append(f"f0                              = {frequency_hz:.6f}d0")
        else:
            updated.append(line)
    return "\n".join(updated) + "\n"


def write_shifted_surface_stations(destination: Path, shift_m: float) -> int:
    x_start = ARRAY_X_START_M + shift_m
    x_end = ARRAY_X_END_M + shift_m
    lines: list[str] = []
    x_values: list[float] = []
    x = x_start
    while x <= x_end + 1.0e-9:
        x_values.append(min(x, x_end))
        x += ARRAY_DX_M
    if abs(x_values[-1] - x_end) > 1.0e-9:
        x_values.append(x_end)

    for i, x_value in enumerate(x_values, start=1):
        station = f"S{i:04d}"
        lines.append(
            f"{station:<8s} {NETWORK:<8s} {x_value:16.6f} {ARRAY_Z_SURFACE_M:16.6f} 0.0 0.0"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(lines)


def restore_output_files(backup_dir: Path | None, output_dir: Path) -> None:
    if backup_dir is None or not backup_dir.exists():
        return
    output_dir.mkdir(exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for path in backup_dir.iterdir():
        destination = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)


def run_solver_existing_stations(workdir: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )

    output_dir = workdir / "OUTPUT_FILES"
    output_dir.mkdir(exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    prepare_binaries(workdir)
    subprocess.run([str(workdir / "bin" / "xmeshfem2D")], cwd=workdir, check=True, env=env)
    subprocess.run([str(workdir / "bin" / "xspecfem2D")], cwd=workdir, check=True, env=env)


def load_existing_metadata(results_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not results_dir.exists():
        return rows
    for case_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metadata_path = case_dir / "metadata.json"
        if metadata_path.exists():
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda item: float(item["delta_x_m"]))
    return rows


def main() -> None:
    data_dir = ROOT / "DATA"
    output_dir = ROOT / "OUTPUT_FILES"
    results_dir = Path(__file__).resolve().with_name("results_glposition_large_10hz")
    summary_path = results_dir / "summary.csv"

    original_source = load_text(data_dir / "SOURCE")
    original_stations = load_text(data_dir / "STATIONS") if (data_dir / "STATIONS").exists() else None

    backup_dir = Path(__file__).resolve().with_name("_output_files_backup_glposition_large")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        shutil.copytree(output_dir, backup_dir)

    summary_rows = load_existing_metadata(results_dir)
    if summary_rows:
        write_summary(summary_path, summary_rows)

    try:
        prepare_binaries(ROOT)
        for delta_x_m in DELTA_X_VALUES_M:
            case_dir = results_dir / case_id(delta_x_m)
            metadata_path = case_dir / "metadata.json"
            if metadata_path.exists():
                print(f"=== Large-offset grounding-line shift {delta_x_m:g} m already done; skipping ===", flush=True)
                continue

            coordinate_shift_m = -float(delta_x_m)
            source_x_m = SOURCE_X_BASE_M + coordinate_shift_m
            nstations = write_shifted_surface_stations(data_dir / "STATIONS", coordinate_shift_m)
            save_text(data_dir / "SOURCE", replace_source(original_source, source_x_m, FREQUENCY_HZ))

            print(
                f"=== Large-offset grounding-line shift {delta_x_m:g} m: source/array shift {coordinate_shift_m:g} m ===",
                flush=True,
            )
            run_solver_existing_stations(ROOT)

            case_dir.mkdir(parents=True, exist_ok=True)
            save_compact_gather(output_dir, case_dir / "surface_gather.npz")

            reflection = estimate_case_reflection(output_dir, FREQUENCY_HZ)
            row = {
                "case_id": case_id(delta_x_m),
                "geometry": "supported_cavity_100m",
                "source_frequency_hz": FREQUENCY_HZ,
                "delta_x_m": float(delta_x_m),
                "coordinate_shift_m": coordinate_shift_m,
                "source_x_m": source_x_m,
                "array_x_start_m": ARRAY_X_START_M + coordinate_shift_m,
                "array_x_end_m": ARRAY_X_END_M + coordinate_shift_m,
                "nstations": nstations,
                **reflection,
            }
            metadata_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            summary_rows.append(row)
            summary_rows.sort(key=lambda item: float(item["delta_x_m"]))
            write_summary(summary_path, summary_rows)
    finally:
        save_text(data_dir / "SOURCE", original_source)
        if original_stations is not None:
            save_text(data_dir / "STATIONS", original_stations)
        restore_output_files(backup_dir, output_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


if __name__ == "__main__":
    main()
