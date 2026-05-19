#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DAS_DIR = ROOT / "das"
if str(DAS_DIR) not in sys.path:
    sys.path.insert(0, str(DAS_DIR))

from io_utils import load_specfem_component  # noqa: E402
from reflection import estimate_reflection_coefficient  # noqa: E402
from synthetic import project_velocity, synthesize_das  # noqa: E402


ROCK = {"rho": 2700.0, "vp": 4000.0, "vs": 2300.0}
SEDIMENT = {"rho": 2000.0, "vp": 2200.0, "vs": 700.0}

WATER_FRACTIONS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
SHEAR_SCALES = [1.0, 0.7, 0.4]


@dataclass
class SweepCase:
    case_id: str
    water_fraction: float
    shear_scale: float
    rho: float
    vp: float
    vs: float
    zp_ratio_to_ice: float
    zs_ratio_to_ice: float


def mix_material(water_fraction: float, shear_scale: float) -> SweepCase:
    rho = (1.0 - water_fraction) * ROCK["rho"] + water_fraction * SEDIMENT["rho"]
    vp = (1.0 - water_fraction) * ROCK["vp"] + water_fraction * SEDIMENT["vp"]
    vs_base = (1.0 - water_fraction) * ROCK["vs"] + water_fraction * SEDIMENT["vs"]
    vs = max(50.0, shear_scale * vs_base)

    ice_rho, ice_vp, ice_vs = 917.0, 3800.0, 1900.0
    zp_ratio = (rho * vp) / (ice_rho * ice_vp)
    zs_ratio = (rho * vs) / (ice_rho * ice_vs)
    case_id = f"wf{int(round(100 * water_fraction)):03d}_ss{int(round(100 * shear_scale)):03d}"
    return SweepCase(
        case_id=case_id,
        water_fraction=water_fraction,
        shear_scale=shear_scale,
        rho=rho,
        vp=vp,
        vs=vs,
        zp_ratio_to_ice=zp_ratio,
        zs_ratio_to_ice=zs_ratio,
    )


def build_cases() -> list[SweepCase]:
    return [mix_material(wf, ss) for wf in WATER_FRACTIONS for ss in SHEAR_SCALES]


def load_text(path: Path) -> str:
    return path.read_text(encoding="ascii")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="ascii")


def replace_source_frequency(source_text: str, frequency_hz: float) -> str:
    lines = source_text.splitlines()
    updated: list[str] = []
    for line in lines:
        if line.strip().startswith("f0"):
            updated.append(f"f0                              = {frequency_hz:.6f}d0")
        else:
            updated.append(line)
    return "\n".join(updated) + "\n"


def replace_material_2(par_text: str, case: SweepCase) -> str:
    lines = par_text.splitlines()
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("2 1 "):
            updated.append(
                f"2 1 {case.rho:.6f}d0 {case.vp:.6f}d0 {case.vs:.6f}d0 0 0 9999 9999 0 0 0 0 0 0"
            )
        elif line.startswith("save_binary_seismograms_single"):
            updated.append("save_binary_seismograms_single  = .false.")
        elif line.startswith("save_ASCII_seismograms"):
            updated.append("save_ASCII_seismograms          = .true.")
        elif line.startswith("output_color_image"):
            updated.append("output_color_image              = .false.")
        elif line.startswith("output_grid_Gnuplot"):
            updated.append("output_grid_Gnuplot             = .false.")
        elif line.startswith("output_grid_ASCII"):
            updated.append("output_grid_ASCII               = .false.")
        else:
            updated.append(line)
    return "\n".join(updated) + "\n"


def filter_surface_stations(stations_path: Path) -> None:
    lines = [
        line
        for line in stations_path.read_text(encoding="ascii").splitlines()
        if line.strip().startswith("S")
    ]
    stations_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def prepare_binaries(workdir: Path) -> None:
    bin_dir = workdir / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("xmeshfem2D", "xspecfem2D"):
        target = bin_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path("../../../bin") / name)


def run_solver(workdir: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    subprocess.run(["python3", "DATA/make_stations.py"], cwd=workdir, check=True, env=env)
    filter_surface_stations(workdir / "DATA" / "STATIONS")

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


def save_compact_gather(output_dir: Path, destination: Path) -> None:
    gather_x = load_specfem_component(output_dir, component="BXX", station_prefix="S")
    gather_z = load_specfem_component(output_dir, component="BXZ", station_prefix="S")
    destination.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(
        destination,
        time=gather_x.time,
        x=gather_x.x,
        z=gather_x.z,
        bxx=gather_x.data,
        bxz=gather_z.data,
        stations=np.asarray(gather_x.stations, dtype=object),
    )


def reflection_settings(frequency_hz: float) -> dict[str, float | tuple[float, float]]:
    if frequency_hz <= 1.5:
        return {
            "phase_velocity_mps": 1900.0,
            "phase_velocity_halfwidth_mps": 300.0,
            "x_min_m": -1800.0,
            "x_max_m": -600.0,
            "incident_search_window_s": (4.0, 6.6),
            "reflected_search_window_s": (5.2, 8.8),
            "wavelet_halfwidth_s": 1.10,
            "x_anchor_m": -1200.0,
            "fk_transition_fraction": 0.20,
            "fk_min_frequency_hz": 0.15,
        }
    if frequency_hz <= 5.5:
        return {
            "phase_velocity_mps": 1900.0,
            "phase_velocity_halfwidth_mps": 300.0,
            "x_min_m": -1800.0,
            "x_max_m": -600.0,
            "incident_search_window_s": (4.2, 5.8),
            "reflected_search_window_s": (5.2, 7.6),
            "wavelet_halfwidth_s": 0.60,
            "x_anchor_m": -1200.0,
            "fk_transition_fraction": 0.25,
            "fk_min_frequency_hz": 0.35,
        }
    return {
        "phase_velocity_mps": 1900.0,
        "phase_velocity_halfwidth_mps": 300.0,
        "x_min_m": -1800.0,
        "x_max_m": -600.0,
        "incident_search_window_s": (4.2, 5.6),
        "reflected_search_window_s": (5.2, 7.2),
        "wavelet_halfwidth_s": 0.40,
        "x_anchor_m": -1200.0,
        "fk_transition_fraction": 0.25,
        "fk_min_frequency_hz": 0.50,
    }


def estimate_case_reflection(output_dir: Path, frequency_hz: float) -> dict[str, float]:
    gather_x = load_specfem_component(output_dir, component="BXX", station_prefix="S")
    gather_z = load_specfem_component(output_dir, component="BXZ", station_prefix="S")
    projected = project_velocity(gather_x, gather_z=gather_z, fiber_angle_deg=0.0)
    das = synthesize_das(projected, gauge_length_m=6.28, channel_spacing_m=6.28)
    settings = reflection_settings(frequency_hz)
    estimate = estimate_reflection_coefficient(das, **settings)
    return {
        "reflection_coefficient": estimate.coefficient,
        "correlation": estimate.correlation,
        "alignment_lag_s": estimate.alignment_lag_s,
        "incident_center_s": estimate.incident_center_s,
        "reflected_center_s": estimate.reflected_center_s,
    }


def write_summary(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grounded-side material sweep.")
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=10.0,
        help="Source dominant frequency in Hz. Default: 10.0",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional explicit results directory. Defaults to sweep/results_<freq>hz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "DATA"
    output_dir = ROOT / "OUTPUT_FILES"
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        freq_label = f"{args.frequency_hz:g}".replace(".", "p")
        results_dir = Path(__file__).resolve().with_name(f"results_{freq_label}hz")
    summary_path = results_dir / "summary.csv"

    original_par = load_text(data_dir / "Par_file")
    original_source = load_text(data_dir / "SOURCE")
    original_stations = load_text(data_dir / "STATIONS") if (data_dir / "STATIONS").exists() else None

    cases = build_cases()
    summary_rows: list[dict[str, float | str]] = []

    try:
        save_text(data_dir / "SOURCE", replace_source_frequency(original_source, args.frequency_hz))
        for case in cases:
            print(f"=== Running {case.case_id} at f0={args.frequency_hz:g} Hz ===", flush=True)
            save_text(data_dir / "Par_file", replace_material_2(original_par, case))
            run_solver(ROOT)

            case_dir = results_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            save_compact_gather(output_dir, case_dir / "surface_gather.npz")

            reflection = estimate_case_reflection(output_dir, args.frequency_hz)
            metadata = {
                "case_id": case.case_id,
                "source_frequency_hz": args.frequency_hz,
                "water_fraction": case.water_fraction,
                "shear_scale": case.shear_scale,
                "rho": case.rho,
                "vp": case.vp,
                "vs": case.vs,
                "zp_ratio_to_ice": case.zp_ratio_to_ice,
                "zs_ratio_to_ice": case.zs_ratio_to_ice,
                **reflection,
            }
            (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            summary_rows.append(metadata)
            write_summary(summary_path, summary_rows)
    finally:
        save_text(data_dir / "Par_file", original_par)
        save_text(data_dir / "SOURCE", original_source)
        if original_stations is not None:
            save_text(data_dir / "STATIONS", original_stations)

    print(f"Wrote sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
