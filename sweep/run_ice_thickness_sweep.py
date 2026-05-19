#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil

from run_grounded_material_sweep import (
    ROOT,
    estimate_case_reflection,
    load_text,
    replace_source_frequency,
    run_solver,
    save_compact_gather,
    save_text,
    write_summary,
)


THICKNESSES_M = [75, 100, 125, 150, 175, 250, 325, 400]
FREE_SURFACE_Z = 1500.0
CAVITY_WATER_THICKNESS_M = 100.0
VERTICAL_ELEMENT_SIZE_M = 25.0
FREQUENCY_HZ = 10.0
DEFAULT_DT_S = 0.4e-3
DEFAULT_TOTAL_TIME_S = 10.8
MIN_ICE_ELEMENTS = 2
MIN_CAVITY_ELEMENTS = 4


def case_id(thickness_m: int) -> str:
    return f"h{thickness_m:03d}m"


def replace_source_depth(source_text: str, thickness_m: int) -> tuple[str, float]:
    ice_thickness = float(thickness_m)
    depth_below_surface = min(10.0, max(2.0, 0.35 * ice_thickness))
    zs = FREE_SURFACE_Z - depth_below_surface

    lines = source_text.splitlines()
    updated: list[str] = []
    for line in lines:
        if line.strip().startswith("zs"):
            updated.append(f"zs                              = {zs:.6f}d0")
        else:
            updated.append(line)
    return "\n".join(updated) + "\n", depth_below_surface


def build_interfaces_text(thickness_m: int) -> tuple[str, dict[str, float | int]]:
    ice_thickness = float(thickness_m)
    ice_base_z = FREE_SURFACE_Z - ice_thickness
    cavity_bottom_z = ice_base_z - CAVITY_WATER_THICKNESS_M
    underlayer_thickness = cavity_bottom_z

    nz_under = max(1, int(round(underlayer_thickness / VERTICAL_ELEMENT_SIZE_M)))
    nz_cavity = max(MIN_CAVITY_ELEMENTS, int(round(CAVITY_WATER_THICKNESS_M / VERTICAL_ELEMENT_SIZE_M)))
    nz_ice = max(MIN_ICE_ELEMENTS, int(round(ice_thickness / VERTICAL_ELEMENT_SIZE_M)))

    text = f"""#
# Generated flat-interface geometry for the ice-thickness sweep.
# z increases upward in the internal mesher convention.
#
# number of interfaces
#
4
#
# interface number 1 (bottom of the mesh)
#
2
-12000.0 0.0
  6000.0 0.0
#
# interface number 2 (top of the full-width solid underlayer)
#
2
-12000.0 {cavity_bottom_z:.1f}
  6000.0 {cavity_bottom_z:.1f}
#
# interface number 3 (bottom of the ice layer)
#
2
-12000.0 {ice_base_z:.1f}
  6000.0 {ice_base_z:.1f}
#
# interface number 4 (free surface)
#
2
-12000.0 {FREE_SURFACE_Z:.1f}
  6000.0 {FREE_SURFACE_Z:.1f}
#
# number of spectral elements per layer in the vertical direction
#
# layer 1: full-width solid underlayer
{nz_under}
#
# layer 2: 100 m cavity-water / grounded-transition band
{nz_cavity}
#
# layer 3: ice layer
{nz_ice}
"""
    metadata = {
        "ice_thickness_m": ice_thickness,
        "ice_base_z": ice_base_z,
        "cavity_bottom_z": cavity_bottom_z,
        "nz_under": nz_under,
        "nz_cavity": nz_cavity,
        "nz_ice": nz_ice,
        "underlayer_dz_m": underlayer_thickness / nz_under,
        "cavity_dz_m": CAVITY_WATER_THICKNESS_M / nz_cavity,
        "ice_dz_m": ice_thickness / nz_ice,
    }
    return text, metadata


def replace_mesh_block(par_text: str, metadata: dict[str, float | int]) -> str:
    lines = par_text.splitlines()
    updated: list[str] = []
    nz_under = int(metadata["nz_under"])
    nz_cavity = int(metadata["nz_cavity"])
    nz_ice = int(metadata["nz_ice"])
    layer2_start = nz_under + 1
    layer2_end = nz_under + nz_cavity
    layer3_start = layer2_end + 1
    layer3_end = layer2_end + nz_ice

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("title"):
            updated.append(
                "title                           = Grounding-line starter model: thickness sweep"
            )
            i += 1
            continue
        if line.startswith("nbregions"):
            updated.append("nbregions                       = 4")
            updated.append(f"1  360  1 {nz_under} 2")
            updated.append(f"1  240 {layer2_start} {layer2_end} 1")
            updated.append(f"241 360 {layer2_start} {layer2_end} 2")
            updated.append(f"1  360 {layer3_start} {layer3_end} 3")
            i += 5
            continue
        updated.append(line)
        i += 1

    return "\n".join(updated) + "\n"


def replace_time_stepping(par_text: str, metadata: dict[str, float | int]) -> tuple[str, float, int]:
    min_dz = min(
        float(metadata["underlayer_dz_m"]),
        float(metadata["cavity_dz_m"]),
        float(metadata["ice_dz_m"]),
    )
    dt = min(DEFAULT_DT_S, DEFAULT_DT_S * min_dz / VERTICAL_ELEMENT_SIZE_M)
    nstep = int(round(DEFAULT_TOTAL_TIME_S / dt))
    if nstep <= 0:
        raise ValueError("Computed non-positive NSTEP")

    lines = par_text.splitlines()
    updated: list[str] = []
    for line in lines:
        if line.startswith("NSTEP"):
            updated.append(f"NSTEP                           = {nstep}")
        elif line.startswith("DT"):
            updated.append(f"DT                              = {dt:.10f}d0")
        else:
            updated.append(line)
    return "\n".join(updated) + "\n", dt, nstep


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


def load_existing_metadata(results_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not results_dir.exists():
        return rows
    for case_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metadata_path = case_dir / "metadata.json"
        if metadata_path.exists():
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda item: float(item["ice_thickness_m"]))
    return rows


def main() -> None:
    data_dir = ROOT / "DATA"
    output_dir = ROOT / "OUTPUT_FILES"
    results_dir = Path(__file__).resolve().with_name("results_icethickness_10hz")
    summary_path = results_dir / "summary.csv"

    original_par = load_text(data_dir / "Par_file")
    original_interfaces = load_text(data_dir / "interfaces_groundingline.dat")
    original_source = load_text(data_dir / "SOURCE")
    original_stations = load_text(data_dir / "STATIONS") if (data_dir / "STATIONS").exists() else None

    backup_dir = Path(__file__).resolve().with_name("_output_files_backup_icethickness")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        shutil.copytree(output_dir, backup_dir)

    summary_rows = load_existing_metadata(results_dir)
    if summary_rows:
        write_summary(summary_path, summary_rows)

    try:
        for thickness_m in THICKNESSES_M:
            case_dir = results_dir / case_id(thickness_m)
            metadata_path = case_dir / "metadata.json"
            if metadata_path.exists():
                print(f"=== Ice thickness {thickness_m} m already done; skipping ===", flush=True)
                continue

            print(f"=== Running ice thickness {thickness_m} m at f0={FREQUENCY_HZ:g} Hz ===", flush=True)
            interfaces_text, metadata = build_interfaces_text(thickness_m)
            par_text = replace_mesh_block(original_par, metadata)
            par_text, dt_s, nstep = replace_time_stepping(par_text, metadata)
            source_text = replace_source_frequency(original_source, FREQUENCY_HZ)
            source_text, source_depth_below_surface = replace_source_depth(source_text, thickness_m)

            save_text(data_dir / "interfaces_groundingline.dat", interfaces_text)
            save_text(data_dir / "Par_file", par_text)
            save_text(data_dir / "SOURCE", source_text)
            run_solver(ROOT)

            case_dir.mkdir(parents=True, exist_ok=True)
            save_compact_gather(output_dir, case_dir / "surface_gather.npz")

            reflection = estimate_case_reflection(output_dir, FREQUENCY_HZ)
            row = {
                "case_id": case_id(thickness_m),
                "geometry": "supported_cavity_100m",
                "source_frequency_hz": FREQUENCY_HZ,
                "source_depth_below_surface_m": source_depth_below_surface,
                "dt_s": dt_s,
                "nstep": nstep,
                **metadata,
                **reflection,
            }
            metadata_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            summary_rows.append(row)
            summary_rows.sort(key=lambda item: float(item["ice_thickness_m"]))
            write_summary(summary_path, summary_rows)
    finally:
        save_text(data_dir / "Par_file", original_par)
        save_text(data_dir / "interfaces_groundingline.dat", original_interfaces)
        save_text(data_dir / "SOURCE", original_source)
        if original_stations is not None:
            save_text(data_dir / "STATIONS", original_stations)
        restore_output_files(backup_dir, output_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


if __name__ == "__main__":
    main()
