#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil

from run_grounded_material_sweep import (
    ROOT,
    estimate_case_reflection,
    load_text,
    prepare_binaries,
    run_solver,
    save_compact_gather,
    save_text,
    write_summary,
)


RESULTS_DIR = Path(__file__).resolve().with_name("results_seafloorshape_nz1_10hz")
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

XMIN = -12000.0
XMAX = 6000.0

# Near-pinch-out geometry:
# keep a 100 m cavity far from the grounding line, then taper to 30 m over
# the last two kilometers before x = 0. A more aggressive 10 m taper produced
# negative Jacobians in the internal mesher.
RAMP_START_X_M = -2000.0
RAMP_MID_X_M = -1000.0
RAMP_END_X_M = 0.0
UNDERLAYER_Z_FAR_M = 1100.0
UNDERLAYER_Z_MID_M = 1135.0
UNDERLAYER_Z_NEAR_GL_M = 1170.0
ICE_BASE_Z_M = 1200.0


def parfile_with_single_cavity_element(par_text: str) -> str:
    lines = par_text.splitlines()
    updated: list[str] = []
    region_block_replaced = False
    skip_old_region_lines = 0
    for line in lines:
        if skip_old_region_lines > 0:
            skip_old_region_lines -= 1
            continue
        stripped = line.strip()
        if stripped.startswith("#   nz = 56..60: x < 0 water cavity, x > 0 grounded substrate"):
            updated.append("#   nz = 56      : x < 0 water cavity, x > 0 grounded substrate")
        elif stripped.startswith("#   nz = 61..75: ice layer"):
            updated.append("#   nz = 57..71 : ice layer")
        elif stripped == "nbregions                       = 4":
            updated.append("nbregions                       = 4")
            updated.append("1  360  1 55 2")
            updated.append("1  240 56 56 1")
            updated.append("241 360 56 56 2")
            updated.append("1  360 57 71 3")
            region_block_replaced = True
            skip_old_region_lines = 4
        elif line.startswith("NSTEP"):
            updated.append("NSTEP                           = 54000")
        elif line.startswith("DT"):
            updated.append("DT                              = 0.2d-3")
        elif line.startswith("output_color_image"):
            updated.append("output_color_image              = .false.")
        elif line.startswith("output_grid_Gnuplot"):
            updated.append("output_grid_Gnuplot             = .false.")
        elif line.startswith("output_grid_ASCII"):
            updated.append("output_grid_ASCII               = .false.")
        else:
            updated.append(line)
    if not region_block_replaced:
        raise RuntimeError("Could not replace nbregions block for the nz1 cavity comparison")
    return "\n".join(updated) + "\n"


def flat_interfaces() -> str:
    return """#
# Flat-interface geometry for the grounding-line starter example.
# Coordinates use the internal mesher convention with z increasing upward.
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
-12000.0 1100.0
  6000.0 1100.0
#
# interface number 3 (bottom of the ice layer)
#
2
-12000.0 1200.0
  6000.0 1200.0
#
# interface number 4 (free surface)
#
2
-12000.0 1500.0
  6000.0 1500.0
#
# number of spectral elements per layer in the vertical direction
#
# layer 1: full-width solid underlayer thickness = 1100 m -> dz = 20 m
55
#
# layer 2: cavity/grounded transition band thickness = 100 m -> one element
1
#
# layer 3: ice thickness = 300 m -> dz = 20 m
15
"""


def sloping_interfaces() -> str:
    return f"""#
# Near-pinch-out seafloor geometry for the grounding-line starter example.
# The cavity remains 100 m thick far from the grounding line, then the seafloor
# ramps upward over the last two kilometers before x = 0 so the water thickness
# is 30 m at the grounding line instead of pinching out exactly.
#
# number of interfaces
#
4
#
# interface number 1 (bottom of the mesh)
#
2
{XMIN:.1f} 0.0
 {XMAX:.1f} 0.0
#
# interface number 2 (top of the full-width solid underlayer / seafloor)
#
5
{XMIN:.1f} {UNDERLAYER_Z_FAR_M:.1f}
{RAMP_START_X_M:.1f} {UNDERLAYER_Z_FAR_M:.1f}
{RAMP_MID_X_M:.1f} {UNDERLAYER_Z_MID_M:.1f}
{RAMP_END_X_M:.1f} {UNDERLAYER_Z_NEAR_GL_M:.1f}
{XMAX:.1f} {UNDERLAYER_Z_NEAR_GL_M:.1f}
#
# interface number 3 (bottom of the ice layer)
#
2
{XMIN:.1f} {ICE_BASE_Z_M:.1f}
 {XMAX:.1f} {ICE_BASE_Z_M:.1f}
#
# interface number 4 (free surface)
#
2
{XMIN:.1f} 1500.0
 {XMAX:.1f} 1500.0
#
# number of spectral elements per layer in the vertical direction
#
# layer 1: full-width solid underlayer thickness = 1100-1170 m
55
#
# layer 2: cavity/grounded transition band thickness = 30-100 m -> one element
1
#
# layer 3: ice thickness = 300 m
15
"""


def load_existing_metadata() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not RESULTS_DIR.exists():
        return rows
    for case_id in ("flat_100m_nz1", "slope_30m_nz1_dt02"):
        metadata_path = RESULTS_DIR / case_id / "metadata.json"
        if metadata_path.exists():
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return rows


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


def run_case(case_id: str, geometry_name: str, interfaces_text: str, notes: str) -> dict[str, float | str]:
    data_dir = ROOT / "DATA"
    output_dir = ROOT / "OUTPUT_FILES"
    save_text(data_dir / "interfaces_groundingline.dat", interfaces_text)
    run_solver(ROOT)

    case_dir = RESULTS_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    save_compact_gather(output_dir, case_dir / "surface_gather.npz")
    reflection = estimate_case_reflection(output_dir, 10.0)
    metadata = {
        "case_id": case_id,
        "geometry": geometry_name,
        "source_frequency_hz": 10.0,
        "cavity_far_left_thickness_m": 100.0,
        "cavity_min_thickness_m": 100.0 if case_id == "flat_100m_nz1" else 30.0,
        "layer2_vertical_elements": 1,
        "ramp_start_x_m": RAMP_START_X_M if case_id == "slope_30m_nz1_dt02" else "",
        "ramp_end_x_m": RAMP_END_X_M if case_id == "slope_30m_nz1_dt02" else "",
        "notes": notes,
        **reflection,
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    data_dir = ROOT / "DATA"
    output_dir = ROOT / "OUTPUT_FILES"

    original_par = load_text(data_dir / "Par_file")
    original_source = load_text(data_dir / "SOURCE")
    original_interfaces = load_text(data_dir / "interfaces_groundingline.dat")
    original_stations = load_text(data_dir / "STATIONS") if (data_dir / "STATIONS").exists() else None

    backup_dir = Path(__file__).resolve().with_name("_output_files_backup_seafloorshape")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        shutil.copytree(output_dir, backup_dir)

    prepare_binaries(ROOT)
    summary_rows = load_existing_metadata()
    if summary_rows:
        write_summary(SUMMARY_PATH, summary_rows)

    try:
        save_text(data_dir / "Par_file", parfile_with_single_cavity_element(original_par))
        cases = [
            (
                "flat_100m_nz1",
                "flat_supported_cavity",
                flat_interfaces(),
                "Flat 100 m cavity thickness all the way to the grounding line, using one element in the cavity layer.",
            ),
            (
                "slope_30m_nz1_dt02",
                "sloping_near_pinchout_cavity",
                sloping_interfaces(),
                "Near-pinch-out cavity: 100 m far left, tapering to 30 m at x = 0, using one element in the cavity layer and DT = 0.2 ms for stability.",
            ),
        ]
        existing_ids = {str(row["case_id"]) for row in summary_rows}
        for case_id, geometry_name, interfaces_text, notes in cases:
            if case_id in existing_ids:
                print(f"=== {case_id} already done; skipping ===", flush=True)
                continue
            print(f"=== Running {case_id} ({geometry_name}) ===", flush=True)
            metadata = run_case(case_id, geometry_name, interfaces_text, notes)
            summary_rows.append(metadata)
            write_summary(SUMMARY_PATH, summary_rows)
    finally:
        save_text(data_dir / "Par_file", original_par)
        save_text(data_dir / "SOURCE", original_source)
        save_text(data_dir / "interfaces_groundingline.dat", original_interfaces)
        if original_stations is not None:
            save_text(data_dir / "STATIONS", original_stations)
        restore_output_files(backup_dir, output_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


if __name__ == "__main__":
    main()
