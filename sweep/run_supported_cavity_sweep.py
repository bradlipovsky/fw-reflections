#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil

from run_grounded_material_sweep import (
    ROOT,
    estimate_case_reflection,
    filter_surface_stations,
    load_text,
    mix_material,
    prepare_binaries,
    replace_material_2,
    replace_source_frequency,
    run_solver,
    save_compact_gather,
    save_text,
    write_summary,
)


FREQUENCIES_HZ = [10.0, 5.0, 1.0]

# Keep this follow-on sweep intentionally small so we can compare the effect of
# the new cavity-support geometry against the archived deep-cavity results.
CASE_SETTINGS = [
    (0.0, 1.0),
    (1.0 / 3.0, 0.7),
    (2.0 / 3.0, 0.4),
    (1.0, 0.4),
]

CASE_IDS = [mix_material(wf, ss).case_id for wf, ss in CASE_SETTINGS]


def build_cases():
    return [mix_material(wf, ss) for wf, ss in CASE_SETTINGS]


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


def load_existing_metadata(results_dir: Path, case_order: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for case_id in case_order:
        metadata_path = results_dir / case_id / "metadata.json"
        if metadata_path.exists():
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return rows


def main() -> None:
    data_dir = ROOT / "DATA"
    output_dir = ROOT / "OUTPUT_FILES"
    original_par = load_text(data_dir / "Par_file")
    original_source = load_text(data_dir / "SOURCE")
    original_stations = load_text(data_dir / "STATIONS") if (data_dir / "STATIONS").exists() else None

    backup_dir = Path(__file__).resolve().with_name("_output_files_backup_supported_cavity")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        shutil.copytree(output_dir, backup_dir)

    cases = build_cases()
    try:
        for frequency_hz in FREQUENCIES_HZ:
            freq_label = f"{frequency_hz:g}".replace(".", "p")
            results_dir = Path(__file__).resolve().with_name(f"results_supportedcavity_{freq_label}hz")
            summary_path = results_dir / "summary.csv"
            summary_rows = load_existing_metadata(results_dir, CASE_IDS)
            if summary_rows:
                write_summary(summary_path, summary_rows)

            save_text(data_dir / "SOURCE", replace_source_frequency(original_source, frequency_hz))
            for case in cases:
                case_dir = results_dir / case.case_id
                metadata_path = case_dir / "metadata.json"
                if metadata_path.exists():
                    print(
                        f"=== Supported cavity: {case.case_id} at f0={frequency_hz:g} Hz already done; skipping ===",
                        flush=True,
                    )
                    continue
                print(
                    f"=== Supported cavity: {case.case_id} at f0={frequency_hz:g} Hz ===",
                    flush=True,
                )
                save_text(data_dir / "Par_file", replace_material_2(original_par, case))
                run_solver(ROOT)

                case_dir.mkdir(parents=True, exist_ok=True)
                save_compact_gather(output_dir, case_dir / "surface_gather.npz")

                reflection = estimate_case_reflection(output_dir, frequency_hz)
                metadata = {
                    "case_id": case.case_id,
                    "geometry": "supported_cavity_100m",
                    "source_frequency_hz": frequency_hz,
                    "water_fraction": case.water_fraction,
                    "shear_scale": case.shear_scale,
                    "rho": case.rho,
                    "vp": case.vp,
                    "vs": case.vs,
                    "zp_ratio_to_ice": case.zp_ratio_to_ice,
                    "zs_ratio_to_ice": case.zs_ratio_to_ice,
                    **reflection,
                }
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                summary_rows.append(metadata)
                write_summary(summary_path, summary_rows)
    finally:
        save_text(data_dir / "Par_file", original_par)
        save_text(data_dir / "SOURCE", original_source)
        if original_stations is not None:
            save_text(data_dir / "STATIONS", original_stations)
        restore_output_files(backup_dir, output_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


if __name__ == "__main__":
    main()
