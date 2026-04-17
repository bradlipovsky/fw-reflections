#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from io_utils import load_gather
from synthetic import project_velocity, synthesize_das


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple gauge-length DAS strain-rate product from SPECFEM2D or spec2npy output."
    )
    parser.add_argument(
        "--input",
        default="../OUTPUT_FILES",
        help="Directory of SPECFEM .semv files or a .npz/.npy gather file",
    )
    parser.add_argument(
        "--station-prefix",
        default="S",
        help="Receiver prefix to use, e.g. S for near-surface or B for buried",
    )
    parser.add_argument("--gauge-length", type=float, default=50.0, help="Gauge length in meters")
    parser.add_argument(
        "--channel-spacing",
        type=float,
        default=25.0,
        help="Target DAS channel spacing in meters",
    )
    parser.add_argument(
        "--fiber-angle",
        type=float,
        default=0.0,
        help="Fiber angle in degrees measured from +x; 0 means horizontal DAS along the profile",
    )
    parser.add_argument(
        "--output",
        default="products/surface_das.npz",
        help="Output NPZ path for the DAS product",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    gather_x = load_gather(input_path, component="BXX", station_prefix=args.station_prefix)
    gather_z = None
    try:
        gather_z = load_gather(input_path, component="BXZ", station_prefix=args.station_prefix)
    except Exception:
        gather_z = None

    projected = project_velocity(gather_x, gather_z=gather_z, fiber_angle_deg=args.fiber_angle)
    record = synthesize_das(
        projected,
        gauge_length_m=args.gauge_length,
        channel_spacing_m=args.channel_spacing,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        time=record.time,
        x=record.x,
        strain_rate=record.strain_rate,
        particle_velocity=record.particle_velocity,
        gauge_length_m=record.gauge_length_m,
        channel_spacing_m=record.channel_spacing_m,
        fiber_angle_deg=args.fiber_angle,
        source=str(input_path),
        station_prefix=args.station_prefix,
    )
    print(f"Wrote DAS product to {output_path}")


if __name__ == "__main__":
    main()
