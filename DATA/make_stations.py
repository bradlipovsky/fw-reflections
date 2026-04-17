#!/usr/bin/env python3
"""Generate a simple two-line receiver array for the grounding-line example."""

from pathlib import Path

# Key editable receiver parameters
X_START = -3500.0
X_END = -200.0
DX = 50.0
Z_SURFACE_NEAR = 1499.0
Z_SHALLOW = 1450.0
NETWORK = "GL"


def build_line(prefix: str, z_value: float, start_index: int):
    entries = []
    count = int(round((X_END - X_START) / DX)) + 1
    for i in range(count):
        x = X_START + i * DX
        station = f"{prefix}{start_index + i:04d}"
        entries.append(
            f"{station:<8s} {NETWORK:<8s} {x:16.6f} {z_value:16.6f} 0.0 0.0"
        )
    return entries


def main():
    lines = []
    lines.extend(build_line("S", Z_SURFACE_NEAR, 1))
    lines.extend(build_line("B", Z_SHALLOW, 1))
    destination = Path(__file__).with_name("STATIONS")
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines)} stations to {destination}")


if __name__ == "__main__":
    main()
