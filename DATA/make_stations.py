#!/usr/bin/env python3
"""Generate a simple two-line receiver array for the grounding-line example."""

from pathlib import Path

# Key editable receiver parameters
# Keep the array centered on the grounding line with a 4 km total aperture.
X_START = -2000.0
X_END = 2000.0
DX = 6.28
Z_SURFACE_NEAR = 1499.0
Z_SHALLOW = 1450.0
NETWORK = "GL"


def build_line(prefix: str, z_value: float, start_index: int):
    entries = []
    x_values = []
    x = X_START
    while x <= X_END + 1.0e-9:
        x_values.append(min(x, X_END))
        x += DX
    if abs(x_values[-1] - X_END) > 1.0e-9:
        x_values.append(X_END)

    for i, x in enumerate(x_values):
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
