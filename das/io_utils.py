from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass
class ReceiverGather:
    """Regular receiver gather with one component per trace."""

    time: np.ndarray
    x: np.ndarray
    z: np.ndarray
    data: np.ndarray
    stations: list[str]
    component: str


def _station_sort_key(name: str) -> tuple[str, int]:
    match = re.match(r"([A-Za-z]+)(\d+)$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def read_station_table(path: str | Path) -> dict[str, dict[str, float | str]]:
    """Read SPECFEM2D's output_list_stations.txt."""

    table: dict[str, dict[str, float | str]] = {}
    with Path(path).open("r", encoding="ascii") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            station, network, x, z = parts[:4]
            table[station] = {
                "network": network,
                "x": float(x),
                "z": float(z),
            }
    return table


def _read_semv_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.loadtxt(path)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"Unexpected trace format in {path}")
    return array[:, 0], array[:, 1]


def load_specfem_component(
    output_dir: str | Path,
    component: str,
    station_prefix: str | None = None,
) -> ReceiverGather:
    """Load one SPECFEM2D component from OUTPUT_FILES."""

    output_dir = Path(output_dir)
    station_table = read_station_table(output_dir / "output_list_stations.txt")

    pattern = f"*.{component}.semv"
    files = sorted(output_dir.glob(pattern))
    if station_prefix is not None:
        files = [path for path in files if path.stem.split(".")[1].startswith(station_prefix)]
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern} found in {output_dir} for prefix {station_prefix!r}"
        )

    traces: list[np.ndarray] = []
    time = None
    stations: list[str] = []
    x_values: list[float] = []
    z_values: list[float] = []

    for path in files:
        _, station, _, _ = path.name.split(".")
        t_vec, trace = _read_semv_trace(path)
        if time is None:
            time = t_vec
        elif not np.allclose(time, t_vec):
            raise ValueError(f"Time vector mismatch in {path}")

        if station not in station_table:
            raise KeyError(f"Station {station} not found in output_list_stations.txt")

        stations.append(station)
        x_values.append(float(station_table[station]["x"]))
        z_values.append(float(station_table[station]["z"]))
        traces.append(trace)

    order = sorted(range(len(stations)), key=lambda idx: _station_sort_key(stations[idx]))
    stations = [stations[idx] for idx in order]
    x = np.asarray([x_values[idx] for idx in order], dtype=float)
    z = np.asarray([z_values[idx] for idx in order], dtype=float)
    data = np.asarray([traces[idx] for idx in order], dtype=float)

    return ReceiverGather(
        time=np.asarray(time, dtype=float),
        x=x,
        z=z,
        data=data,
        stations=stations,
        component=component,
    )


def _load_npz(npz_path: Path) -> ReceiverGather:
    archive = np.load(npz_path, allow_pickle=True)
    required = {"time", "data", "x"}
    missing = required - set(archive.files)
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {sorted(missing)}")

    z = archive["z"] if "z" in archive.files else np.zeros_like(archive["x"], dtype=float)
    stations = (
        archive["stations"].tolist()
        if "stations" in archive.files
        else [f"C{i:04d}" for i in range(len(archive["x"]))]
    )
    component = str(archive["component"]) if "component" in archive.files else "UNKNOWN"

    return ReceiverGather(
        time=np.asarray(archive["time"], dtype=float),
        x=np.asarray(archive["x"], dtype=float),
        z=np.asarray(z, dtype=float),
        data=np.asarray(archive["data"], dtype=float),
        stations=[str(item) for item in stations],
        component=component,
    )


def _load_npy(npy_path: Path) -> ReceiverGather:
    data = np.load(npy_path)
    metadata_path = npy_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{npy_path} was found, but {metadata_path.name} is needed for time/x metadata"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ReceiverGather(
        time=np.asarray(metadata["time"], dtype=float),
        x=np.asarray(metadata["x"], dtype=float),
        z=np.asarray(metadata.get("z", np.zeros(len(metadata["x"]))), dtype=float),
        data=np.asarray(data, dtype=float),
        stations=[str(item) for item in metadata.get("stations", [])]
        or [f"C{i:04d}" for i in range(data.shape[0])],
        component=str(metadata.get("component", "UNKNOWN")),
    )


def load_gather(path: str | Path, component: str | None = None, station_prefix: str | None = None) -> ReceiverGather:
    """Load either SPECFEM output or simple spec2npy-style arrays."""

    path = Path(path)
    if path.is_dir():
        if component is None:
            raise ValueError("A component such as 'BXX' or 'BXZ' is required when loading a directory")
        return load_specfem_component(path, component=component, station_prefix=station_prefix)
    if path.suffix == ".npz":
        return _load_npz(path)
    if path.suffix == ".npy":
        return _load_npy(path)
    raise ValueError(f"Unsupported input type for {path}")


def save_gather_npz(gather: ReceiverGather, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        time=gather.time,
        x=gather.x,
        z=gather.z,
        data=gather.data,
        stations=np.asarray(gather.stations, dtype=object),
        component=gather.component,
    )


def list_station_prefixes(stations: Iterable[str]) -> list[str]:
    return sorted({re.match(r"[A-Za-z]+", station).group(0) for station in stations if re.match(r"[A-Za-z]+", station)})

