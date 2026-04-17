from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from io_utils import ReceiverGather


@dataclass
class DasRecord:
    """Gauge-length DAS approximation from particle velocity gathers."""

    time: np.ndarray
    x: np.ndarray
    strain_rate: np.ndarray
    particle_velocity: np.ndarray
    gauge_length_m: float
    channel_spacing_m: float
    fiber_angle_deg: float


def project_velocity(
    gather_x: ReceiverGather,
    gather_z: ReceiverGather | None = None,
    fiber_angle_deg: float = 0.0,
) -> ReceiverGather:
    """Project 2D velocity onto a straight fiber direction."""

    if gather_z is not None:
        if not np.allclose(gather_x.time, gather_z.time):
            raise ValueError("BXX and BXZ gathers do not share the same time vector")
        if not np.allclose(gather_x.x, gather_z.x):
            raise ValueError("BXX and BXZ gathers do not share the same x coordinates")
        if not np.allclose(gather_x.z, gather_z.z):
            raise ValueError("BXX and BXZ gathers do not share the same z coordinates")

    angle = math.radians(fiber_angle_deg)
    projected = gather_x.data * math.cos(angle)
    if gather_z is not None:
        projected = projected + gather_z.data * math.sin(angle)

    return ReceiverGather(
        time=gather_x.time.copy(),
        x=gather_x.x.copy(),
        z=gather_x.z.copy(),
        data=projected,
        stations=gather_x.stations.copy(),
        component=f"fiber_{fiber_angle_deg:.1f}deg",
    )


def _interp_traces(source_x: np.ndarray, source_data: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """Interpolate each time sample along x."""

    interpolated = np.empty((len(target_x), source_data.shape[1]), dtype=float)
    for it in range(source_data.shape[1]):
        interpolated[:, it] = np.interp(target_x, source_x, source_data[:, it])
    return interpolated


def synthesize_das(
    velocity_gather: ReceiverGather,
    gauge_length_m: float,
    channel_spacing_m: float | None = None,
    x_start: float | None = None,
    x_end: float | None = None,
    integrate_to_strain: bool = False,
) -> DasRecord:
    """
    Approximate DAS axial strain rate using a gauge-length difference of particle velocity.

    strain_rate(x, t) ~= [v(x + L/2, t) - v(x - L/2, t)] / L
    """

    if gauge_length_m <= 0.0:
        raise ValueError("gauge_length_m must be positive")

    x = np.asarray(velocity_gather.x, dtype=float)
    data = np.asarray(velocity_gather.data, dtype=float)
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("Receiver coordinates must be strictly increasing in x")

    native_spacing = np.median(np.diff(x))
    spacing = native_spacing if channel_spacing_m is None else float(channel_spacing_m)

    left_limit = float(x.min() + gauge_length_m / 2.0)
    right_limit = float(x.max() - gauge_length_m / 2.0)
    if x_start is not None:
        left_limit = max(left_limit, float(x_start))
    if x_end is not None:
        right_limit = min(right_limit, float(x_end))
    if right_limit <= left_limit:
        raise ValueError("Requested DAS channels do not fit inside the receiver aperture")

    channel_x = np.arange(left_limit, right_limit + 0.5 * spacing, spacing)
    left_x = channel_x - gauge_length_m / 2.0
    right_x = channel_x + gauge_length_m / 2.0

    v_left = _interp_traces(x, data, left_x)
    v_right = _interp_traces(x, data, right_x)
    strain_rate = (v_right - v_left) / gauge_length_m

    if integrate_to_strain:
        dt = float(np.median(np.diff(velocity_gather.time)))
        particle_velocity = np.cumsum(strain_rate, axis=1) * dt
    else:
        particle_velocity = _interp_traces(x, data, channel_x)

    return DasRecord(
        time=velocity_gather.time.copy(),
        x=channel_x,
        strain_rate=strain_rate,
        particle_velocity=particle_velocity,
        gauge_length_m=float(gauge_length_m),
        channel_spacing_m=float(spacing),
        fiber_angle_deg=0.0,
    )


def fk_filter(
    record: DasRecord,
    direction: str = "right",
    taper_width: int = 5,
) -> DasRecord:
    """
    FK filter to isolate left-moving or right-moving waves.

    Parameters
    ----------
    record : DasRecord
        Input DAS record with strain_rate shaped (n_channels, n_times).
    direction : str
        ``'right'`` keeps waves with positive apparent velocity (moving in +x),
        ``'left'`` keeps waves with negative apparent velocity (moving in -x).
    taper_width : int
        Width in wavenumber samples of the cosine taper applied at k=0 to
        avoid a sharp spectral edge. Set to 0 for a hard cut.

    Returns
    -------
    DasRecord
        Filtered record.
    """
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'")

    data = record.strain_rate.copy()  # (nx, nt)
    nx, nt = data.shape

    # 2-D FFT over space (axis 0) and time (axis 1)
    fk = np.fft.fft2(data)

    # Build frequency and wavenumber axes
    freqs = np.fft.fftfreq(nt)   # normalised; sign is what matters
    ks = np.fft.fftfreq(nx)

    # For a wave moving to the right (+x), f and k have the *same* sign
    # (positive f with positive k, negative f with negative k).
    # For a wave moving to the left (-x), f and k have *opposite* signs.
    kk, ff = np.meshgrid(ks, freqs, indexing="ij")  # shape (nx, nt)

    if direction == "right":
        # keep same-sign quadrants
        mask = (kk * ff >= 0).astype(float)
    else:
        # keep opposite-sign quadrants
        mask = (kk * ff <= 0).astype(float)

    # Always keep the zero-frequency and zero-wavenumber lines (DC)
    mask[0, :] = 1.0
    mask[:, 0] = 1.0

    # Cosine taper near k=0 to reduce ringing
    if taper_width > 0:
        for ik in range(1, min(taper_width + 1, nx // 2)):
            weight = 0.5 * (1 - np.cos(np.pi * ik / taper_width))
            # positive k side
            mask[ik, :] = mask[ik, :] * weight + (1.0 - weight)
            # negative k side (symmetric)
            mask[-ik, :] = mask[-ik, :] * weight + (1.0 - weight)

    filtered = np.real(np.fft.ifft2(fk * mask))

    return DasRecord(
        time=record.time.copy(),
        x=record.x.copy(),
        strain_rate=filtered,
        particle_velocity=record.particle_velocity.copy(),
        gauge_length_m=record.gauge_length_m,
        channel_spacing_m=record.channel_spacing_m,
        fiber_angle_deg=record.fiber_angle_deg,
    )


def clip_time_window(record: DasRecord, tmin: float | None = None, tmax: float | None = None) -> DasRecord:
    mask = np.ones_like(record.time, dtype=bool)
    if tmin is not None:
        mask &= record.time >= tmin
    if tmax is not None:
        mask &= record.time <= tmax
    return DasRecord(
        time=record.time[mask],
        x=record.x.copy(),
        strain_rate=record.strain_rate[:, mask],
        particle_velocity=record.particle_velocity[:, mask],
        gauge_length_m=record.gauge_length_m,
        channel_spacing_m=record.channel_spacing_m,
        fiber_angle_deg=record.fiber_angle_deg,
    )
