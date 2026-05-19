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


def _cosine_taper_1d(
    length: int,
    taper_fraction: float,
    taper_left: bool = True,
    taper_right: bool = True,
) -> np.ndarray:
    """Cosine taper with a flat center and optional left/right edges."""

    if length <= 1 or taper_fraction <= 0.0:
        return np.ones(length, dtype=float)

    ntaper = int(round(taper_fraction * length))
    ntaper = max(1, min(ntaper, length // 2))
    window = np.ones(length, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ntaper)))
    if taper_left:
        window[:ntaper] = ramp
    if taper_right:
        window[-ntaper:] = ramp[::-1]
    return window


def _cosine_band_weight(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    taper_width: float,
) -> np.ndarray:
    """
    Smoothly window a scalar band with cosine shoulders.

    Values inside ``[vmin, vmax]`` get weight 1.0, values outside the wider
    ``[vmin - taper_width, vmax + taper_width]`` interval get 0.0, and the
    transition zone is cosine tapered to reduce ringing from a hard FK mask.
    """

    if taper_width <= 0.0:
        return ((values >= vmin) & (values <= vmax)).astype(float)

    weight = np.zeros_like(values, dtype=float)
    core = (values >= vmin) & (values <= vmax)
    low_taper = (values >= vmin - taper_width) & (values < vmin)
    high_taper = (values > vmax) & (values <= vmax + taper_width)

    weight[core] = 1.0
    weight[low_taper] = 0.5 * (
        1.0 - np.cos(np.pi * (values[low_taper] - (vmin - taper_width)) / taper_width)
    )
    weight[high_taper] = 0.5 * (
        1.0 + np.cos(np.pi * (values[high_taper] - vmax) / taper_width)
    )
    return weight


def _apply_fk_mask(
    data: np.ndarray,
    mask_builder,
    dt: float,
    dx: float,
    time_pad_fraction_before: float = 2.0,
    time_pad_fraction_after: float = 1.0,
    space_pad_fraction: float = 0.5,
    time_taper_fraction: float = 0.05,
    space_taper_fraction: float = 0.05,
) -> np.ndarray:
    """
    Apply an FK-domain mask after tapering and zero-padding.

    The taper and padding reduce circular wrap-around and edge ringing. The
    extra pre-event padding is intentionally larger than the post-event padding
    because the seismic record starts at ``t = 0`` and FK filtering on a finite
    gather otherwise tends to smear energy back onto the first few samples.
    """

    nx, nt = data.shape
    time_window = _cosine_taper_1d(nt, time_taper_fraction, taper_left=False, taper_right=True)
    space_window = _cosine_taper_1d(nx, space_taper_fraction)
    tapered = data * np.outer(space_window, time_window)

    pad_t_before = int(round(time_pad_fraction_before * nt))
    pad_t_after = int(round(time_pad_fraction_after * nt))
    pad_x = int(round(space_pad_fraction * nx))
    padded = np.pad(tapered, ((pad_x, pad_x), (pad_t_before, pad_t_after)), mode="constant")

    nx_pad, nt_pad = padded.shape
    fk = np.fft.fft2(padded)
    freqs = np.fft.fftfreq(nt_pad, d=dt)
    ks = np.fft.fftfreq(nx_pad, d=dx)
    kk, ff = np.meshgrid(ks, freqs, indexing="ij")
    mask = mask_builder(kk, ff)
    filtered = np.real(np.fft.ifft2(fk * mask))

    return filtered[pad_x:pad_x + nx, pad_t_before:pad_t_before + nt]


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
    dt = float(np.median(np.diff(record.time)))
    dx = float(np.median(np.diff(record.x)))

    def build_mask(kk: np.ndarray, ff: np.ndarray) -> np.ndarray:
        if direction == "right":
            mask = (kk * ff >= 0).astype(float)
        else:
            mask = (kk * ff <= 0).astype(float)

        # Keep only the zero-wavenumber DC line to avoid injecting an
        # arbitrary time-constant spatial pattern.
        zero_k = np.abs(kk) == 0.0
        zero_f = np.abs(ff) == 0.0
        mask[zero_k & zero_f] = 1.0

        if taper_width > 0:
            nx_pad = mask.shape[0]
            for ik in range(1, min(taper_width + 1, nx_pad // 2)):
                weight = 0.5 * (1 - np.cos(np.pi * ik / taper_width))
                mask[ik, :] = mask[ik, :] * weight + (1.0 - weight)
                mask[-ik, :] = mask[-ik, :] * weight + (1.0 - weight)
        return mask

    filtered = _apply_fk_mask(data, build_mask, dt=dt, dx=dx)

    return DasRecord(
        time=record.time.copy(),
        x=record.x.copy(),
        strain_rate=filtered,
        particle_velocity=record.particle_velocity.copy(),
        gauge_length_m=record.gauge_length_m,
        channel_spacing_m=record.channel_spacing_m,
        fiber_angle_deg=record.fiber_angle_deg,
    )


def fk_filter_velocity_band(
    record: DasRecord,
    phase_velocity_min: float,
    phase_velocity_max: float,
    direction: str | None = None,
    transition_fraction: float = 0.25,
    min_frequency_hz: float = 0.5,
) -> DasRecord:
    """
    FK filter that keeps only energy within an apparent phase-velocity band.

    The apparent phase velocity is estimated in the 2-D f-k spectrum as v = f / k,
    using physical frequency (Hz) and wavenumber (cycles/m). The zero-wavenumber
    column is removed because it does not map to a finite phase velocity.
    """

    if phase_velocity_min <= 0.0 or phase_velocity_max <= 0.0:
        raise ValueError("phase velocity bounds must be positive")
    if phase_velocity_max < phase_velocity_min:
        raise ValueError("phase_velocity_max must be >= phase_velocity_min")
    if direction not in (None, "left", "right"):
        raise ValueError("direction must be None, 'left', or 'right'")
    if transition_fraction < 0.0:
        raise ValueError("transition_fraction must be non-negative")
    if min_frequency_hz < 0.0:
        raise ValueError("min_frequency_hz must be non-negative")

    data = record.strain_rate.copy()
    nx, nt = data.shape
    if nx < 2 or nt < 2:
        raise ValueError("FK filtering requires at least two channels and two time samples")

    dt = float(np.median(np.diff(record.time)))
    dx = float(np.median(np.diff(record.x)))
    if dt <= 0.0 or dx <= 0.0:
        raise ValueError("record must have increasing time and x coordinates")

    band_width = max(phase_velocity_max - phase_velocity_min, 1.0)
    transition_width = max(transition_fraction * band_width, 1.0)

    def build_mask(kk: np.ndarray, ff: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(kk, dtype=float)
        nonzero_k = np.abs(kk) > 0.0
        nonzero_f = np.abs(ff) >= min_frequency_hz
        phase_velocity = np.full_like(ff, np.inf, dtype=float)
        phase_velocity[nonzero_k] = np.abs(ff[nonzero_k] / kk[nonzero_k])

        band_weight = _cosine_band_weight(
            phase_velocity,
            phase_velocity_min,
            phase_velocity_max,
            transition_width,
        )
        band_weight *= (nonzero_k & nonzero_f).astype(float)

        if direction == "right":
            direction_weight = (kk * ff >= 0.0).astype(float)
        elif direction == "left":
            direction_weight = (kk * ff <= 0.0).astype(float)
        else:
            direction_weight = np.ones_like(kk, dtype=float)

        mask = band_weight * direction_weight
        return mask

    filtered = _apply_fk_mask(data, build_mask, dt=dt, dx=dx)

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
