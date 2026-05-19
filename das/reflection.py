from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthetic import DasRecord, fk_filter_velocity_band


@dataclass
class DirectionalBeam:
    """Moveout-aligned stack for one propagation direction."""

    time: np.ndarray
    x: np.ndarray
    aligned_traces: np.ndarray
    stack: np.ndarray
    direction: str
    phase_velocity_mps: float
    x_anchor_m: float


@dataclass
class ReflectionEstimate:
    """Signed reflection-coefficient estimate from DAS wavelets."""

    coefficient: float
    correlation: float
    alignment_lag_s: float
    incident_center_s: float
    reflected_center_s: float
    time_window_s: np.ndarray
    incident_wavelet: np.ndarray
    reflected_wavelet: np.ndarray
    incident_beam: DirectionalBeam
    reflected_beam: DirectionalBeam
    phase_velocity_mps: float
    phase_velocity_min_mps: float
    phase_velocity_max_mps: float
    x_min_m: float
    x_max_m: float
    window_halfwidth_s: float


def _select_channels(record: DasRecord, x_min_m: float, x_max_m: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (record.x >= x_min_m) & (record.x <= x_max_m)
    if not np.any(mask):
        raise ValueError("No DAS channels fall inside the requested x window")
    return record.x[mask], record.strain_rate[mask, :]


def _direction_sign(direction: str) -> float:
    if direction == "right":
        return 1.0
    if direction == "left":
        return -1.0
    raise ValueError("direction must be 'left' or 'right'")


def align_directional_energy(
    record: DasRecord,
    phase_velocity_mps: float,
    direction: str,
    x_min_m: float,
    x_max_m: float,
    x_anchor_m: float | None = None,
) -> DirectionalBeam:
    """
    Align a directional DAS gather by linear moveout and stack it.

    For a right-moving packet, traces are shifted by ``+(x - x_anchor)/v``.
    For a left-moving packet, they are shifted by ``-(x - x_anchor)/v``.
    """

    if phase_velocity_mps <= 0.0:
        raise ValueError("phase_velocity_mps must be positive")

    time = np.asarray(record.time, dtype=float)
    x, traces = _select_channels(record, x_min_m=x_min_m, x_max_m=x_max_m)
    anchor = float(np.mean(x) if x_anchor_m is None else x_anchor_m)
    sign = _direction_sign(direction)

    aligned = np.empty_like(traces)
    for idx, x_i in enumerate(x):
        shift_s = sign * (x_i - anchor) / phase_velocity_mps
        aligned[idx, :] = np.interp(
            time + shift_s,
            time,
            traces[idx, :],
            left=0.0,
            right=0.0,
        )

    stack = np.mean(aligned, axis=0)
    return DirectionalBeam(
        time=time,
        x=x,
        aligned_traces=aligned,
        stack=stack,
        direction=direction,
        phase_velocity_mps=float(phase_velocity_mps),
        x_anchor_m=anchor,
    )


def pick_beam_peak_time(
    beam: DirectionalBeam,
    search_tmin_s: float,
    search_tmax_s: float,
) -> float:
    """Pick the strongest packet center inside a search window using the beam envelope."""

    mask = (beam.time >= search_tmin_s) & (beam.time <= search_tmax_s)
    if not np.any(mask):
        raise ValueError("Search window does not overlap the beam time axis")
    subset = analytic_envelope(beam.stack[mask])
    if subset.size == 0:
        raise ValueError("Search window is empty")
    local_index = int(np.argmax(subset))
    return float(beam.time[mask][local_index])


def analytic_envelope(data: np.ndarray) -> np.ndarray:
    """Return the instantaneous amplitude using an FFT-based Hilbert transform."""

    array = np.asarray(data, dtype=float)
    n = array.size
    if n == 0:
        return array.copy()

    spectrum = np.fft.fft(array)
    hilbert = np.zeros(n, dtype=float)
    if n % 2 == 0:
        hilbert[0] = 1.0
        hilbert[n // 2] = 1.0
        hilbert[1 : n // 2] = 2.0
    else:
        hilbert[0] = 1.0
        hilbert[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * hilbert)
    return np.abs(analytic)


def extract_wavelet(
    time: np.ndarray,
    data: np.ndarray,
    center_s: float,
    halfwidth_s: float,
    taper_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and taper a windowed wavelet centered on ``center_s``."""

    if halfwidth_s <= 0.0:
        raise ValueError("halfwidth_s must be positive")

    mask = (time >= center_s - halfwidth_s) & (time <= center_s + halfwidth_s)
    if not np.any(mask):
        raise ValueError("Wavelet window does not overlap the provided time axis")

    t_window = np.asarray(time[mask], dtype=float)
    wavelet = np.asarray(data[mask], dtype=float).copy()
    wavelet = wavelet - np.mean(wavelet)

    n = len(wavelet)
    ntaper = max(1, min(int(round(taper_fraction * n)), n // 2))
    if ntaper > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ntaper)))
        window = np.ones(n, dtype=float)
        window[:ntaper] = ramp
        window[-ntaper:] = ramp[::-1]
        wavelet *= window

    return t_window - center_s, wavelet


def _trim_wavelets(
    incident_wavelet: np.ndarray,
    reflected_wavelet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim two wavelets to their common centered overlap."""

    inc = np.asarray(incident_wavelet, dtype=float)
    ref = np.asarray(reflected_wavelet, dtype=float)
    if inc.shape != ref.shape:
        n = min(inc.size, ref.size)
        if n < 2:
            raise ValueError("Wavelets must have overlapping samples")
        inc_start = (inc.size - n) // 2
        ref_start = (ref.size - n) // 2
        inc = inc[inc_start : inc_start + n]
        ref = ref[ref_start : ref_start + n]
    return inc, ref


def align_wavelets_for_comparison(
    incident_wavelet: np.ndarray,
    reflected_wavelet: np.ndarray,
    dt_s: float,
    max_lag_s: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Align reflected and incident wavelets by the lag with the strongest correlation.

    The search is intentionally limited to a small lag so that we correct modest
    envelope/pick offsets without allowing arbitrary cycle skipping.
    """

    inc, ref = _trim_wavelets(incident_wavelet, reflected_wavelet)
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    max_lag_samples = max(0, min(int(round(max_lag_s / dt_s)), inc.size // 3))
    best_score = None
    best_corr = 0.0
    best_lag = 0
    best_inc = inc
    best_ref = ref

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            inc_seg = inc[: inc.size - lag]
            ref_seg = ref[lag:]
        else:
            inc_seg = inc[-lag:]
            ref_seg = ref[: ref.size + lag]
        if inc_seg.size < 8:
            continue
        inc_norm = float(np.linalg.norm(inc_seg))
        ref_norm = float(np.linalg.norm(ref_seg))
        if inc_norm == 0.0 or ref_norm == 0.0:
            corr = 0.0
        else:
            corr = float(np.dot(inc_seg, ref_seg) / (inc_norm * ref_norm))
        score = abs(corr)
        if best_score is None or score > best_score:
            best_score = score
            best_corr = corr
            best_lag = lag
            best_inc = inc_seg
            best_ref = ref_seg

    return best_inc, best_ref, float(best_lag * dt_s), float(best_corr)


def signed_projection_coefficient(
    incident_wavelet: np.ndarray,
    reflected_wavelet: np.ndarray,
    dt_s: float,
    max_lag_s: float = 0.06,
) -> tuple[float, float, float]:
    """
    Return a signed packet-amplitude ratio and normalized correlation.

    The magnitude comes from the envelope energy ratio, which is more stable for
    dispersed reflected packets than a pointwise least-squares projection. The
    sign comes from the correlation of the best small-lag alignment.
    """

    inc, ref, lag_s, corr = align_wavelets_for_comparison(
        incident_wavelet=incident_wavelet,
        reflected_wavelet=reflected_wavelet,
        dt_s=dt_s,
        max_lag_s=max_lag_s,
    )

    inc_env = analytic_envelope(inc)
    ref_env = analytic_envelope(ref)
    denom = float(np.linalg.norm(inc_env))
    if denom <= 0.0:
        raise ValueError("Incident wavelet has zero energy")
    coeff_mag = float(np.linalg.norm(ref_env) / denom)
    coeff = float(np.sign(corr) * coeff_mag) if corr != 0.0 else 0.0
    return coeff, corr, lag_s


def estimate_reflection_coefficient(
    das_record: DasRecord,
    phase_velocity_mps: float,
    phase_velocity_halfwidth_mps: float,
    x_min_m: float,
    x_max_m: float,
    incident_search_window_s: tuple[float, float],
    reflected_search_window_s: tuple[float, float],
    wavelet_halfwidth_s: float = 0.35,
    x_anchor_m: float | None = None,
    fk_transition_fraction: float = 0.25,
    fk_min_frequency_hz: float = 0.5,
    max_alignment_lag_s: float = 0.06,
) -> ReflectionEstimate:
    """
    Estimate a signed Rayleigh-wave reflection coefficient from a DAS record.

    The current estimator is a pragmatic first-pass workflow:

    1. FK filter into a narrow apparent Rayleigh-wave speed band.
    2. Moveout-align that band-limited gather twice: once as a right-moving
       packet and once as a left-moving packet.
    3. Pick an incident and reflected packet in separate search windows.
    4. Compute the signed least-squares ratio between the two stacked wavelets.

    Because both packets are measured in the same left-side ice aperture, the
    result preserves polarity and is easy to compare across parameter sweeps.
    It is still an approximate reflection coefficient because it neglects any
    residual geometrical spreading and attenuation differences between the two
    paths.
    """

    phase_velocity_min = max(1.0, phase_velocity_mps - phase_velocity_halfwidth_mps)
    phase_velocity_max = phase_velocity_mps + phase_velocity_halfwidth_mps

    band_limited = fk_filter_velocity_band(
        das_record,
        phase_velocity_min=phase_velocity_min,
        phase_velocity_max=phase_velocity_max,
        direction=None,
        transition_fraction=fk_transition_fraction,
        min_frequency_hz=fk_min_frequency_hz,
    )

    incident_beam = align_directional_energy(
        band_limited,
        phase_velocity_mps=phase_velocity_mps,
        direction="right",
        x_min_m=x_min_m,
        x_max_m=x_max_m,
        x_anchor_m=x_anchor_m,
    )
    reflected_beam = align_directional_energy(
        band_limited,
        phase_velocity_mps=phase_velocity_mps,
        direction="left",
        x_min_m=x_min_m,
        x_max_m=x_max_m,
        x_anchor_m=incident_beam.x_anchor_m if x_anchor_m is None else x_anchor_m,
    )

    incident_center = pick_beam_peak_time(
        incident_beam,
        search_tmin_s=incident_search_window_s[0],
        search_tmax_s=incident_search_window_s[1],
    )
    reflected_center = pick_beam_peak_time(
        reflected_beam,
        search_tmin_s=reflected_search_window_s[0],
        search_tmax_s=reflected_search_window_s[1],
    )

    time_window, incident_wavelet = extract_wavelet(
        incident_beam.time,
        incident_beam.stack,
        center_s=incident_center,
        halfwidth_s=wavelet_halfwidth_s,
    )
    _, reflected_wavelet = extract_wavelet(
        reflected_beam.time,
        reflected_beam.stack,
        center_s=reflected_center,
        halfwidth_s=wavelet_halfwidth_s,
    )

    dt_s = float(np.median(np.diff(time_window)))
    coeff, corr, lag_s = signed_projection_coefficient(
        incident_wavelet=incident_wavelet,
        reflected_wavelet=reflected_wavelet,
        dt_s=dt_s,
        max_lag_s=max_alignment_lag_s,
    )

    return ReflectionEstimate(
        coefficient=coeff,
        correlation=corr,
        alignment_lag_s=lag_s,
        incident_center_s=incident_center,
        reflected_center_s=reflected_center,
        time_window_s=time_window,
        incident_wavelet=incident_wavelet,
        reflected_wavelet=reflected_wavelet,
        incident_beam=incident_beam,
        reflected_beam=reflected_beam,
        phase_velocity_mps=float(phase_velocity_mps),
        phase_velocity_min_mps=float(phase_velocity_min),
        phase_velocity_max_mps=float(phase_velocity_max),
        x_min_m=float(x_min_m),
        x_max_m=float(x_max_m),
        window_halfwidth_s=float(wavelet_halfwidth_s),
    )
