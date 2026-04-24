"""Driving action generation for simulation."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _lowpass_noise(
    rng: np.random.Generator,
    n_steps: int,
    sample_rate_hz: float,
    cutoff_hz: float,
    amplitude: float,
) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=n_steps)
    sos = signal.butter(3, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, noise)
    max_abs = float(np.max(np.abs(filtered)))
    if max_abs < 1e-9:
        return np.zeros(n_steps, dtype=np.float64)
    return amplitude * filtered / max_abs


def _add_steer_event(
    rng: np.random.Generator,
    steer_rad: np.ndarray,
    sample_rate_hz: float,
    max_amplitude_rad: float,
) -> None:
    n_steps = steer_rad.shape[0]
    duration = int(rng.uniform(0.4, 1.4) * sample_rate_hz)
    if duration <= 2 or duration >= n_steps:
        return
    start = int(rng.integers(0, n_steps - duration))
    amplitude = rng.uniform(0.4, 0.9) * max_amplitude_rad * rng.choice([-1.0, 1.0])
    steer_rad[start : start + duration] += amplitude


def _add_fx_event(
    rng: np.random.Generator,
    fx_n: np.ndarray,
    sample_rate_hz: float,
    max_amplitude_n: float,
) -> None:
    n_steps = fx_n.shape[0]
    duration = int(rng.uniform(0.5, 2.0) * sample_rate_hz)
    if duration <= 2 or duration >= n_steps:
        return
    start = int(rng.integers(0, n_steps - duration))
    amplitude = rng.uniform(0.6, 1.0) * max_amplitude_n * rng.choice([-1.0, 1.0])
    fx_n[start : start + duration] += amplitude


def _fade_in(actions: np.ndarray, sample_rate_hz: float, duration_s: float = 0.5) -> np.ndarray:
    n_ramp = min(actions.shape[0], max(1, int(round(duration_s * sample_rate_hz))))
    ramp = np.linspace(0.0, 1.0, n_ramp, dtype=np.float64)
    actions[:n_ramp] *= ramp[:, None]
    return actions


def generate_actions(
    config: dict,
    rng: np.random.Generator,
    n_steps: int,
    sample_rate_hz: float,
) -> np.ndarray:
    """Generate steering and longitudinal-force commands.

    Returns:
        Array with shape ``[n_steps, 2]`` containing ``[delta_rad, fx_n]``.
    """

    action_config = config["action"]
    steer_amp_deg = rng.uniform(*action_config["steer_amplitude_deg_range"])
    steer_amp_rad = np.deg2rad(steer_amp_deg)
    fx_amp_n = rng.uniform(*action_config["fx_amplitude_n_range"])

    steer_rad = _lowpass_noise(
        rng,
        n_steps=n_steps,
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=float(action_config["steer_cutoff_hz"]),
        amplitude=steer_amp_rad,
    )
    fx_n = _lowpass_noise(
        rng,
        n_steps=n_steps,
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=float(action_config["fx_cutoff_hz"]),
        amplitude=fx_amp_n,
    )

    event_probability = float(action_config["event_probability"])
    if rng.random() < event_probability:
        _add_steer_event(rng, steer_rad, sample_rate_hz, steer_amp_rad)
    if rng.random() < event_probability:
        _add_fx_event(rng, fx_n, sample_rate_hz, fx_amp_n)

    steer_limit = np.deg2rad(20.0)
    fx_limit = 4500.0
    steer_rad = np.clip(steer_rad, -steer_limit, steer_limit)
    fx_n = np.clip(fx_n, -fx_limit, fx_limit)
    actions = np.stack([steer_rad, fx_n], axis=-1).astype(np.float64)
    return _fade_in(actions, sample_rate_hz=sample_rate_hz)
