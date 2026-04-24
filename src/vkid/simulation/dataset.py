"""Dataset generation and persistence for simulated VKID data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vkid.simulation.actions import generate_actions
from vkid.simulation.dynamics import SimulationResult, simulate_sequence
from vkid.simulation.parameters import PARAMETER_NAMES, conditions_to_array, sample_conditions


def _empty_dataset(n_conditions: int, n_sequences: int, n_steps: int) -> dict[str, np.ndarray]:
    return {
        "states": np.zeros((n_conditions, n_sequences, n_steps, 3), dtype=np.float64),
        "actions": np.zeros((n_conditions, n_sequences, n_steps, 2), dtype=np.float64),
        "inputs": np.zeros((n_conditions, n_sequences, n_steps, 5), dtype=np.float64),
        "poses": np.zeros((n_conditions, n_sequences, n_steps, 3), dtype=np.float64),
        "alpha_f": np.zeros((n_conditions, n_sequences, n_steps), dtype=np.float64),
        "alpha_r": np.zeros((n_conditions, n_sequences, n_steps), dtype=np.float64),
        "fyf": np.zeros((n_conditions, n_sequences, n_steps), dtype=np.float64),
        "fyr": np.zeros((n_conditions, n_sequences, n_steps), dtype=np.float64),
    }


def _write_sequence(target: dict[str, np.ndarray], condition_idx: int, sequence_idx: int, result: SimulationResult) -> None:
    target["states"][condition_idx, sequence_idx] = result.state
    target["actions"][condition_idx, sequence_idx] = result.action
    target["inputs"][condition_idx, sequence_idx] = np.concatenate([result.state, result.action], axis=-1)
    target["poses"][condition_idx, sequence_idx] = result.pose
    target["alpha_f"][condition_idx, sequence_idx] = result.diagnostics.alpha_f_rad
    target["alpha_r"][condition_idx, sequence_idx] = result.diagnostics.alpha_r_rad
    target["fyf"][condition_idx, sequence_idx] = result.diagnostics.fyf_n
    target["fyr"][condition_idx, sequence_idx] = result.diagnostics.fyr_n


def _is_plausible_sequence(result: SimulationResult, config: dict) -> bool:
    sanity = config.get("sanity", {})
    min_vx_mps = float(sanity.get("min_vx_kmh", 0.0)) / 3.6
    max_abs_vy_mps = float(sanity.get("max_abs_vy_kmh", 1e9)) / 3.6
    max_abs_yaw_rate = np.deg2rad(float(sanity.get("max_abs_yaw_rate_deg_s", 1e9)))

    state = result.state
    return bool(
        np.isfinite(state).all()
        and np.min(state[:, 0]) >= min_vx_mps
        and np.max(np.abs(state[:, 1])) <= max_abs_vy_mps
        and np.max(np.abs(state[:, 2])) <= max_abs_yaw_rate
    )


def _simulate_plausible_sequence(
    config: dict,
    rng: np.random.Generator,
    condition,
    n_steps: int,
    sample_rate_hz: float,
    vx0_range_kmh: list[float],
) -> SimulationResult:
    max_attempts = int(config.get("sanity", {}).get("max_attempts_per_sequence", 1))
    last_result: SimulationResult | None = None
    for _ in range(max_attempts):
        actions = generate_actions(config, rng, n_steps=n_steps, sample_rate_hz=sample_rate_hz)
        vx0_mps = rng.uniform(*vx0_range_kmh) / 3.6
        result = simulate_sequence(condition, actions, sample_rate_hz=sample_rate_hz, vx0_mps=vx0_mps)
        if _is_plausible_sequence(result, config):
            return result
        last_result = result
    assert last_result is not None
    return last_result


def generate_dataset(config: dict) -> dict[str, np.ndarray]:
    """Generate the full simulation dataset described by a config dictionary."""

    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    n_conditions = int(config["dataset"]["conditions"])
    n_sequences = int(config["dataset"]["sequences_per_condition"])
    sequence_length_s = float(config["dataset"]["sequence_length_s"])
    sample_rate_hz = float(config["dataset"]["sample_rate_hz"])
    n_steps = int(round(sequence_length_s * sample_rate_hz))

    conditions = sample_conditions(config, seed=seed)
    arrays = _empty_dataset(n_conditions, n_sequences, n_steps)
    vx0_range_kmh = config["vehicle"]["vx0_kmh_range"]

    for condition_idx, condition in enumerate(conditions):
        for sequence_idx in range(n_sequences):
            result = _simulate_plausible_sequence(
                config,
                rng,
                condition,
                n_steps=n_steps,
                sample_rate_hz=sample_rate_hz,
                vx0_range_kmh=vx0_range_kmh,
            )
            _write_sequence(arrays, condition_idx, sequence_idx, result)

    states = arrays["states"]
    train_fraction = float(config["dataset"]["train_condition_fraction"])
    n_train = int(round(n_conditions * train_fraction))
    condition_ids = np.arange(n_conditions, dtype=np.int64)

    dataset = {
        **arrays,
        "time": np.arange(n_steps, dtype=np.float64) / sample_rate_hz,
        "target_next": states[:, :, 1:, :],
        "target_delta": states[:, :, 1:, :] - states[:, :, :-1, :],
        "condition_params": conditions_to_array(conditions),
        "condition_param_names": PARAMETER_NAMES,
        "condition_ids": condition_ids,
        "train_condition_ids": condition_ids[:n_train],
        "val_condition_ids": condition_ids[n_train:],
        "sample_rate_hz": np.array(sample_rate_hz, dtype=np.float64),
        "sequence_length_s": np.array(sequence_length_s, dtype=np.float64),
    }
    return dataset


def save_dataset(dataset: dict[str, np.ndarray], output_path: str | Path) -> None:
    """Persist a generated dataset as a compressed NumPy archive."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
