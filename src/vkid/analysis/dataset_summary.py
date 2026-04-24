"""Summaries and plots for generated VKID simulation datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vkid")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load a VKID ``.npz`` dataset into an in-memory dictionary."""

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_condition_table(dataset: dict[str, np.ndarray]) -> pd.DataFrame:
    """Create one row per dynamics condition."""

    names = [str(name) for name in dataset["condition_param_names"]]
    params = dataset["condition_params"]
    train_ids = set(int(item) for item in dataset["train_condition_ids"])
    rows: list[dict[str, Any]] = []
    for condition_id, values in enumerate(params):
        row: dict[str, Any] = {"condition_id": condition_id, "split": "train" if condition_id in train_ids else "val"}
        row.update({name: float(value) for name, value in zip(names, values)})
        rows.append(row)
    return pd.DataFrame(rows)


def make_sequence_table(dataset: dict[str, np.ndarray]) -> pd.DataFrame:
    """Create one row per condition/sequence rollout."""

    states = dataset["states"]
    actions = dataset["actions"]
    poses = dataset["poses"]
    rows: list[dict[str, Any]] = []
    for condition_id in range(states.shape[0]):
        for sequence_id in range(states.shape[1]):
            state = states[condition_id, sequence_id]
            action = actions[condition_id, sequence_id]
            pose = poses[condition_id, sequence_id]
            displacement_m = float(np.linalg.norm(pose[-1, :2] - pose[0, :2]))
            path_length_m = float(np.sum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)))
            rows.append(
                {
                    "condition_id": condition_id,
                    "sequence_id": sequence_id,
                    "mean_vx_kmh": float(np.mean(state[:, 0]) * 3.6),
                    "min_vx_kmh": float(np.min(state[:, 0]) * 3.6),
                    "max_vx_kmh": float(np.max(state[:, 0]) * 3.6),
                    "max_abs_vy_kmh": float(np.max(np.abs(state[:, 1])) * 3.6),
                    "max_abs_yaw_rate_deg_s": float(np.rad2deg(np.max(np.abs(state[:, 2])))),
                    "max_abs_steer_deg": float(np.rad2deg(np.max(np.abs(action[:, 0])))),
                    "max_abs_fx_n": float(np.max(np.abs(action[:, 1]))),
                    "final_x_m": float(pose[-1, 0]),
                    "final_y_m": float(pose[-1, 1]),
                    "displacement_m": displacement_m,
                    "path_length_m": path_length_m,
                }
            )
    return pd.DataFrame(rows)


def make_range_summary(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute global shape and range statistics."""

    states = dataset["states"]
    actions = dataset["actions"]
    target_delta = dataset["target_delta"]
    return {
        "conditions": int(states.shape[0]),
        "sequences_per_condition": int(states.shape[1]),
        "steps_per_sequence": int(states.shape[2]),
        "total_sequences": int(states.shape[0] * states.shape[1]),
        "total_timesteps": int(np.prod(states.shape[:3])),
        "total_transitions": int(np.prod(target_delta.shape[:3])),
        "train_conditions": int(len(dataset["train_condition_ids"])),
        "val_conditions": int(len(dataset["val_condition_ids"])),
        "state_dim": int(states.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "sample_rate_hz": float(dataset["sample_rate_hz"]),
        "sequence_length_s": float(dataset["sequence_length_s"]),
        "nan_states": int(np.isnan(states).sum()),
        "nan_actions": int(np.isnan(actions).sum()),
        "ranges": {
            "vx_kmh": [float(np.min(states[..., 0]) * 3.6), float(np.max(states[..., 0]) * 3.6)],
            "vy_kmh": [float(np.min(states[..., 1]) * 3.6), float(np.max(states[..., 1]) * 3.6)],
            "yaw_rate_deg_s": [
                float(np.rad2deg(np.min(states[..., 2]))),
                float(np.rad2deg(np.max(states[..., 2]))),
            ],
            "steer_deg": [
                float(np.rad2deg(np.min(actions[..., 0]))),
                float(np.rad2deg(np.max(actions[..., 0]))),
            ],
            "fx_n": [float(np.min(actions[..., 1])), float(np.max(actions[..., 1]))],
        },
    }


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_trajectory_grid(dataset: dict[str, np.ndarray], output_path: str | Path) -> None:
    """Plot each condition in its own subplot with all sequences overlaid."""

    poses = dataset["poses"]
    params = dataset["condition_params"]
    n_conditions = poses.shape[0]
    n_cols = 4
    n_rows = int(np.ceil(n_conditions / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for condition_id, axis in enumerate(axes_flat):
        if condition_id >= n_conditions:
            axis.axis("off")
            continue
        for sequence_id in range(poses.shape[1]):
            pose = poses[condition_id, sequence_id]
            axis.plot(pose[:, 0], pose[:, 1], lw=1.3, alpha=0.75)
        axis.set_title(
            f"C{condition_id}: D={params[condition_id, 2]:.2f}, "
            f"m={params[condition_id, 4]:.0f}, lf={params[condition_id, 6]:.2f}"
        )
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.axis("equal")
        axis.grid(True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_action_state_overview(dataset: dict[str, np.ndarray], output_path: str | Path) -> None:
    """Plot compact dataset-level histograms for state and action ranges."""

    states = dataset["states"]
    actions = dataset["actions"]
    features = [
        ("Vx [km/h]", states[..., 0].ravel() * 3.6),
        ("Vy [km/h]", states[..., 1].ravel() * 3.6),
        ("yaw rate [deg/s]", np.rad2deg(states[..., 2].ravel())),
        ("steer [deg]", np.rad2deg(actions[..., 0].ravel())),
        ("Fx [N]", actions[..., 1].ravel()),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, (label, values) in zip(axes.ravel(), features):
        axis.hist(values, bins=50, color="#4C78A8", alpha=0.85)
        axis.set_title(label)
        axis.grid(True, alpha=0.3)
    axes.ravel()[-1].axis("off")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize_dataset(dataset_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write dataset summary artifacts and return their paths."""

    dataset = load_dataset(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    condition_table_path = output / "condition_summary.csv"
    sequence_table_path = output / "sequence_summary.csv"
    range_summary_path = output / "range_summary.json"
    trajectory_grid_path = output / "trajectory_grid.png"
    feature_hist_path = output / "feature_histograms.png"

    make_condition_table(dataset).to_csv(condition_table_path, index=False)
    make_sequence_table(dataset).to_csv(sequence_table_path, index=False)
    write_json(make_range_summary(dataset), range_summary_path)
    plot_trajectory_grid(dataset, trajectory_grid_path)
    plot_action_state_overview(dataset, feature_hist_path)

    return {
        "condition_table": condition_table_path,
        "sequence_table": sequence_table_path,
        "range_summary": range_summary_path,
        "trajectory_grid": trajectory_grid_path,
        "feature_histograms": feature_hist_path,
    }
