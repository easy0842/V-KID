"""Sanity-check visualizations for generated simulation data."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vkid")

import matplotlib.pyplot as plt
import numpy as np


def plot_sanity(dataset: dict[str, np.ndarray], output_dir: str | Path, condition_idx: int = 0, sequence_idx: int = 0) -> Path:
    """Create a compact sanity plot for one simulated sequence."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    time = dataset["time"]
    state = dataset["states"][condition_idx, sequence_idx]
    action = dataset["actions"][condition_idx, sequence_idx]
    pose = dataset["poses"][condition_idx, sequence_idx]
    alpha_f = dataset["alpha_f"][condition_idx, sequence_idx]
    alpha_r = dataset["alpha_r"][condition_idx, sequence_idx]
    fyf = dataset["fyf"][condition_idx, sequence_idx]
    fyr = dataset["fyr"][condition_idx, sequence_idx]
    params = dataset["condition_params"][condition_idx]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        "VKID simulator sanity check "
        f"(condition={condition_idx}, sequence={sequence_idx}, "
        f"B={params[0]:.2f}, C={params[1]:.2f}, D={params[2]:.2f}, m={params[4]:.0f} kg)"
    )

    axes[0, 0].plot(time, state[:, 0] * 3.6, label="Vx")
    axes[0, 0].plot(time, state[:, 1] * 3.6, label="Vy")
    axes[0, 0].set_ylabel("velocity [km/h]")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(time, np.rad2deg(state[:, 2]), label="yaw rate")
    axes[0, 1].set_ylabel("yaw rate [deg/s]")
    axes[0, 1].grid(True)

    axes[1, 0].plot(time, np.rad2deg(action[:, 0]), label="steer")
    axes[1, 0].set_ylabel("steer [deg]")
    axes[1, 0].grid(True)

    axes[1, 1].plot(time, action[:, 1], label="Fx")
    axes[1, 1].set_ylabel("longitudinal force [N]")
    axes[1, 1].grid(True)

    axes[2, 0].plot(time, np.rad2deg(alpha_f), label="front")
    axes[2, 0].plot(time, np.rad2deg(alpha_r), label="rear")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("slip angle [deg]")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(pose[:, 0], pose[:, 1], label="trajectory")
    axes[2, 1].set_xlabel("X [m]")
    axes[2, 1].set_ylabel("Y [m]")
    axes[2, 1].axis("equal")
    axes[2, 1].grid(True)

    force_ax = axes[2, 1].inset_axes([0.55, 0.55, 0.42, 0.4])
    force_ax.plot(time, fyf / 1000.0, label="Fyf")
    force_ax.plot(time, fyr / 1000.0, label="Fyr")
    force_ax.set_ylabel("Fy [kN]")
    force_ax.set_xlabel("t [s]")
    force_ax.grid(True)

    path = output_path / f"sanity_condition_{condition_idx:02d}_sequence_{sequence_idx:02d}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
