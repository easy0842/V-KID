"""Progressive identification evaluation for trained VKID encoders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vkid")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vkid.data.sampler import VkidSimulationDataset
from vkid.training.mlp_baseline import build_model


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = config.get("eval", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _normalized_prefix(dataset: VkidSimulationDataset, condition_id: int, sequence_id: int, length: int) -> np.ndarray:
    values = dataset.inputs[condition_id, sequence_id, :length]
    return ((values - dataset.input_mean) / dataset.input_std).astype(np.float32)


@torch.no_grad()
def _encode_contexts(
    model,
    contexts: list[np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(context.shape[0] for context in contexts)
    input_dim = contexts[0].shape[-1]
    padded = np.zeros((len(contexts), max_len, input_dim), dtype=np.float32)
    mask = np.zeros((len(contexts), max_len), dtype=bool)
    for row, context in enumerate(contexts):
        padded[row, : context.shape[0]] = context
        mask[row, : context.shape[0]] = True

    context_tensor = torch.as_tensor(padded, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    mu, logvar = model.encode(context_tensor, mask_tensor)
    sigma = torch.exp(0.5 * logvar)
    return mu.cpu().numpy(), sigma.cpu().numpy()


def _build_reference_database(
    model,
    dataset: VkidSimulationDataset,
    condition_ids: np.ndarray,
    reference_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    labels: list[int] = []
    for condition_id in condition_ids:
        for sequence_id in range(dataset.n_sequences):
            contexts.append(_normalized_prefix(dataset, int(condition_id), sequence_id, reference_length))
            labels.append(int(condition_id))

    mu, _ = _encode_contexts(model, contexts, device)
    references = []
    for condition_id in condition_ids:
        matching = mu[np.array(labels) == int(condition_id)]
        references.append(matching.mean(axis=0))
    return np.asarray(condition_ids, dtype=np.int64), np.stack(references, axis=0)


def _rank_conditions(mu: np.ndarray, reference_mu: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(reference_mu - mu[None, :], axis=-1)
    return np.argsort(distances)


def _context_excitation(dataset: VkidSimulationDataset, condition_id: int, sequence_id: int, length: int) -> dict[str, float]:
    states = dataset.states[condition_id, sequence_id, :length]
    actions = dataset.actions[condition_id, sequence_id, :length]
    return {
        "steer_std_deg": float(np.rad2deg(actions[:, 0]).std()),
        "fx_std_n": float(actions[:, 1].std()),
        "yaw_rate_std_deg_s": float(np.rad2deg(states[:, 2]).std()),
        "vy_std_kmh": float((states[:, 1] * 3.6).std()),
        "max_abs_steer_deg": float(np.rad2deg(np.abs(actions[:, 0]).max())),
        "max_abs_yaw_rate_deg_s": float(np.rad2deg(np.abs(states[:, 2]).max())),
    }


def _write_plots(summary: pd.DataFrame, detail: pd.DataFrame, output_dir: Path, sample_rate_hz: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seconds = summary["context_length"] / sample_rate_hz

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(seconds, summary["sigma_mean"], marker="o", label="mean sigma")
    ax.fill_between(seconds, summary["sigma_p25"], summary["sigma_p75"], alpha=0.2, label="25-75%")
    ax.set_xlabel("context time [s]")
    ax.set_ylabel("sigma")
    ax.set_title("Progressive uncertainty")
    ax.grid(True)
    ax.legend()
    fig.savefig(output_dir / "sigma_vs_context.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(seconds, summary["top1_accuracy"], marker="o", label="top-1")
    ax.plot(seconds, summary["top3_accuracy"], marker="o", label="top-3")
    ax.set_xlabel("context time [s]")
    ax.set_ylabel("matching accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Progressive matching accuracy")
    ax.grid(True)
    ax.legend()
    fig.savefig(output_dir / "matching_vs_context.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(seconds, summary["mu_to_reference_mean"], marker="o")
    ax.set_xlabel("context time [s]")
    ax.set_ylabel("distance to reference mu")
    ax.set_title("Latent convergence to condition reference")
    ax.grid(True)
    fig.savefig(output_dir / "mu_distance_vs_context.png", dpi=170)
    plt.close(fig)

    sigma_dim_columns = [column for column in summary.columns if column.startswith("sigma_dim_")]
    if sigma_dim_columns:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for column in sigma_dim_columns:
            ax.plot(seconds, summary[column], marker="o", label=column.replace("sigma_dim_", "z"))
        ax.set_xlabel("context time [s]")
        ax.set_ylabel("sigma")
        ax.set_title("Sigma by latent dimension")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(output_dir / "sigma_by_dimension.png", dpi=170)
        plt.close(fig)

    condition_summary = (
        detail.groupby(["condition_id", "context_length"])
        .agg(context_time_s=("context_time_s", "mean"), sigma_mean=("sigma_mean", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for condition_id, group in condition_summary.groupby("condition_id"):
        ax.plot(group["context_time_s"], group["sigma_mean"], marker="o", label=f"C{condition_id}")
    ax.set_xlabel("context time [s]")
    ax.set_ylabel("sigma")
    ax.set_title("Sigma by condition")
    ax.grid(True)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(output_dir / "sigma_by_condition.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, feature in zip(axes, ["steer_std_deg", "yaw_rate_std_deg_s", "fx_std_n"]):
        axis.scatter(detail[feature], detail["sigma_mean"], alpha=0.75)
        corr = detail[[feature, "sigma_mean"]].corr(method="spearman").iloc[0, 1]
        axis.set_title(f"{feature} vs sigma\nSpearman={corr:.2f}")
        axis.set_xlabel(feature)
        axis.set_ylabel("sigma")
        axis.grid(True)
    fig.savefig(output_dir / "sigma_vs_excitation.png", dpi=170)
    plt.close(fig)


def evaluate_progressive(config: dict[str, Any]) -> dict[str, Path]:
    """Evaluate sigma and matching behavior as context length grows."""

    dataset = VkidSimulationDataset(config["data"]["dataset_path"])
    checkpoint_path = Path(config["model"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    model = build_model(train_config)
    model.load_state_dict(checkpoint["model_state"])
    device = _device_from_config(config)
    model.to(device)
    model.eval()

    output_dir = Path(config["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    context_lengths = [int(item) for item in config["eval"]["context_lengths"]]
    split = config["eval"].get("split", "val")
    eval_condition_ids = dataset.condition_ids_for_split(split)
    reference_condition_ids = dataset.condition_ids_for_split(config["eval"].get("reference_split", "all"))
    reference_length = int(config["eval"].get("reference_length", max(context_lengths)))

    ref_ids, ref_mu = _build_reference_database(model, dataset, reference_condition_ids, reference_length, device)
    rows: list[dict[str, Any]] = []
    for context_length in context_lengths:
        contexts: list[np.ndarray] = []
        labels: list[tuple[int, int]] = []
        for condition_id in eval_condition_ids:
            for sequence_id in range(dataset.n_sequences):
                contexts.append(_normalized_prefix(dataset, int(condition_id), sequence_id, context_length))
                labels.append((int(condition_id), sequence_id))

        mu, sigma = _encode_contexts(model, contexts, device)
        for row_idx, (condition_id, sequence_id) in enumerate(labels):
            ranked_indices = _rank_conditions(mu[row_idx], ref_mu)
            ranked_condition_ids = ref_ids[ranked_indices]
            top1 = int(ranked_condition_ids[0])
            top3 = [int(item) for item in ranked_condition_ids[:3]]
            own_ref_idx = int(np.where(ref_ids == condition_id)[0][0])
            row = {
                "context_length": context_length,
                "context_time_s": context_length / dataset.sample_rate_hz,
                "condition_id": condition_id,
                "sequence_id": sequence_id,
                "sigma_mean": float(sigma[row_idx].mean()),
                "sigma_min": float(sigma[row_idx].min()),
                "sigma_max": float(sigma[row_idx].max()),
                "mu_norm": float(np.linalg.norm(mu[row_idx])),
                "mu_to_reference": float(np.linalg.norm(mu[row_idx] - ref_mu[own_ref_idx])),
                "top1_condition_id": top1,
                "top3_condition_ids": " ".join(str(item) for item in top3),
                "top1_correct": top1 == condition_id,
                "top3_correct": condition_id in top3,
                **_context_excitation(dataset, condition_id, sequence_id, context_length),
            }
            for dim, value in enumerate(sigma[row_idx]):
                row[f"sigma_dim_{dim}"] = float(value)
            for dim, value in enumerate(mu[row_idx]):
                row[f"mu_dim_{dim}"] = float(value)
            rows.append(
                row
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("context_length")
        .agg(
            context_time_s=("context_time_s", "mean"),
            sigma_mean=("sigma_mean", "mean"),
            sigma_p25=("sigma_mean", lambda x: float(np.quantile(x, 0.25))),
            sigma_p75=("sigma_mean", lambda x: float(np.quantile(x, 0.75))),
            mu_to_reference_mean=("mu_to_reference", "mean"),
            top1_accuracy=("top1_correct", "mean"),
            top3_accuracy=("top3_correct", "mean"),
        )
        .reset_index()
    )
    sigma_dim_columns = [column for column in detail.columns if column.startswith("sigma_dim_")]
    if sigma_dim_columns:
        sigma_dim_summary = detail.groupby("context_length")[sigma_dim_columns].mean().reset_index()
        summary = summary.merge(sigma_dim_summary, on="context_length", how="left")

    detail_path = output_dir / "progressive_detail.csv"
    summary_path = output_dir / "progressive_summary.csv"
    reference_path = output_dir / "reference_latents.npz"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    np.savez_compressed(reference_path, condition_ids=ref_ids, reference_mu=ref_mu)
    correlation_path = output_dir / "sigma_excitation_correlations.csv"
    excitation_columns = [
        "steer_std_deg",
        "fx_std_n",
        "yaw_rate_std_deg_s",
        "vy_std_kmh",
        "max_abs_steer_deg",
        "max_abs_yaw_rate_deg_s",
    ]
    correlations = [
        {
            "feature": feature,
            "spearman_with_sigma_mean": float(detail[[feature, "sigma_mean"]].corr(method="spearman").iloc[0, 1]),
        }
        for feature in excitation_columns
    ]
    pd.DataFrame(correlations).to_csv(correlation_path, index=False)
    _write_plots(summary, detail, output_dir, dataset.sample_rate_hz)

    return {
        "detail": detail_path,
        "summary": summary_path,
        "reference_latents": reference_path,
        "sigma_plot": output_dir / "sigma_vs_context.png",
        "matching_plot": output_dir / "matching_vs_context.png",
        "mu_distance_plot": output_dir / "mu_distance_vs_context.png",
        "sigma_dimension_plot": output_dir / "sigma_by_dimension.png",
        "sigma_condition_plot": output_dir / "sigma_by_condition.png",
        "sigma_excitation_plot": output_dir / "sigma_vs_excitation.png",
        "sigma_excitation_correlations": correlation_path,
    }
