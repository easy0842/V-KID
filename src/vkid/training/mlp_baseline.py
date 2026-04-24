"""Training loop for the Transformer VAE + MLP decoder baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vkid.data.sampler import CrossSequenceBatchSampler, VkidSimulationDataset
from vkid.models.vae_mlp import VaeMlpDynamicsModel
from vkid.training.losses import dynamics_mse, kl_divergence_standard_normal, triplet_consistency_loss


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = config.get("train", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _to_tensor_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    tensor_batch: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.dtype == bool:
            tensor_batch[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
        elif np.issubdtype(value.dtype, np.integer):
            tensor_batch[key] = torch.as_tensor(value, dtype=torch.long, device=device)
        else:
            tensor_batch[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return tensor_batch


def build_model(config: dict[str, Any]) -> VaeMlpDynamicsModel:
    model_cfg = config["model"]
    return VaeMlpDynamicsModel(
        input_dim=int(model_cfg["input_dim"]),
        target_dim=int(model_cfg["state_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        d_model=int(model_cfg["d_model"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        transformer_heads=int(model_cfg["transformer_heads"]),
        transformer_ff_dim=int(model_cfg["transformer_ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        mlp_hidden_dim=int(model_cfg["mlp_hidden_dim"]),
        mlp_layers=int(model_cfg["mlp_layers"]),
    )


def compute_loss(
    model: VaeMlpDynamicsModel,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["anchor_context"], batch["anchor_mask"], batch["query_input"])
    pred_loss = dynamics_mse(output["pred_delta"], batch["query_target"])
    kl_loss = kl_divergence_standard_normal(output["mu"], output["logvar"])

    positive_mu, _ = model.encode(batch["positive_context"], batch["positive_mask"])
    negative_mu, _ = model.encode(batch["negative_context"], batch["negative_mask"])
    triplet_loss = triplet_consistency_loss(
        output["mu"],
        positive_mu,
        negative_mu,
        margin=float(config["loss"]["triplet_margin"]),
    )

    beta_kl = float(config["loss"]["beta_kl"])
    lambda_consistency = float(config["loss"]["lambda_consistency"])
    total = pred_loss + beta_kl * kl_loss + lambda_consistency * triplet_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "mse": float(pred_loss.detach().cpu()),
        "rmse": float(torch.sqrt(pred_loss.detach()).cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "triplet": float(triplet_loss.detach().cpu()),
        "sigma_mean": float(torch.exp(0.5 * output["logvar"].detach()).mean().cpu()),
        "mu_abs_mean": float(output["mu"].detach().abs().mean().cpu()),
    }
    return total, metrics


@torch.no_grad()
def evaluate(
    model: VaeMlpDynamicsModel,
    sampler: CrossSequenceBatchSampler,
    config: dict[str, Any],
    device: torch.device,
    split: str,
    steps: int,
) -> dict[str, float]:
    model.eval()
    collected: list[dict[str, float]] = []
    for _ in range(steps):
        batch = _to_tensor_batch(sampler.sample_batch(int(config["train"]["batch_size"]), split=split), device)
        _, metrics = compute_loss(model, batch, config)
        collected.append(metrics)
    return {key: float(np.mean([item[key] for item in collected])) for key in collected[0]}


def train_mlp_baseline(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _device_from_config(config)
    dataset = VkidSimulationDataset(config["data"]["dataset_path"])
    sampler = CrossSequenceBatchSampler(
        dataset,
        context_min=int(config["data"]["train_context_min"]),
        context_max=int(config["data"]["train_context_max"]),
        queries_per_context=int(config["data"]["queries_per_context"]),
        seed=seed,
        normalize=bool(config["data"].get("normalize", True)),
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    checkpoint_dir = Path(config["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(config["train"]["epochs"])
    steps_per_epoch = int(config["train"].get("steps_per_epoch", 100))
    val_steps = int(config["train"].get("val_steps", 10))
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics: list[dict[str, float]] = []
        for _ in range(steps_per_epoch):
            batch = _to_tensor_batch(sampler.sample_batch(int(config["train"]["batch_size"]), split="train"), device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, batch, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["train"].get("grad_clip_norm", 1.0)))
            optimizer.step()
            train_metrics.append(metrics)

        train_summary = {f"train_{key}": float(np.mean([item[key] for item in train_metrics])) for key in train_metrics[0]}
        val_summary = {f"val_{key}": value for key, value in evaluate(model, sampler, config, device, "val", val_steps).items()}
        row = {"epoch": float(epoch), **train_summary, **val_summary}
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_rmse={row['train_rmse']:.5f} val_rmse={row['val_rmse']:.5f} "
            f"train_kl={row['train_kl']:.4f} train_triplet={row['train_triplet']:.4f}"
        )

        if row["val_mse"] < best_val:
            best_val = row["val_mse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_mse": best_val,
                },
                checkpoint_dir / "best.pt",
            )

    torch.save({"model_state": model.state_dict(), "config": config, "history": history}, checkpoint_dir / "last.pt")
    return {"history": history, "best_val_mse": best_val, "checkpoint_dir": str(checkpoint_dir)}
