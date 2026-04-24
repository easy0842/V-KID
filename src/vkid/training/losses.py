"""Training losses for VKID baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def triplet_consistency_loss(
    anchor_mu: torch.Tensor,
    positive_mu: torch.Tensor,
    negative_mu: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive_distance = torch.linalg.norm(anchor_mu - positive_mu, dim=-1)
    negative_distance = torch.linalg.norm(anchor_mu - negative_mu, dim=-1)
    return F.relu(positive_distance - negative_distance + margin).mean()


def dynamics_mse(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_delta, target_delta)
