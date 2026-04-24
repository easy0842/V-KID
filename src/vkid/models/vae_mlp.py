"""Transformer VAE encoder with an MLP dynamics decoder."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.shape[1]]


class TransformerVaeEncoder(nn.Module):
    """Encode variable-length context sequences into ``mu`` and ``logvar``."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(context)
        hidden = self.position(hidden)
        padding_mask = ~mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        weights = mask.float().unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.mu(pooled), self.logvar(pooled).clamp(min=-10.0, max=5.0)


class MlpDynamicsDecoder(nn.Module):
    """Predict state delta from latent dynamics code and query input."""

    def __init__(self, latent_dim: int, input_dim: int, target_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim + input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, query_input: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(1).expand(-1, query_input.shape[1], -1)
        decoder_input = torch.cat([z_expanded, query_input], dim=-1)
        return self.net(decoder_input)


class VaeMlpDynamicsModel(nn.Module):
    """Full baseline model."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        latent_dim: int,
        d_model: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ff_dim: int,
        dropout: float,
        mlp_hidden_dim: int,
        mlp_layers: int,
    ):
        super().__init__()
        self.encoder = TransformerVaeEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
        )
        self.decoder = MlpDynamicsDecoder(
            latent_dim=latent_dim,
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dim=mlp_hidden_dim,
            n_layers=mlp_layers,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, context: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(context, mask)

    def forward(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
        query_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(context, mask)
        z = self.reparameterize(mu, logvar)
        pred_delta = self.decoder(z, query_input)
        return {"pred_delta": pred_delta, "mu": mu, "logvar": logvar, "z": z}
