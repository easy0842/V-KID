import torch

from vkid.models.vae_mlp import VaeMlpDynamicsModel
from vkid.training.losses import kl_divergence_standard_normal, triplet_consistency_loss


def test_vae_mlp_forward_shapes() -> None:
    model = VaeMlpDynamicsModel(
        input_dim=5,
        target_dim=3,
        latent_dim=4,
        d_model=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=32,
        dropout=0.0,
        mlp_hidden_dim=16,
        mlp_layers=2,
    )
    context = torch.randn(3, 7, 5)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[0, 5:] = False
    query = torch.randn(3, 11, 5)
    output = model(context, mask, query)

    assert output["pred_delta"].shape == (3, 11, 3)
    assert output["mu"].shape == (3, 4)
    assert output["logvar"].shape == (3, 4)


def test_losses_are_finite() -> None:
    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)
    kl = kl_divergence_standard_normal(mu, logvar)
    triplet = triplet_consistency_loss(mu, mu + 0.1, mu + 2.0, margin=1.0)
    assert torch.isfinite(kl)
    assert torch.isfinite(triplet)
