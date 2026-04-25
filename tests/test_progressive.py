import torch

from vkid.analysis.progressive import _rank_conditions


def test_rank_conditions_orders_by_euclidean_distance() -> None:
    references = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]).numpy()
    ranked = _rank_conditions(torch.tensor([1.9, 0.0]).numpy(), references)
    assert ranked.tolist() == [1, 0, 2]
