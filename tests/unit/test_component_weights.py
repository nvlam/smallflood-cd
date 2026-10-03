import torch

from smallflood_cd.data.component_weights import area_weight_lookup, component_weight_map


def test_smaller_components_receive_larger_weights() -> None:
    weights = area_weight_lookup(
        torch.tensor([1, 4, 16]), reference_area=4, alpha=1, gamma=0.5, weight_cap=4
    )
    assert weights[0] > weights[1] > weights[2]


def test_background_remains_unit_weight_and_cap_is_enforced() -> None:
    labels = torch.tensor([[[[0, 1, 0], [2, 2, 0], [0, 0, 0]]]])
    weights = component_weight_map(
        labels, reference_area=100, alpha=1, gamma=1, weight_cap=3
    )
    assert torch.all(weights[labels == 0] == 1)
    assert torch.all(weights[labels > 0] <= 4)  # 1 + alpha * cap

