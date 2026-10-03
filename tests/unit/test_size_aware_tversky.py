import torch

from smallflood_cd.losses.size_aware_tversky import SizeAwareTverskyLoss


def test_weighting_a_missed_positive_increases_loss() -> None:
    logits = torch.tensor([[[[5.0, -5.0, -5.0]]]])
    target = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    valid = torch.ones_like(target)
    loss = SizeAwareTverskyLoss()
    uniform = loss(logits, target, torch.ones_like(target), valid)
    missed_weighted = loss(logits, target, torch.tensor([[[[1.0, 5.0, 1.0]]]]), valid)
    assert missed_weighted > uniform


def test_invalid_pixels_do_not_affect_loss() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 0.0]]]])
    weights = torch.ones_like(target)
    loss = SizeAwareTverskyLoss()
    first = loss(torch.tensor([[[[2.0, -20.0]]]]), target, weights, valid)
    second = loss(torch.tensor([[[[2.0, 20.0]]]]), target, weights, valid)
    assert torch.allclose(first, second)

