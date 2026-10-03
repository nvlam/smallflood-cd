import torch

from smallflood_cd.data.boundary_targets import morphological_boundary
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput


def test_composite_loss_is_finite() -> None:
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 5:8, 5:8] = 1
    logits = torch.zeros_like(target, requires_grad=True)
    boundary = morphological_boundary(target)
    output = ChangeDetectionOutput(logits, torch.zeros_like(target), logits)
    result = SmallFloodLoss()(
        output,
        target,
        torch.ones_like(target),
        torch.ones_like(target),
        boundary,
        torch.ones_like(target),
    )
    assert torch.isfinite(result.total)
    result.total.backward()

