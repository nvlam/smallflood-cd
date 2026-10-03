import torch

from smallflood_cd.data.boundary_supervision import build_boundary_supervision


def test_boundary_target_has_hollow_interior() -> None:
    target = torch.zeros(1, 1, 9, 9)
    target[:, :, 2:7, 2:7] = 1
    supervision = build_boundary_supervision(target, torch.ones_like(target))
    assert supervision.target[0, 0, 4, 4] == 0
    assert supervision.target[0, 0, 2, 4] == 1


def test_uncertain_boundary_pixels_are_not_supervised() -> None:
    target = torch.zeros(1, 1, 9, 9)
    target[:, :, 2:7, 2:7] = 1
    uncertain = torch.zeros_like(target)
    uncertain[:, :, 2, 4] = 1
    supervision = build_boundary_supervision(
        target, torch.ones_like(target), uncertain, uncertainty_dilation=1
    )
    assert supervision.valid_mask[0, 0, 2, 4] == 0
    assert supervision.valid_mask[0, 0, 2, 3] == 0
    assert supervision.valid_mask[0, 0, 6, 4] == 1

