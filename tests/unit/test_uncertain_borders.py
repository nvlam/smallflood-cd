import torch

from smallflood_cd.data.uncertain_borders import apply_uncertain_border_mask


def test_uncertain_mask_dilates_to_safety_band() -> None:
    valid = torch.ones(1, 1, 7, 7)
    uncertain = torch.zeros_like(valid)
    uncertain[:, :, 3, 3] = 1
    masked = apply_uncertain_border_mask(valid, uncertain, dilation_radius=1)
    assert masked.sum() == valid.numel() - 9
    assert torch.all(masked[:, :, 2:5, 2:5] == 0)


def test_zero_radius_excludes_only_declared_uncertainty() -> None:
    valid = torch.ones(1, 1, 5, 5)
    uncertain = torch.zeros_like(valid)
    uncertain[:, :, 1, 1] = 1
    masked = apply_uncertain_border_mask(valid, uncertain, dilation_radius=0)
    assert masked.sum() == valid.numel() - 1

