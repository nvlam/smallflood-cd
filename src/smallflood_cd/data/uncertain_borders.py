from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate_binary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("Dilation radius cannot be negative")
    mask = mask.float()
    if radius == 0:
        return (mask > 0).to(mask.dtype)
    kernel = 2 * radius + 1
    return (F.max_pool2d(mask, kernel, stride=1, padding=radius) > 0).to(mask.dtype)


def apply_uncertain_border_mask(
    valid_mask: torch.Tensor,
    uncertain_mask: torch.Tensor,
    dilation_radius: int = 1,
) -> torch.Tensor:
    """Remove uncertain pixels and their safety band from supervision/evaluation."""
    if valid_mask.shape != uncertain_mask.shape:
        raise ValueError("Valid and uncertain masks must have identical shapes")
    excluded = dilate_binary(uncertain_mask, dilation_radius)
    return valid_mask.float() * (1.0 - excluded)

