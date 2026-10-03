from __future__ import annotations

import torch
import torch.nn.functional as F


def morphological_boundary(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Create a differentiability-independent binary morphological gradient."""
    if radius < 1:
        raise ValueError("Boundary radius must be at least one")
    kernel = 2 * radius + 1
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=radius)
    return (dilated - eroded > 0).to(mask.dtype)

