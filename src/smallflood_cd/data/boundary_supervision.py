from __future__ import annotations

from dataclasses import dataclass

import torch

from smallflood_cd.data.boundary_targets import morphological_boundary
from smallflood_cd.data.uncertain_borders import apply_uncertain_border_mask


@dataclass(frozen=True)
class BoundarySupervision:
    target: torch.Tensor
    valid_mask: torch.Tensor


def build_boundary_supervision(
    change_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertain_mask: torch.Tensor | None = None,
    boundary_radius: int = 1,
    uncertainty_dilation: int = 1,
) -> BoundarySupervision:
    """Derive edge targets and exclude uncertain annotation neighborhoods."""
    boundary = morphological_boundary(change_mask, boundary_radius)
    boundary_valid = valid_mask.float()
    if uncertain_mask is not None:
        boundary_valid = apply_uncertain_border_mask(
            boundary_valid, uncertain_mask, uncertainty_dilation
        )
    return BoundarySupervision(boundary, boundary_valid)
