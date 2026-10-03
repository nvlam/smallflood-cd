from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput
from smallflood_cd.losses.size_aware_tversky import SizeAwareTverskyLoss


def _normalized_masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


@dataclass
class LossOutput:
    total: torch.Tensor
    weighted_bce: torch.Tensor
    weighted_tversky: torch.Tensor
    boundary: torch.Tensor


class SmallFloodLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.4,
        tversky_weight: float = 0.4,
        boundary_weight: float = 0.2,
        false_positive_weight: float = 0.3,
        false_negative_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.fp_weight = false_positive_weight
        self.fn_weight = false_negative_weight
        self.size_aware_tversky = SizeAwareTverskyLoss(
            false_positive_weight, false_negative_weight
        )

    def forward(
        self,
        output: ChangeDetectionOutput,
        target: torch.Tensor,
        component_weights: torch.Tensor,
        valid_mask: torch.Tensor,
        boundary_target: torch.Tensor,
        boundary_valid_mask: torch.Tensor,
    ) -> LossOutput:
        target = target.float()
        valid_mask = valid_mask.float()
        positive_weights = torch.where(target > 0.5, component_weights, 1.0)
        pixel_weights = valid_mask * positive_weights
        bce_values = F.binary_cross_entropy_with_logits(
            output.change_logits, target, reduction="none"
        )
        weighted_bce = _normalized_masked_mean(bce_values, pixel_weights)

        weighted_tversky = self.size_aware_tversky(
            output.change_logits, target, component_weights, valid_mask
        )

        boundary_loss = output.change_logits.new_zeros(())
        if output.boundary_logits is not None:
            boundary_bce_values = F.binary_cross_entropy_with_logits(
                output.boundary_logits, boundary_target.float(), reduction="none"
            )
            boundary_bce = _normalized_masked_mean(
                boundary_bce_values, boundary_valid_mask.float()
            )
            boundary_probability = torch.sigmoid(output.boundary_logits)
            intersection = (
                boundary_valid_mask * boundary_probability * boundary_target
            ).sum()
            denominator = (
                boundary_valid_mask * boundary_probability
            ).sum() + (boundary_valid_mask * boundary_target).sum()
            boundary_dice = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
            boundary_loss = 0.5 * (boundary_bce + boundary_dice)

        total = (
            self.bce_weight * weighted_bce
            + self.tversky_weight * weighted_tversky
            + self.boundary_weight * boundary_loss
        )
        return LossOutput(total, weighted_bce, weighted_tversky, boundary_loss)
