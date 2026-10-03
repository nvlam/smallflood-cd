from __future__ import annotations

import torch
import torch.nn as nn


class SizeAwareTverskyLoss(nn.Module):
    """Tversky loss with component-size weights applied to changed pixels."""

    def __init__(
        self,
        false_positive_weight: float = 0.3,
        false_negative_weight: float = 0.7,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if false_positive_weight < 0 or false_negative_weight < 0:
            raise ValueError("Tversky coefficients must be non-negative")
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        component_weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        target = target.float()
        valid = valid_mask.float()
        positive_weights = torch.where(target > 0.5, component_weights.float(), 1.0)
        true_positive = (valid * positive_weights * target * probability).sum()
        false_positive = (valid * (1.0 - target) * probability).sum()
        false_negative = (
            valid * positive_weights * target * (1.0 - probability)
        ).sum()
        index = (true_positive + self.epsilon) / (
            true_positive
            + self.false_positive_weight * false_positive
            + self.false_negative_weight * false_negative
            + self.epsilon
        )
        return 1.0 - index

