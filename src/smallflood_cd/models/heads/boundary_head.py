from __future__ import annotations

import torch
import torch.nn as nn

from smallflood_cd.models.blocks import DepthwiseSeparableConv


class BoundaryRefinementHead(nn.Module):
    def __init__(self, channels: int, residual_init: float = 0.1) -> None:
        super().__init__()
        self.features = DepthwiseSeparableConv(channels, channels)
        self.boundary = nn.Conv2d(channels, 1, 1)
        self.residual = nn.Conv2d(channels + 1, 1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, decoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(decoded)
        boundary_logits = self.boundary(features)
        boundary_probability = torch.sigmoid(boundary_logits)
        residual = self.residual(torch.cat([decoded, boundary_probability], dim=1))
        scale = self.residual_scale.clamp(0.0, 1.0)
        return boundary_logits, scale * residual

