from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from smallflood_cd.models.decoders.lightweight_decoder import LightweightDecoder
from smallflood_cd.models.encoders.mobilenetv3_siamese import MobileNetV3SiameseEncoder
from smallflood_cd.models.fusion.temporal_difference import TemporalDifferenceFusion
from smallflood_cd.models.heads.boundary_head import BoundaryRefinementHead
from smallflood_cd.models.heads.change_head import ChangeHead


@dataclass
class ChangeDetectionOutput:
    change_logits: torch.Tensor
    boundary_logits: torch.Tensor | None
    preliminary_logits: torch.Tensor


class SmallFloodCDNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        projection_channels: tuple[int, int, int, int] = (24, 32, 64, 96),
        pretrained: bool = False,
        coherence_channels: int = 0,
        use_boundary_head: bool = True,
        boundary_residual_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MobileNetV3SiameseEncoder(input_channels, pretrained)
        self.fusion = TemporalDifferenceFusion(
            self.encoder.output_channels, projection_channels, coherence_channels
        )
        self.decoder = LightweightDecoder(projection_channels)
        self.change_head = ChangeHead(self.decoder.out_channels)
        self.boundary_head = (
            BoundaryRefinementHead(self.decoder.out_channels, boundary_residual_init)
            if use_boundary_head
            else None
        )

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor, coherence: torch.Tensor | None = None
    ) -> ChangeDetectionOutput:
        pre_features, post_features = self.encoder(pre, post)
        fused = self.fusion(pre_features, post_features, coherence)
        decoded = self.decoder(fused)
        preliminary = self.change_head(decoded)
        boundary_logits = None
        refined = preliminary
        if self.boundary_head is not None:
            boundary_logits, residual = self.boundary_head(decoded)
            refined = preliminary + residual
        target_size = pre.shape[-2:]
        return ChangeDetectionOutput(
            change_logits=F.interpolate(refined, target_size, mode="bilinear", align_corners=False),
            boundary_logits=(
                F.interpolate(boundary_logits, target_size, mode="bilinear", align_corners=False)
                if boundary_logits is not None
                else None
            ),
            preliminary_logits=F.interpolate(
                preliminary, target_size, mode="bilinear", align_corners=False
            ),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def assert_parameter_budget(self, maximum: int = 5_000_000) -> None:
        count = self.parameter_count()
        if count > maximum:
            raise RuntimeError(f"Parameter budget exceeded: {count:,} > {maximum:,}")

