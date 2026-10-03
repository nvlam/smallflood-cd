from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smallflood_cd.models.blocks import ConvNormAct, DepthwiseSeparableConv


class DecoderStage(nn.Module):
    def __init__(self, skip_channels: int, top_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(skip_channels + top_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, skip: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        top = F.interpolate(top, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, top], dim=1))


class LightweightDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.bottleneck = DepthwiseSeparableConv(c4, c4)
        self.stage3 = DecoderStage(c3, c4, c3)
        self.stage2 = DecoderStage(c2, c3, c2)
        self.stage1 = DecoderStage(c1, c2, c1)
        self.out_channels = c1

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        top = self.bottleneck(f4)
        top = self.stage3(f3, top)
        top = self.stage2(f2, top)
        return self.stage1(f1, top)

