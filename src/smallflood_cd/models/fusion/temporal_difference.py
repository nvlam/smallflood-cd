from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smallflood_cd.models.blocks import ConvNormAct


class TemporalDifferenceFusion(nn.Module):
    def __init__(
        self,
        encoder_channels: tuple[int, ...],
        projection_channels: tuple[int, ...],
        coherence_channels: int = 0,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != len(projection_channels):
            raise ValueError("Encoder and projection channel lists must have equal length")
        self.coherence_channels = coherence_channels
        self.projections = nn.ModuleList(
            ConvNormAct(2 * source + coherence_channels, target)
            for source, target in zip(encoder_channels, projection_channels, strict=True)
        )

    def forward(
        self,
        pre_features: tuple[torch.Tensor, ...],
        post_features: tuple[torch.Tensor, ...],
        coherence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.coherence_channels and coherence is None:
            raise ValueError("Coherence input is required by this model configuration")
        results = []
        for pre, post, projection in zip(
            pre_features, post_features, self.projections, strict=True
        ):
            terms = [torch.abs(post - pre), post * pre]
            if coherence is not None:
                terms.append(F.interpolate(coherence, size=pre.shape[-2:], mode="bilinear"))
            results.append(projection(torch.cat(terms, dim=1)))
        return tuple(results)

