"""Versioned SAR adaptation of upstream BIT base_transformer_pos_s4_dd8.

Not a reproduction of the original RGB training recipe. See docs/bit_sar_v2.md.
"""
from __future__ import annotations

import torch
from torch import nn

from smallflood_cd.models.baselines.bit_upstream.networks import BASE_Transformer, init_weights
from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput

UPSTREAM_COMMIT = "adcd7aea6f234586ffffdd4e9959404f96271711"


class BITSARV2(nn.Module):
    """Fixed dd8 architecture, SAR [VH,VV], fresh normal initialization, no downloads."""

    def __init__(self, input_channels: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("bit_sar_v2 requires two SAR channels [VH, VV]")
        if pretrained:
            raise ValueError("bit_sar_v2 v2 specifies fresh initialization, not pretrained weights")
        self.core = BASE_Transformer(
            input_nc=2, output_nc=2, token_len=4, resnet_stages_num=4,
            with_pos="learned", enc_depth=1, dec_depth=8,
            dim_head=64, decoder_dim_head=64,
        )
        # input_nc in upstream does not replace its hard-coded RGB stem.
        self.core.resnet.conv1 = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        # dd8 never executes these classification-backbone modules. Do not count dead weights.
        self.core.resnet.layer4 = nn.Identity()
        self.core.resnet.avgpool = nn.Identity()
        self.core.resnet.fc = nn.Identity()
        init_weights(self.core, init_type="normal", init_gain=0.02)
        # All BN buffers are fresh; do not inherit pretrained running statistics.
        for module in self.core.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

    def forward_two_class(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        if pre.ndim != 4 or pre.shape != post.shape or pre.shape[1] != 2:
            raise ValueError("Expected matching N×2×H×W pre/post tensors")
        if min(pre.shape[-2:]) < 32 or any(n % 8 for n in pre.shape[-2:]):
            raise ValueError("Spatial dimensions must be >=32 and divisible by 8")
        # Upstream stores intermediate token tensors for visualization. Do not retain graphs.
        try:
            return self.core(pre, post)
        finally:
            for name in ("tokens", "tokens_"):
                if hasattr(self.core, name):
                    delattr(self.core, name)

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor, coherence: torch.Tensor | None = None
    ) -> ChangeDetectionOutput:
        if coherence is not None:
            raise ValueError("bit_sar_v2 is intensity-only; coherence is unsupported")
        classes = self.forward_two_class(pre, post)
        # sigmoid(z1-z0) == softmax([z0,z1])[1]; class 1 means change.
        logits = classes[:, 1:2] - classes[:, 0:1]
        return ChangeDetectionOutput(logits, None, logits)

    def parameter_count(self) -> int:
        """All retained trainable parameters; no unused layer4/fc included."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
