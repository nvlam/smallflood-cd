from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput


class SemanticTokenizer(nn.Module):
    def __init__(self, channels: int, token_count: int) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, token_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        values = features.flatten(2).transpose(1, 2)
        weights = self.attention(features).flatten(2).softmax(dim=-1)
        return torch.bmm(weights, values).reshape(batch, -1, channels)


class BITBaseline(nn.Module):
    """Compact implementation of the Bitemporal Image Transformer design."""

    def __init__(
        self,
        input_channels: int = 2,
        embedding_dim: int = 64,
        token_count: int = 4,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        if input_channels != 3:
            original = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                input_channels,
                original.out_channels,
                original.kernel_size,
                original.stride,
                original.padding,
                bias=False,
            )
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.project = nn.Conv2d(512, embedding_dim, 1)
        self.tokenizer = SemanticTokenizer(embedding_dim, token_count)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, transformer_layers)
        self.pixel_to_token = nn.MultiheadAttention(
            embedding_dim, attention_heads, batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, 1, 1),
        )
        self.token_count = token_count

    def _features(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.project(self.backbone(tensor))

    def _decode_tokens(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        pixels = features.flatten(2).transpose(1, 2)
        decoded, _ = self.pixel_to_token(pixels, tokens, tokens, need_weights=False)
        return decoded.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor, coherence: torch.Tensor | None = None
    ) -> ChangeDetectionOutput:
        del coherence
        pre_features, post_features = self._features(pre), self._features(post)
        pre_tokens = self.tokenizer(pre_features)
        post_tokens = self.tokenizer(post_features)
        encoded = self.temporal_encoder(torch.cat([pre_tokens, post_tokens], dim=1))
        pre_context = encoded[:, : self.token_count]
        post_context = encoded[:, self.token_count :]
        decoded_pre = self._decode_tokens(pre_features, pre_context)
        decoded_post = self._decode_tokens(post_features, post_context)
        logits = self.decoder(torch.abs(decoded_post - decoded_pre))
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return ChangeDetectionOutput(logits, None, logits)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
