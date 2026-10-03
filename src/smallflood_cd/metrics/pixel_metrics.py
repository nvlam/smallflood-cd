from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float


def binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute valid-pixel precision, recall, F1, IoU, and accuracy."""
    prediction = torch.sigmoid(logits) >= threshold
    truth = target.bool()
    valid = valid_mask.bool()
    prediction, truth = prediction[valid], truth[valid]
    tp = torch.logical_and(prediction, truth).sum().item()
    fp = torch.logical_and(prediction, ~truth).sum().item()
    fn = torch.logical_and(~prediction, truth).sum().item()
    tn = torch.logical_and(~prediction, ~truth).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return BinaryMetrics(precision, recall, f1, iou, accuracy)
