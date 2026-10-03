from __future__ import annotations

from dataclasses import dataclass

import torch

from smallflood_cd.data.boundary_targets import morphological_boundary
from smallflood_cd.data.uncertain_borders import dilate_binary


@dataclass(frozen=True)
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float
    iou: float


def boundary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    probability_threshold: float = 0.5,
    boundary_radius: int = 1,
    tolerance: int = 2,
) -> BoundaryMetrics:
    """Compute exact boundary IoU and tolerance-aware boundary precision/recall/F1."""
    prediction = (torch.sigmoid(logits) >= probability_threshold).float()
    truth = target.float()
    valid = valid_mask.float()
    predicted_boundary = morphological_boundary(prediction, boundary_radius) * valid
    target_boundary = morphological_boundary(truth, boundary_radius) * valid

    intersection = (predicted_boundary * target_boundary).sum().item()
    union = ((predicted_boundary + target_boundary) > 0).sum().item()
    if union == 0:
        boundary_iou = 1.0
    else:
        boundary_iou = intersection / union

    predicted_count = predicted_boundary.sum().item()
    target_count = target_boundary.sum().item()
    if predicted_count == 0 and target_count == 0:
        return BoundaryMetrics(1.0, 1.0, 1.0, boundary_iou)
    dilated_target = dilate_binary(target_boundary, tolerance) * valid
    dilated_prediction = dilate_binary(predicted_boundary, tolerance) * valid
    matched_prediction = (predicted_boundary * dilated_target).sum().item()
    matched_target = (target_boundary * dilated_prediction).sum().item()
    precision = matched_prediction / max(predicted_count, 1.0)
    recall = matched_target / max(target_count, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return BoundaryMetrics(precision, recall, f1, boundary_iou)

