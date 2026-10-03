from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from smallflood_cd.data.connected_components import component_areas, label_connected_components


@dataclass(frozen=True)
class ComponentMetrics:
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    small_component_recall: float
    matched_components: int
    predicted_components: int
    target_components: int
    small_target_components: int


def _component_iou_matrix(
    predicted_labels: np.ndarray,
    target_labels: np.ndarray,
) -> np.ndarray:
    predicted_count = int(predicted_labels.max())
    target_count = int(target_labels.max())
    matrix = np.zeros((predicted_count, target_count), dtype=np.float64)
    if predicted_count == 0 or target_count == 0:
        return matrix
    predicted_areas = component_areas(predicted_labels)
    target_areas = component_areas(target_labels)
    for predicted_id in range(1, predicted_count + 1):
        predicted_mask = predicted_labels == predicted_id
        overlapping_targets = np.unique(target_labels[predicted_mask])
        for target_id in overlapping_targets:
            if target_id == 0:
                continue
            intersection = np.count_nonzero(predicted_mask & (target_labels == target_id))
            union = predicted_areas[predicted_id - 1] + target_areas[target_id - 1] - intersection
            matrix[predicted_id - 1, target_id - 1] = intersection / max(union, 1)
    return matrix


def _greedy_one_to_one_matches(
    iou_matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = [
        (predicted, target, float(iou_matrix[predicted, target]))
        for predicted, target in zip(*np.where(iou_matrix >= threshold), strict=True)
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches = []
    for predicted, target, iou in candidates:
        if predicted in used_predictions or target in used_targets:
            continue
        used_predictions.add(predicted)
        used_targets.add(target)
        matches.append((predicted, target, iou))
    return matches


def component_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    small_area_threshold: int,
    probability_threshold: float = 0.5,
    match_iou_threshold: float = 0.1,
    connectivity: int = 8,
) -> ComponentMetrics:
    """Calculate one-to-one object metrics over a batch of binary masks."""
    if small_area_threshold < 1:
        raise ValueError("Small-component threshold must be positive")
    if not 0 <= match_iou_threshold <= 1:
        raise ValueError("Component IoU threshold must be in [0, 1]")
    predictions = (torch.sigmoid(logits) >= probability_threshold).detach().cpu().numpy()
    targets = target.bool().detach().cpu().numpy()
    valid = valid_mask.bool().detach().cpu().numpy()
    matched_total = predicted_total = target_total = 0
    small_total = small_matched = 0
    matched_ious: list[float] = []
    for index in range(predictions.shape[0]):
        prediction = np.squeeze(predictions[index]) & np.squeeze(valid[index])
        truth = np.squeeze(targets[index]) & np.squeeze(valid[index])
        predicted_labels = label_connected_components(prediction, connectivity)
        target_labels = label_connected_components(truth, connectivity)
        target_areas = component_areas(target_labels)
        matrix = _component_iou_matrix(predicted_labels, target_labels)
        matches = _greedy_one_to_one_matches(matrix, match_iou_threshold)
        matched_targets = {target_index for _, target_index, _ in matches}
        matched_total += len(matches)
        predicted_total += int(predicted_labels.max())
        target_total += int(target_labels.max())
        matched_ious.extend(iou for _, _, iou in matches)
        small_indices = {
            component_index
            for component_index, area in enumerate(target_areas)
            if area <= small_area_threshold
        }
        small_total += len(small_indices)
        small_matched += len(small_indices & matched_targets)
    precision = matched_total / max(predicted_total, 1)
    recall = matched_total / max(target_total, 1)
    f1 = 2 * matched_total / max(predicted_total + target_total, 1)
    return ComponentMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=(sum(matched_ious) / len(matched_ious) if matched_ious else 0.0),
        small_component_recall=small_matched / max(small_total, 1),
        matched_components=matched_total,
        predicted_components=predicted_total,
        target_components=target_total,
        small_target_components=small_total,
    )

