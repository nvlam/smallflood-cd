import pytest
import torch

from smallflood_cd.metrics.component_metrics import component_metrics


def _logits(binary: torch.Tensor) -> torch.Tensor:
    return torch.where(binary > 0, torch.tensor(10.0), torch.tensor(-10.0))


def test_perfect_components_score_one() -> None:
    target = torch.zeros(1, 1, 12, 12)
    target[:, :, 1:3, 1:3] = 1
    target[:, :, 7:11, 7:11] = 1
    metrics = component_metrics(
        _logits(target), target, torch.ones_like(target), small_area_threshold=4
    )
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.small_component_recall == 1.0
    assert metrics.mean_matched_iou == 1.0


def test_small_component_recall_detects_missed_small_region() -> None:
    target = torch.zeros(1, 1, 12, 12)
    target[:, :, 1:3, 1:3] = 1  # area 4, small
    target[:, :, 6:11, 6:11] = 1  # area 25, large
    prediction = torch.zeros_like(target)
    prediction[:, :, 6:11, 6:11] = 1
    metrics = component_metrics(
        _logits(prediction), target, torch.ones_like(target), small_area_threshold=4
    )
    assert metrics.recall == 0.5
    assert metrics.small_component_recall == 0.0


def test_one_prediction_cannot_match_two_targets() -> None:
    target = torch.zeros(1, 1, 7, 9)
    target[:, :, 2:5, 1:3] = 1
    target[:, :, 2:5, 6:8] = 1
    prediction = torch.zeros_like(target)
    prediction[:, :, 2:5, 1:8] = 1  # one bridge joins both targets
    metrics = component_metrics(
        _logits(prediction),
        target,
        torch.ones_like(target),
        small_area_threshold=6,
        match_iou_threshold=0.1,
    )
    assert metrics.matched_components == 1
    assert metrics.predicted_components == 1
    assert metrics.target_components == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)

