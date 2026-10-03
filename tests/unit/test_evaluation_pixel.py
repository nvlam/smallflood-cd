import torch

from smallflood_cd.metrics.pixel_metrics import binary_metrics


def _logits(binary: torch.Tensor) -> torch.Tensor:
    return torch.where(binary > 0, torch.tensor(10.0), torch.tensor(-10.0))


def test_pixel_metrics_match_known_confusion_matrix() -> None:
    target = torch.tensor([[[[1, 1, 0, 0]]]])
    prediction = torch.tensor([[[[1, 0, 1, 0]]]])
    metrics = binary_metrics(_logits(prediction), target, torch.ones_like(target))
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.iou == 1 / 3


def test_invalid_pixel_is_excluded_from_pixel_metrics() -> None:
    target = torch.tensor([[[[1, 0]]]])
    prediction = torch.tensor([[[[1, 1]]]])
    valid = torch.tensor([[[[1, 0]]]])
    metrics = binary_metrics(_logits(prediction), target, valid)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0

