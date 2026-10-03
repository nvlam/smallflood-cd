import torch

from smallflood_cd.metrics.boundary_metrics import boundary_metrics


def _logits(binary: torch.Tensor) -> torch.Tensor:
    return torch.where(binary > 0, torch.tensor(10.0), torch.tensor(-10.0))


def test_perfect_boundary_scores_one() -> None:
    target = torch.zeros(1, 1, 12, 12)
    target[:, :, 3:9, 3:9] = 1
    metrics = boundary_metrics(_logits(target), target, torch.ones_like(target))
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0


def test_tolerance_f1_accepts_offset_while_exact_iou_remains_strict() -> None:
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:10, 4:10] = 1
    prediction = torch.zeros_like(target)
    prediction[:, :, 4:10, 5:11] = 1
    metrics = boundary_metrics(
        _logits(prediction), target, torch.ones_like(target), tolerance=1
    )
    assert metrics.f1 == 1.0
    assert 0.0 < metrics.iou < 1.0


def test_invalid_boundary_region_is_excluded() -> None:
    target = torch.zeros(1, 1, 12, 12)
    target[:, :, 3:9, 3:9] = 1
    prediction = target.clone()
    prediction[:, :, 3:9, 8:10] = 0
    valid = torch.ones_like(target)
    valid[:, :, :, 7:] = 0
    metrics = boundary_metrics(_logits(prediction), target, valid, tolerance=0)
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0

