import torch

from smallflood_cd.metrics.evaluator import evaluate_change_detection


def test_unified_evaluator_maps_all_methodology_metrics() -> None:
    target = torch.zeros(1, 1, 10, 10)
    target[:, :, 2:5, 2:5] = 1
    logits = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
    report = evaluate_change_detection(
        logits, target, torch.ones_like(target), small_area_threshold=9
    )
    flattened = report.to_flat_dict()
    required = {
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_iou",
        "component_precision",
        "component_recall",
        "component_f1",
        "component_small_component_recall",
        "boundary_iou",
        "boundary_f1",
    }
    assert required.issubset(flattened)
    assert all(flattened[key] == 1.0 for key in required)

