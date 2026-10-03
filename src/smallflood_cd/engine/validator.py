from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch.utils.data import DataLoader

from smallflood_cd.data.uncertain_borders import apply_uncertain_border_mask

SELECTION_PROTOCOL = {
    "version": "event_pixel_v2", "metric": "event_macro_f1", "mode": "max",
    "probability_threshold": 0.5,
    "aggregation": "sum valid-pixel confusion counts within event, then equal event mean",
    "undefined_policy": "null for zero denominator; omit from macro and report count",
    "tie_policy": "keep earlier checkpoint",
    "checkpoint_filename": "best_composite.ckpt (legacy name; not composite scoring)",
}


def metrics_from_counts(counts):
    tp, fp, fn, tn = map(int, counts)
    def divide(a, b):
        return a / b if b else None
    return {"precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "f1": divide(2 * tp, 2 * tp + fp + fn), "iou": divide(tp, tp + fp + fn),
            "accuracy": divide(tp + tn, tp + fp + fn + tn)}


def summarize_events(by_event):
    if not by_event:
        raise ValueError("No validation events")
    total = [sum(int(c[i]) for c in by_event.values()) for i in range(4)]
    if sum(total) == 0:
        raise ValueError("No valid validation pixels")
    events = {str(event): metrics_from_counts(c) for event, c in by_event.items()}
    result = {f"global_pixel_{key}": value for key, value in metrics_from_counts(total).items()}
    result.update({f"global_{key}": value for key, value in zip(("tp", "fp", "fn", "tn"), total)})
    result["event_count"] = len(events)
    for metric in ("precision", "recall", "f1", "iou", "accuracy"):
        defined = [m[metric] for m in events.values() if m[metric] is not None]
        result[f"event_macro_{metric}"] = sum(defined) / len(defined) if defined else None
        result[f"event_macro_{metric}_defined_count"] = len(defined)
    for event, metrics in events.items():
        for key, value in metrics.items():
            result[f"event/{event}/{key}"] = value
    result["selection_score"] = checkpoint_score(result)
    return result


def checkpoint_score(metrics):
    value = metrics.get("event_macro_f1")
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Finite event_macro_f1 required for checkpoint selection")
    return float(value)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    uncertainty_dilation: int = 1,
) -> dict:
    if threshold != 0.5:
        raise ValueError("event_pixel_v2 selection uses fixed threshold 0.5")
    model.eval()
    by_event = defaultdict(lambda: torch.zeros(4, dtype=torch.int64, device=device))
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid_mask"].to(device)
        uncertain = batch.get("uncertain_mask")
        if isinstance(uncertain, torch.Tensor):
            valid = apply_uncertain_border_mask(valid, uncertain.to(device), uncertainty_dilation)
        valid = valid > 0.5
        if not torch.isfinite(pre).all() or not torch.isfinite(post).all():
            raise ValueError("Nonfinite validation input")
        if not ((target[valid] == 0) | (target[valid] == 1)).all():
            raise ValueError("Validation targets must be binary on valid pixels")
        coherence = batch.get("coherence")
        if isinstance(coherence, torch.Tensor):
            coherence = coherence.to(device)
        output = model(pre, post, coherence)
        if output.change_logits.shape != target.shape or not torch.isfinite(output.change_logits).all():
            raise ValueError("Invalid validation logits")
        pred, truth = output.change_logits.sigmoid() >= threshold, target > 0.5
        event_ids = batch["event_id"]
        for index, event_id in enumerate(event_ids):
            p, t, v = pred[index], truth[index], valid[index]
            by_event[str(event_id)] += torch.stack(((p & t & v).sum(), (p & ~t & v).sum(),
                                                  (~p & t & v).sum(), (~p & ~t & v).sum()))
    return summarize_events({event: counts.cpu().tolist() for event, counts in by_event.items()})
