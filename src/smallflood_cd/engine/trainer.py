from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from smallflood_cd.data.boundary_supervision import build_boundary_supervision
from smallflood_cd.data.uncertain_borders import apply_uncertain_border_mask
from smallflood_cd.engine.checkpointing import save_checkpoint
from smallflood_cd.engine.validator import SELECTION_PROTOCOL, checkpoint_score, validate
from smallflood_cd.losses import SmallFloodLoss


@dataclass
class TrainingResult:
    best_score: float
    best_checkpoint: Path
    epochs_completed: int


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: SmallFloodLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    output_directory: str | Path,
    config: dict[str, Any],
    patience: int = 15,
    gradient_clip_norm: float = 1.0,
    validation_callback: Callable[[], dict[str, float]] | None = None,
) -> TrainingResult:
    output_directory = Path(output_directory)
    if (output_directory / "metrics.jsonl").exists() or (output_directory / "checkpoints").exists():
        raise FileExistsError("Use a new run directory; implicit resume/overwrite is not supported")
    best_checkpoint = output_directory / "checkpoints" / "best_composite.ckpt"
    best_score, stale_epochs, global_step = float("-inf"), 0, 0
    model.to(device)
    output_directory.mkdir(parents=True, exist_ok=True)
    config = deepcopy(config)
    config["selection"] = dict(SELECTION_PROTOCOL)
    (output_directory / "selection_protocol.json").write_text(json.dumps(SELECTION_PROTOCOL, indent=2))
    epoch_log = output_directory / "metrics.jsonl"
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            target = batch["target"].to(device).float()
            valid = batch["valid_mask"].to(device).float()
            coherence = batch.get("coherence")
            if isinstance(coherence, torch.Tensor):
                coherence = coherence.to(device)
            component_weights = batch.get("component_weights")
            if not isinstance(component_weights, torch.Tensor):
                component_weights = torch.ones_like(target)
            else:
                component_weights = component_weights.to(device)
            uncertain = batch.get("uncertain_mask")
            if isinstance(uncertain, torch.Tensor):
                uncertain = uncertain.to(device)
                valid = apply_uncertain_border_mask(valid, uncertain)
            boundary = build_boundary_supervision(target, valid, uncertain)
            optimizer.zero_grad(set_to_none=True)
            output = model(pre, post, coherence)
            losses = loss_function(
                output,
                target,
                component_weights,
                valid,
                boundary.target,
                boundary.valid_mask,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(losses.total.detach().cpu())
            batch_count += 1
            global_step += 1
        if scheduler is not None:
            scheduler.step()
        metrics = (
            validation_callback()
            if validation_callback is not None
            else validate(model, validation_loader, device,
                          uncertainty_dilation=int(config.get("data", {}).get("uncertainty_dilation", 1)))
        )
        score = checkpoint_score(metrics)
        metrics["selection_score"] = score
        with epoch_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": epoch_loss / max(batch_count, 1),
                        **{f"validation_{key}": value for key, value in metrics.items()},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        save_checkpoint(
            output_directory / "checkpoints" / "last.ckpt",
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            max(best_score, score),
            config,
        )
        if score > best_score:
            best_score, stale_epochs = score, 0
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_score,
                config,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                return TrainingResult(best_score, best_checkpoint, epoch + 1)
    return TrainingResult(best_score, best_checkpoint, epochs)
