from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from smallflood_cd.data.connected_components import ConnectedComponentCache
from smallflood_cd.data.datasets import ManifestDataset
from smallflood_cd.data.preprocessing import load_array
from smallflood_cd.data.splits import validate_event_isolation
from smallflood_cd.data.uncertain_borders import apply_uncertain_border_mask
from smallflood_cd.engine.checkpointing import load_model_state
from smallflood_cd.engine.reproducibility import seed_everything
from smallflood_cd.engine.trainer import fit
from smallflood_cd.engine.validator import SELECTION_PROTOCOL, validate
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.metrics.evaluator import evaluate_change_detection
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_yaml


RESULT_FIELDS = [
    "experiment",
    "variant",
    "seed",
    "pixel_iou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "boundary_iou",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "component_precision",
    "component_recall",
    "component_f1",
    "component_mean_matched_iou",
    "component_small_component_recall",
    "inference_mean_ms",
    "inference_std_ms",
    "inference_p95_ms",
    "parameter_count",
    "checkpoint",
    "run_directory",
]


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def expand_grid(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    grid = config.pop("grid", None)
    if not grid:
        return [("default", config)]
    keys = sorted(grid)
    variants = []
    for values in product(*(grid[key] for key in keys)):
        variant = deepcopy(config)
        labels = []
        for key, value in zip(keys, values, strict=True):
            _set_dotted(variant, key, value)
            labels.append(f"{key.replace('.', '-')}-{value}")
        variants.append(("__".join(labels), variant))
    return variants


def load_matrix(path: str | Path) -> tuple[list[int], list[dict[str, Any]], Path]:
    matrix_path = Path(path).resolve()
    matrix = load_yaml(matrix_path)
    seeds = [int(seed) for seed in matrix["seeds"]]
    return seeds, list(matrix["experiments"]), matrix_path.parent


def _morphological_closing(binary: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(binary.float(), 3, stride=1, padding=1)
    return -F.max_pool2d(-dilated, 3, stride=1, padding=1)


def _shift(tensor: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels == 0:
        return tensor
    shifted = torch.roll(tensor, shifts=(pixels, pixels), dims=(-2, -1))
    shifted[..., :pixels, :] = 0
    shifted[..., :, :pixels] = 0
    return shifted


@torch.inference_mode()
def collect_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    evaluation: dict[str, Any],
    small_area_threshold: int,
    data_config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    for batch in loader:
        pre = batch["pre"].to(device)
        post = _shift(batch["post"].to(device), int(evaluation.get("shift_pixels", 0)))
        target = batch["target"].to(device)
        valid = batch["valid_mask"].to(device)
        uncertain = batch.get("uncertain_mask")
        if isinstance(uncertain, torch.Tensor):
            valid = apply_uncertain_border_mask(
                valid,
                uncertain.to(device),
                int(data_config.get("uncertainty_dilation", 1)),
            )
        coherence = batch.get("coherence")
        if isinstance(coherence, torch.Tensor):
            coherence = coherence.to(device)
        logits = model(pre, post, coherence).change_logits
        if evaluation.get("morphology") == "closing":
            binary = (torch.sigmoid(logits) >= evaluation["probability_threshold"]).float()
            binary = _morphological_closing(binary)
            logits = torch.where(binary > 0.5, 20.0, -20.0)
        for index in range(pre.shape[0]):
            report = evaluate_change_detection(
                logits[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                small_area_threshold,
                float(evaluation.get("probability_threshold", 0.5)),
                float(evaluation.get("component_match_iou", 0.1)),
                int(data_config.get("boundary_radius", 1)),
                int(data_config.get("boundary_tolerance", 2)),
                int(data_config.get("connectivity", 8)),
            )
            rows.append(report.to_flat_dict())
    if not rows:
        raise RuntimeError("Evaluation split is empty")
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    return {key: float(statistics.fmean(float(row[key]) for row in rows)) for key in numeric_keys}


@torch.inference_mode()
def benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
    channels: int,
    patch_size: int,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    model.eval().to(device)
    pre = torch.zeros(1, channels, patch_size, patch_size, device=device)
    post = torch.zeros_like(pre)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    for _ in range(warmup_runs):
        model(pre, post)
    synchronize()
    durations = []
    for _ in range(timed_runs):
        synchronize()
        start = time.perf_counter_ns()
        model(pre, post)
        synchronize()
        durations.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "inference_mean_ms": statistics.fmean(durations),
        "inference_std_ms": statistics.pstdev(durations),
        "inference_p95_ms": ordered[p95_index],
    }


def _load_component_stats(path: str | Path) -> tuple[float, int]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(values["reference_area"]), int(values["small_area_threshold"])


def _dataset(
    config: dict[str, Any], split: str, loss_config: dict[str, Any]
) -> tuple[ManifestDataset, int]:
    data = config["data"]
    reference_area, small_threshold = _load_component_stats(data["component_stats"])
    use_size = bool(loss_config.get("use_size_aware", False))
    cache = ConnectedComponentCache(
        Path(data["root"]) / "component_cache", int(data.get("connectivity", 8))
    ) if use_size else None
    return (
        ManifestDataset(
            data["manifest"],
            split,
            load_array,
            component_cache=cache,
            reference_component_area=reference_area if use_size else None,
            component_alpha=float(loss_config.get("component_alpha", 1.0)),
            component_gamma=float(loss_config.get("component_gamma", 0.5)),
            component_weight_cap=float(loss_config.get("component_weight_cap", 4.0)),
            use_uncertain_mask=bool(data.get("use_uncertain_mask", True)),
        ),
        small_threshold,
    )


def run_experiment(
    name: str,
    variant: str,
    seed: int,
    config: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    config = deepcopy(config)
    config["selection"] = dict(SELECTION_PROTOCOL)
    config["base"]["seed"] = seed
    seed_everything(seed, bool(config["base"].get("deterministic", True)))
    validate_event_isolation(config["data"]["manifest"])
    run_directory = Path(output_root) / name / variant / f"seed-{seed}"
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (run_directory / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "seed": seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    train_dataset, small_threshold = _dataset(config, "train", config["loss"])
    validation_dataset, _ = _dataset(config, "validation", config["loss"])
    test_dataset, _ = _dataset(config, "test", config["loss"])
    training = config["training"]
    common_loader = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(config["base"].get("num_workers", 0)),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_loader)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader)
    device_name = config["base"].get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else (
            "cpu" if device_name == "auto" else device_name
        )
    )
    model = build_model(config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(training["epochs"])
    )
    loss_config = config["loss"]
    loss = SmallFloodLoss(
        float(loss_config["bce_weight"]),
        float(loss_config["tversky_weight"]),
        float(loss_config["boundary_weight"]),
        float(loss_config.get("tversky_fp", 0.3)),
        float(loss_config.get("tversky_fn", 0.7)),
    )

    def validation_callback() -> dict:
        # Checkpoint selection is unshifted and without morphology for all variants.
        return validate(model, validation_loader, device, threshold=0.5,
                        uncertainty_dilation=int(config["data"].get("uncertainty_dilation", 1)))

    training_result = fit(
        model,
        train_loader,
        validation_loader,
        loss,
        optimizer,
        scheduler,
        device,
        int(training["epochs"]),
        run_directory,
        config,
        int(training["patience"]),
        float(training["gradient_clip_norm"]),
        validation_callback,
    )
    load_model_state(training_result.best_checkpoint, model, device)
    metrics = collect_metrics(
        model, test_loader, device, config["evaluation"], small_threshold, config["data"]
    )
    timing = benchmark_inference(
        model,
        device,
        int(config["model"].get("input_channels", 2)),
        int(config["data"]["patch_size"]),
        int(config["timing"]["warmup_runs"]),
        int(config["timing"]["timed_runs"]),
    )
    result: dict[str, Any] = {
        "experiment": name,
        "variant": variant,
        "seed": seed,
        **metrics,
        **timing,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint": str(training_result.best_checkpoint),
        "run_directory": str(run_directory),
    }
    with (run_directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    (run_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def write_combined_csv(rows: list[dict[str, Any]], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
