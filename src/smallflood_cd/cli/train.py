from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smallflood_cd.data.datasets import ManifestDataset
from smallflood_cd.data.preprocessing import load_array
from smallflood_cd.data.splits import validate_event_isolation
from smallflood_cd.engine.reproducibility import seed_everything
from smallflood_cd.engine.trainer import fit
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmallFlood-CDNet")
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--run-dir")
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    seed_everything(int(config["base"]["seed"]), bool(config["base"]["deterministic"]))
    data_config, training = config["data"], config["training"]
    validate_event_isolation(data_config["manifest"])
    train_dataset = ManifestDataset(data_config["manifest"], "train", load_array)
    validation_dataset = ManifestDataset(data_config["manifest"], "validation", load_array)
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Both train and validation splits must contain patches")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(config["base"]["num_workers"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(config["base"]["num_workers"]),
    )
    model = build_model(config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    loss_config = config["loss"]
    loss = SmallFloodLoss(
        bce_weight=float(loss_config["bce_weight"]),
        tversky_weight=float(loss_config["tversky_weight"]),
        boundary_weight=float(loss_config["boundary_weight"]),
        false_positive_weight=float(loss_config["tversky_fp"]),
        false_negative_weight=float(loss_config["tversky_fn"]),
    )
    run_directory = arguments.run_dir or (
        Path(config["base"]["output_root"])
        / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S_smallflood")
    )
    result = fit(
        model,
        train_loader,
        validation_loader,
        loss,
        optimizer,
        scheduler,
        _device(config["base"]["device"]),
        int(training["epochs"]),
        run_directory,
        config,
        int(training["patience"]),
        float(training["gradient_clip_norm"]),
    )
    print(f"best_f1={result.best_score:.6f} checkpoint={result.best_checkpoint}")


if __name__ == "__main__":
    main()
