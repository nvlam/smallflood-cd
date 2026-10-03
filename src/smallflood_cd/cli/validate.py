from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from smallflood_cd.data.datasets import ManifestDataset
from smallflood_cd.data.preprocessing import load_array
from smallflood_cd.engine.checkpointing import load_model_state
from smallflood_cd.engine.validator import validate
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SmallFlood-CDNet checkpoint")
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--threshold", type=float, default=0.5)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    load_model_state(arguments.checkpoint, model, device)
    dataset = ManifestDataset(config["data"]["manifest"], arguments.split, load_array)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    metrics = validate(model, loader, device, arguments.threshold)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
