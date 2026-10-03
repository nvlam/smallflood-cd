from __future__ import annotations

import argparse

import torch

from smallflood_cd.deployment.onnx_export import export_onnx
from smallflood_cd.engine.checkpointing import load_model_state
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SmallFlood-CDNet to ONNX")
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    model = build_model(config["model"])
    load_model_state(arguments.checkpoint, model, torch.device("cpu"))
    export_onnx(
        model,
        arguments.output,
        input_channels=int(config["model"]["input_channels"]),
        patch_size=int(config["data"]["patch_size"]),
    )


if __name__ == "__main__":
    main()
