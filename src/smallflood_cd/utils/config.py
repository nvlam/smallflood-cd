from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous top-level values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return value


def deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating caller-owned objects."""
    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Resolve the lightweight `defaults` convention used by experiment files."""
    experiment_path = Path(path).resolve()
    experiment = load_yaml(experiment_path)
    defaults = experiment.pop("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Experiment defaults must be a mapping")
    layers: list[dict[str, Any]] = []
    for name in ("base", "data", "model"):
        relative = defaults.get(name)
        if relative:
            layers.append({name: load_yaml(experiment_path.parent / relative)})
    layers.append(experiment)
    return deep_merge(*layers)
