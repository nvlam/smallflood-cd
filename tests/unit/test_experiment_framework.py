import csv
from pathlib import Path

from smallflood_cd.engine.experiment import RESULT_FIELDS, expand_grid, load_matrix, write_combined_csv
from smallflood_cd.utils.config import load_experiment_config


def test_matrix_resolves_every_experiment_and_seed() -> None:
    root = Path(__file__).parents[2]
    matrix_path = root / "configs/experiment/experiment_matrix.yaml"
    seeds, entries, matrix_directory = load_matrix(matrix_path)
    assert seeds == [42, 1337, 2026]
    names = {entry["name"] for entry in entries}
    assert {"proposed", "fc_siam_diff", "bit", "shift_robustness"}.issubset(names)
    for entry in entries:
        config = load_experiment_config(matrix_directory / entry["config"])
        assert "model" in config
        assert "training" in config
        assert "evaluation" in config


def test_grid_expansion_is_deterministic_and_complete() -> None:
    config = {
        "grid": {"loss.gamma": [0.5, 1.0], "evaluation.shift": [0, 1]},
        "loss": {"gamma": 0.0},
        "evaluation": {"shift": 0},
    }
    variants = expand_grid(config)
    assert len(variants) == 4
    assert variants[0][0] == "evaluation-shift-0__loss-gamma-0.5"
    assert variants[-1][1]["loss"]["gamma"] == 1.0


def test_combined_csv_has_frozen_metric_schema(tmp_path) -> None:
    row = {field: 0 for field in RESULT_FIELDS}
    destination = tmp_path / "all_results.csv"
    write_combined_csv([row], destination)
    with destination.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RESULT_FIELDS
        assert len(list(reader)) == 1


def test_result_schema_contains_all_requested_outputs() -> None:
    required = {
        "pixel_iou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "boundary_iou",
        "boundary_f1",
        "component_small_component_recall",
        "inference_mean_ms",
        "parameter_count",
        "checkpoint",
    }
    assert required.issubset(RESULT_FIELDS)
