from pathlib import Path

from smallflood_cd.utils.config import load_experiment_config


def test_experiment_config_resolves_defaults() -> None:
    root = Path(__file__).parents[2]
    config = load_experiment_config(root / "configs/experiment/main.yaml")
    assert config["data"]["patch_size"] == 256
    assert config["model"]["max_parameters"] == 5_000_000
    assert config["training"]["epochs"] == 120
