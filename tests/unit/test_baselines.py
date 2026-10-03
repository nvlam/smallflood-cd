import pytest
import torch

from smallflood_cd.models.baselines.bit import BITBaseline
from smallflood_cd.models.baselines.fc_siam_diff import FCSiamDiff
from smallflood_cd.models.registry import build_model


@pytest.mark.parametrize("model", [FCSiamDiff(), BITBaseline()])
def test_baseline_output_contract_and_gradients(model) -> None:
    pre = torch.randn(1, 2, 64, 64)
    post = torch.randn(1, 2, 64, 64)
    output = model(pre, post)
    assert output.change_logits.shape == (1, 1, 64, 64)
    assert output.boundary_logits is None
    output.change_logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_registry_builds_frozen_baseline_names() -> None:
    assert isinstance(build_model({"name": "fc_siam_diff"}), FCSiamDiff)
    assert isinstance(build_model({"name": "bit"}), BITBaseline)


def test_baselines_report_parameter_counts() -> None:
    assert FCSiamDiff().parameter_count() > 0
    assert BITBaseline().parameter_count() > 0
