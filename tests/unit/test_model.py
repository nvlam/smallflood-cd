import torch

from smallflood_cd.models import SmallFloodCDNet


def test_model_shapes_and_budget() -> None:
    model = SmallFloodCDNet()
    pre = torch.randn(2, 2, 256, 256)
    post = torch.randn(2, 2, 256, 256)
    output = model(pre, post)
    assert output.change_logits.shape == (2, 1, 256, 256)
    assert output.boundary_logits is not None
    assert output.boundary_logits.shape == (2, 1, 256, 256)
    model.assert_parameter_budget()

