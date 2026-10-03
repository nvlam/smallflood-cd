import torch

from smallflood_cd.data.boundary_targets import morphological_boundary


def test_boundary_excludes_solid_center() -> None:
    mask = torch.zeros(1, 1, 7, 7)
    mask[:, :, 2:5, 2:5] = 1
    boundary = morphological_boundary(mask)
    assert boundary[0, 0, 3, 3] == 0
    assert boundary.sum() > 0

