import numpy as np

from smallflood_cd.data.connected_components import (
    ConnectedComponentCache,
    component_areas,
    label_connected_components,
    reference_component_area,
)


def test_connectivity_changes_diagonal_component_count() -> None:
    mask = np.eye(3, dtype=np.uint8)
    labels_four = label_connected_components(mask, connectivity=4)
    labels_eight = label_connected_components(mask, connectivity=8)
    assert component_areas(labels_four).tolist() == [1, 1, 1]
    assert component_areas(labels_eight).tolist() == [3]


def test_component_cache_is_content_addressed(tmp_path) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    cache = ConnectedComponentCache(tmp_path, connectivity=8)
    first = cache.get_or_create(mask)
    second = cache.get_or_create(mask.copy())
    assert not first.cache_hit
    assert second.cache_hit
    assert first.cache_path == second.cache_path
    np.testing.assert_array_equal(first.labels, second.labels)


def test_reference_area_is_training_component_median() -> None:
    first = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int32)
    second = np.array([[1, 0, 2], [1, 0, 2]], dtype=np.int32)
    assert reference_component_area([first, second]) == 2.0

