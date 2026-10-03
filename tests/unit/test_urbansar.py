import numpy as np
import pytest

from smallflood_cd.data.urbansar import ChannelMoments, assign_events, binary_labels, event_group


def test_tracks_and_tiles_of_one_flood_stay_together():
    assert event_group('20230805_Hebei_2_ID_19_33_SAR.tif') == '20230805_Hebei'
    assert event_group('20231201_Jubba_1_SAR.tif') == '20231201_Jubba'
    events = ['a', 'b', 'c', 'd', '20230805_Hebei_1', '20230805_Hebei_2']
    split = assign_events(events, ['20231201_Jubba_1', '20231201_Jubba_2'])
    assert split == assign_events(events[::-1], ['20231201_Jubba_2'])
    assert split['20231201_Jubba'] == 'test'
    assert set(split.values()) == {'train', 'validation', 'test'}


def test_binary_mapping_preserves_nonflood_and_ignores_invalid():
    labels = np.array([[0, 1, 2, 255]])
    valid = np.array([[True, True, True, False]])
    assert binary_labels(labels, valid).tolist() == [[0, 1, 1, 0]]
    with pytest.raises(ValueError):
        binary_labels(labels, np.ones_like(valid))


def test_streaming_moments_match_direct_pooled_training_values():
    pre = np.array([[[1., 3., 999.]], [[4., 8., 999.]]])
    post = pre + 2
    valid = np.array([[True, True, False]])
    moments = ChannelMoments()
    moments.update(pre, valid)
    moments.update(post, valid)
    pooled = np.concatenate([pre[:, valid], post[:, valid]], axis=1)
    np.testing.assert_allclose(moments.as_dict()['mean'], pooled.mean(axis=1))
    np.testing.assert_allclose(moments.as_dict()['std'], pooled.std(axis=1))
