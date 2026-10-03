import numpy as np
import pytest

from smallflood_cd.metrics.object_boundary_v2 import patch_counts, scores, summarize


def canvas():
    return np.zeros((32, 32), bool), np.ones((32, 32), bool)


def test_perfect_interior_small_object():
    t, v = canvas()
    t[10:14, 10:14] = True
    c = patch_counts(t, t, v, 35)
    s = scores(c)
    assert c['component_matched'] == 1
    assert c['small_target_interior'] == 1
    for key in ('component_precision', 'component_recall', 'component_f1',
                'small_component_recall_interior', 'boundary_f1', 'boundary_inner_band_iou'):
        assert s[key] == 1


def test_empty_and_false_positive():
    t, v = canvas()
    s = scores(patch_counts(t, t, v, 35))
    assert s['component_f1'] is None
    assert s['boundary_f1'] is None
    assert s['boundary_inner_band_iou'] is None
    p = t.copy()
    p[10:12, 10:12] = True
    s = scores(patch_counts(p, t, v, 35))
    assert s['component_f1'] == 0
    assert s['boundary_f1'] == 0
    assert s['small_component_recall_interior'] is None


def test_missed_small_and_threshold_inclusive():
    t, v = canvas()
    t[10:15, 10:17] = True  # exactly 35
    c = patch_counts(np.zeros_like(t), t, v, 35)
    assert c['small_target_interior'] == 1
    assert scores(c)['small_component_recall_interior'] == 0
    assert patch_counts(t, t, v, 34)['small_target_interior'] == 0


def test_border_component_is_reported_not_called_complete_small_object():
    t, v = canvas()
    t[0:2, 10:12] = True
    c = patch_counts(t, t, v, 35)
    assert c['small_target_clipped'] == 1
    assert c['target_touching_rim'] == 1
    assert c['small_target_interior'] == 0
    assert scores(c)['small_component_recall_interior'] is None
    assert scores(c)['small_component_recall_clipped'] == 1


def test_invalid_border_does_not_make_an_artificial_boundary():
    t, v = canvas()
    v[:, 16:] = False
    t[:, :] = True
    c = patch_counts(t, t, v, 35)
    assert c['boundary_predicted'] == c['boundary_target'] == 0
    assert c['target_touching_rim'] == 1
    v[:] = False
    c = patch_counts(t, t, v, 35)
    assert c['component_target'] == 0
    assert c['boundary_valid_pixels'] == 0


def test_merge_and_split_match_once():
    t, v = canvas()
    t[10:14, 7:11] = True
    t[10:14, 15:19] = True
    p = t.copy()
    p[11, 11:15] = True
    c = patch_counts(p, t, v, 35)
    assert (c['component_predicted'], c['component_target'], c['component_matched']) == (1, 2, 1)
    assert scores(c)['small_component_recall_interior'] == 0.5
    c = patch_counts(t, p, v, 35)
    assert (c['component_predicted'], c['component_target'], c['component_matched']) == (2, 1, 1)


def test_tolerance_fscore_is_not_inner_band_iou():
    t, v = canvas()
    t[10:20, 10:20] = True
    p = np.roll(t, 1, axis=1)
    s = scores(patch_counts(p, t, v, 35))
    assert s['boundary_f1'] == 1
    assert 0 < s['boundary_inner_band_iou'] < 1


def test_counts_pooled_before_scores_and_event_macro():
    t, v = canvas()
    t[10:14, 10:14] = True
    perfect = patch_counts(t, t, v, 35)
    missed = patch_counts(np.zeros_like(t), t, v, 35)
    r = summarize({'perfect': perfect, 'missed': missed})
    assert r['global']['component_f1'] == pytest.approx(2 / 3)
    assert r['event_macro']['component_f1'] == 0.5
    assert r['counts']['small_target_interior'] == 2
    assert r['global']['small_component_recall_interior'] == 0.5
