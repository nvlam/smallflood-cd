"""Additive patch-scoped object and masked inner-boundary diagnostics.

Not a scene-level instance evaluation. All distances/areas are in pixels.
"""
import numpy as np
from scipy import ndimage as ndi


PROTOCOL = {
    'version': 'object_boundary_patch_v2', 'connectivity': 8,
    'match_iou': 0.1, 'matching': 'greedy decreasing IoU, ties by prediction then GT ID; one-to-one',
    'component_scope': 'valid-masked patch components; do not merge across patches',
    'small_definition': 'GT area <= train-fitted threshold',
    'small_primary': 'recall of GT components not touching patch edge or 8-neighbor invalid pixels',
    'small_secondary': 'recall of all small patch-clipped GT components',
    'boundary': 'one-pixel inner contour; Chebyshev tolerance 2',
    'boundary_iou': 'IoU of two-pixel inner bands, not tolerance-dilated contours',
    'boundary_valid': 'valid mask eroded by 3 pixels; image exterior invalid',
    'undefined': 'zero denominator -> null; both boundaries empty -> F1/IoU null',
    'aggregation': 'sum counts before scores, separately global and per event; then event macro',
    'checkpoint_selection': 'unchanged event_pixel_v2; diagnostics do not select checkpoints',
}


def _erode(mask, radius):
    return ndi.binary_erosion(mask, structure=np.ones((3, 3), bool),
                              iterations=radius, border_value=0)


def _labels(mask, rim):
    labels, number = ndi.label(mask, structure=np.ones((3, 3), bool))
    areas = np.bincount(labels.ravel(), minlength=number + 1)
    censored = np.zeros(number + 1, dtype=bool)
    censored[np.unique(labels[rim])] = True
    censored[0] = False
    return labels, areas, censored, number


def patch_counts(prediction, target, valid, small_threshold):
    if small_threshold < 1:
        raise ValueError('small_threshold must be positive')
    p, t, v = (np.asarray(x, dtype=bool) for x in (prediction, target, valid))
    if p.ndim != 2 or p.shape != t.shape or p.shape != v.shape:
        raise ValueError('Require three identically shaped 2D masks')
    p, t = p & v, t & v
    rim = v & ~_erode(v, 1)
    pl, pa, pc, pn = _labels(p, rim)
    tl, ta, tc, tn = _labels(t, rim)
    overlap = (pl > 0) & (tl > 0)
    # Sparse overlap pairs avoid a dense N_prediction x N_GT allocation.
    keys, intersections = np.unique(pl[overlap].astype(np.int64) * (tn + 1) + tl[overlap],
                                     return_counts=True)
    candidates = []
    for key, intersection in zip(keys, intersections):
        pi, ti = divmod(int(key), tn + 1)
        iou = float(intersection / (pa[pi] + ta[ti] - intersection))
        if iou >= PROTOCOL['match_iou']:
            candidates.append((pi, ti, iou))
    candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
    used_p, used_t, matches = set(), set(), []
    for pi, ti, iou in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            matches.append((pi, ti, iou))
    small = {i for i in range(1, tn + 1) if ta[i] <= small_threshold}
    interior = {i for i in small if not tc[i]}
    safe = _erode(v, 3)
    pb, tb = (p & ~_erode(p, 1)) & safe, (t & ~_erode(t, 1)) & safe
    pd = ndi.binary_dilation(pb, structure=np.ones((3, 3), bool), iterations=2)
    td = ndi.binary_dilation(tb, structure=np.ones((3, 3), bool), iterations=2)
    pband, tband = (p & ~_erode(p, 2)) & safe, (t & ~_erode(t, 2)) & safe
    return {
        'patches': 1, 'component_predicted': pn, 'component_target': tn,
        'component_matched': len(matches), 'matched_iou_sum': sum(x[2] for x in matches),
        'predicted_touching_rim': int(pc.sum()), 'target_touching_rim': int(tc.sum()),
        'small_target_clipped': len(small), 'small_matched_clipped': len(small & used_t),
        'small_target_interior': len(interior), 'small_matched_interior': len(interior & used_t),
        'boundary_predicted': int(pb.sum()), 'boundary_target': int(tb.sum()),
        'boundary_matched_prediction': int((pb & td).sum()),
        'boundary_matched_target': int((tb & pd).sum()),
        'boundary_band_intersection': int((pband & tband).sum()),
        'boundary_band_union': int((pband | tband).sum()),
        'boundary_valid_pixels': int(safe.sum()),
    }


def add_counts(accumulator, counts):
    for key, value in counts.items():
        accumulator[key] = accumulator.get(key, 0) + value


def scores(c):
    def div(a, b):
        return a / b if b else None
    m, p, t = (c[k] for k in ('component_matched', 'component_predicted', 'component_target'))
    bp, bt = c['boundary_predicted'], c['boundary_target']
    precision = div(c['boundary_matched_prediction'], bp)
    recall = div(c['boundary_matched_target'], bt)
    f1 = None if bp + bt == 0 else 0.0
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        'component_precision': div(m, p), 'component_recall': div(m, t),
        'component_f1': div(2 * m, p + t), 'component_mean_matched_iou': div(c['matched_iou_sum'], m),
        'small_component_recall_interior': div(c['small_matched_interior'], c['small_target_interior']),
        'small_component_recall_clipped': div(c['small_matched_clipped'], c['small_target_clipped']),
        'boundary_precision': precision, 'boundary_recall': recall, 'boundary_f1': f1,
        'boundary_inner_band_iou': div(c['boundary_band_intersection'], c['boundary_band_union']),
    }


def summarize(by_event):
    total = {}
    events = []
    for event, counts in sorted(by_event.items()):
        add_counts(total, counts)
        events.append({'event': event, **counts, **scores(counts)})
    if not events:
        raise ValueError('No event counts')
    pooled = scores(total)
    macro, n = {}, {}
    for key in pooled:
        values = [row[key] for row in events if row[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
        n[key] = len(values)
    return {'counts': total, 'global': pooled, 'events': events,
            'event_macro': macro, 'event_macro_defined_counts': n}
