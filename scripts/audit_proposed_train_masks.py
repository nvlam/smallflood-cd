"""Read train label/valid NPYs only; no cache, model, optimizer or test arrays."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy import ndimage as ndi
import torch
from smallflood_cd.data.datasets import read_manifest
from smallflood_cd.data.boundary_supervision import build_boundary_supervision


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(target, valid, reference=65.):
    if target.shape != valid.shape or target.ndim != 2:
        raise ValueError('Expected equal 2-D masks')
    if not np.isin(target, [0, 1]).all() or not np.isin(valid, [0, 1]).all():
        raise ValueError('Nonfinite or nonbinary mask')
    fg, v = target.astype(bool), valid.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    original, n = ndi.label(fg, structure)
    masked, m = ndi.label(fg & v, structure)
    areas = np.bincount(original.ravel())[1:]
    masked_areas = np.bincount(masked.ravel())[1:]
    lookup = np.ones(n + 1)
    lookup[1:] = 1 + np.minimum(4, np.sqrt(reference / (areas + 1e-6)))
    masked_lookup = np.ones(m + 1)
    masked_lookup[1:] = 1 + np.minimum(4, np.sqrt(reference / (masked_areas + 1e-6)))
    weights = lookup[original]
    b = build_boundary_supervision(torch.from_numpy(target.copy())[None, None],
                                   torch.from_numpy(valid.copy())[None, None])
    boundary = b.target.numpy()[0, 0] > .5
    near_invalid = ndi.binary_dilation(~v, structure=structure)
    inner3 = ndi.binary_erosion(v, structure=structure, iterations=3, border_value=0)
    supervised = boundary & v
    small = (original > 0) & (np.r_[0, areas][original] <= 35) & v
    counts = dict(patches=1, pixels=fg.size, valid_pixels=int(v.sum()),
        foreground=int(fg.sum()), invalid_foreground=int((fg & ~v).sum()),
        valid_foreground=int((fg & v).sum()), components=n, masked_components=m,
        patches_with_invalid_foreground=int((fg & ~v).any()),
        patches_with_component_count_change=int(n != m),
        valid_foreground_weight_changed_by_mask=int(((weights != masked_lookup[masked]) & fg & v).sum()),
        boundary_supervised=int(supervised.sum()),
        boundary_supervised_near_invalid_r1=int((supervised & near_invalid).sum()),
        boundary_supervised_outside_eval_interior_r3=int((supervised & ~inner3).sum()),
        valid_foreground_at_weight_cap=int(((weights >= 5) & fg & v).sum()),
        valid_foreground_weight_mass=float(weights[fg & v].sum()),
        valid_small_foreground_weight_mass=float(weights[small].sum()))
    return counts, Counter(map(int, areas)), Counter(map(int, masked_areas))


def run(manifest, stats_path, output):
    stats = json.loads(Path(stats_path).read_text())
    if stats.get('fitted_split') != 'train' or stats.get('connectivity') != 8:
        raise ValueError('Expected train-fitted 8-connected statistics')
    if stats.get('reference_area') != 65 or stats.get('small_area_threshold') != 35:
        raise ValueError('Frozen Proposed statistics changed')
    records = sorted(read_manifest(manifest, 'train'), key=lambda r: (r.event_id, r.patch_id))
    if not records or len({r.patch_id for r in records}) != len(records):
        raise ValueError('Empty train or duplicate IDs')
    if any(r.uncertain_mask_path for r in records):
        raise ValueError('Uncertain masks present: this audit requires a separate protocol')
    before = (sha(manifest), sha(stats_path))
    output.mkdir(parents=True, exist_ok=False)
    totals, events, hist, masked_hist = Counter(), defaultdict(Counter), Counter(), Counter()
    for i, r in enumerate(records, 1):
        arrays = []
        for filename in (r.label_path, r.valid_mask_path):
            if Path(filename).suffix != '.npy':
                raise ValueError('Only prepared NPY masks supported')
            a = np.load(filename, allow_pickle=False)
            if a.shape == (1, 256, 256):
                a = a[0]
            if a.shape != (256, 256):
                raise ValueError(f'Unexpected patch shape: {filename}: {a.shape}')
            arrays.append(a)
        c, h, mh = inspect(*arrays, stats['reference_area'])
        totals.update(c); events[r.event_id].update(c); hist.update(h); masked_hist.update(mh)
        if i % 500 == 0:
            print(f'Checked {i}/{len(records)} TRAIN masks', flush=True)
    if before != (sha(manifest), sha(stats_path)):
        raise RuntimeError('Manifest/statistics changed during audit')
    report = dict(status='COMPLETE', split='train', test_used=False, validation_arrays_used=False,
        model_loaded=False, cache_used=False, training_performed=False,
        manifest_sha256=before[0], component_stats_sha256=before[1], script_sha256=sha(__file__),
        scope='full train label/valid scan; no raw SAR or cache writes; no gradient measurement',
        weight_formula='1 + min(4, sqrt(65/(area+1e-6))); background=1',
        boundary_note='r1 counts actual invalid neighbors; r3 also excludes patch exterior, as evaluation does; neither proves annotation error',
        counts=dict(totals), events={k:dict(v) for k,v in events.items()},
        target_histogram_matches_stats=dict(hist)=={int(k):v for k,v in stats['area_histogram'].items()},
        target_area_histogram=dict(sorted(hist.items())), valid_masked_area_histogram=dict(sorted(masked_hist.items())))
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'output':str(output), 'counts':dict(totals),
                      'target_histogram_matches_stats':report['target_histogram_matches_stats']}, indent=2))
    print('TRAIN MASK AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f'Run from {ROOT}')
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--manifest', type=Path, default=Path('data/processed/urbansarfloods/patch_manifest.csv'))
    p.add_argument('--stats', type=Path, default=Path('data/metadata/component_stats_v1.json'))
    p.add_argument('--output', type=Path, default=Path('artifacts/proposed_train_mask_audit') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    a = p.parse_args()
    if a.self_test:
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q',
                                     'tests/unit/test_audit_proposed_train_masks.py']))
    run(a.manifest, a.stats, a.output)
