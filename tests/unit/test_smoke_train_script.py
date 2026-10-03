import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def test_smoke_trains_reloads_all_models_without_test_data(tmp_path):
    root = Path(__file__).parents[2]
    shutil.copytree(root / 'configs', tmp_path / 'configs')
    processed = tmp_path / 'data/processed/urbansarfloods'
    processed.mkdir(parents=True)
    metadata = tmp_path / 'data/metadata'
    metadata.mkdir(parents=True)
    (metadata / 'component_stats_v1.json').write_text(json.dumps({
        'reference_area': 4, 'small_area_threshold': 4,
    }))
    rows = []
    rng = np.random.default_rng(42)
    for split in ('train', 'validation'):
        for i in range(2):
            patch = f'{split}_{i}'
            pre = rng.normal(size=(2, 32, 32)).astype(np.float32)
            post = pre + rng.normal(0, 0.5, pre.shape).astype(np.float32)
            label = np.zeros((1, 32, 32), np.float32)
            if i:
                label[:, 10:14, 10:14] = 1
            row = {'patch_id': patch, 'event_id': split + '_event', 'split': split,
                   'flood_pixels': int(label.sum())}
            for field, array in [('pre', pre), ('post', post), ('label', label),
                                 ('valid_mask', np.ones_like(label))]:
                destination = processed / f'{patch}_{field}.npy'
                np.save(destination, array)
                row[field + '_path'] = str(destination)
            rows.append(row)
    with (processed / 'patch_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = subprocess.run([
        sys.executable, str(root / 'scripts/smoke_train.py'), '--device', 'cpu',
        '--train-patches', '2', '--val-patches', '2', '--batch-size', '2',
    ], cwd=tmp_path, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ALL SMOKE TRAINING CHECKS PASSED' in result.stdout
    report = json.loads(next((tmp_path / 'runs/smoke').glob('*/summary.json')).read_text())
    assert len(report) == 3
    for row in report:
        assert row['status'] == 'PASS'
        assert row['optimizer_steps'] == 1
        assert row['changed_parameter_tensors'] > 0
        assert row['checkpoint_reload_verified']
        assert not row['test_used']

    # Exercise the strict launcher twice in fresh interpreters, without test data.
    env = {**os.environ, 'PYTHONHASHSEED': '42', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    outputs = []
    for name in ('strict_a', 'strict_b'):
        strict = subprocess.run([
            sys.executable, str(root / 'scripts/next_steps.py'), 'smoke',
            '--device', 'cpu', '--train-patches', '2', '--val-patches', '2',
            '--batch-size', '2', '--output-root', name,
        ], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
        assert strict.returncode == 0, strict.stdout + strict.stderr
        outputs.append(next((tmp_path / name).glob('*/summary.json')).parent)
    comparison = subprocess.run([
        sys.executable, str(root / 'scripts/next_steps.py'), 'compare',
        *map(str, outputs),
    ], cwd=tmp_path, text=True, capture_output=True, timeout=120)
    assert comparison.returncode == 0, comparison.stdout + comparison.stderr
    assert comparison.stdout.count('EXACT MATCH') == 3
