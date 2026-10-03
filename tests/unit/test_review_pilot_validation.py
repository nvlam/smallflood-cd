import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from smallflood_cd.models.registry import build_model


ROOT = Path(__file__).parents[2]


def load_script():
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        spec = importlib.util.spec_from_file_location('review_test', ROOT / 'scripts/review_pilot_validation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_counts_mask_and_undefined_policy():
    module = load_script()
    p = np.array([True, True, False, False, True])
    t = np.array([True, False, True, False, False])
    v = np.array([True, True, True, True, False])
    counts = module.confusion(p, t, v)
    assert counts.tolist() == [1, 1, 1, 1]
    values = module.scores(counts)
    assert values['f1'] == 0.5
    assert values['iou'] == pytest.approx(1 / 3)
    empty_positive = module.scores([0, 0, 0, 10])
    assert empty_positive['f1'] is None
    assert empty_positive['accuracy'] == 1
    assert module.scores([0, 2, 0, 8])['f1'] == 0
    assert module.scores([0, 0, 2, 8])['recall'] == 0


@pytest.mark.parametrize('model_name', ['fc_siam_diff', 'bit_sar_v2', 'proposed_off', 'proposed_on'])
def test_review_reads_validation_only_and_preserves_checkpoints(tmp_path, model_name):
    run = tmp_path / 'pilot' / f'example_{model_name}'
    (run / 'checkpoints').mkdir(parents=True)
    rows = []
    for i in range(2):
        row = {'patch_id': f'p{i}', 'event_id': 'event', 'split': 'validation'}
        target = np.zeros((1, 32, 32), dtype='float32')
        target[:, 10:14, 10:14] = i
        for field, array in {'pre': np.zeros((2, 32, 32), dtype='float32'),
                             'post': np.ones((2, 32, 32), dtype='float32'),
                             'label': target, 'valid_mask': np.ones_like(target)}.items():
            path = tmp_path / f'{i}_{field}.npy'
            np.save(path, array)
            row[field + '_path'] = str(path)
        rows.append(row)
    # These files deliberately do not exist: reading train/test would fail.
    for split in ('train', 'test'):
        rows.append({**rows[0], 'split': split, 'patch_id': split,
                     **{k: '/not-present.npy' for k in rows[0] if k.endswith('_path')}})
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {'base': {'seed': 42}, 'data': {'manifest': str(manifest)},
              'model': {'name': 'fc_siam_diff', 'input_channels': 2, 'widths': [2, 4, 8, 16]}}
    if model_name == 'bit_sar_v2':
        config['model'] = {'name': model_name, 'input_channels': 2, 'pretrained': False}
    if model_name.startswith('proposed_'):
        config['model'] = {'name': 'smallflood_cdnet', 'input_channels': 2,
                           'pretrained': False, 'boundary_head': model_name=='proposed_on'}
        run = run.rename(run.parent/'example_proposed')
    model = build_model(config['model'])
    checkpoints = []
    for name in ('best_composite.ckpt', 'last.ckpt'):
        path = run / 'checkpoints' / name
        torch.save({'config': config, 'model': model.state_dict(), 'epoch': 1}, path)
        checkpoints.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    (run / 'environment.json').write_text(json.dumps({
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'validation_patches': 2}))
    stats = tmp_path / 'component_stats.json'
    stats.write_text(json.dumps({'fitted_split': 'train', 'small_area_threshold': 35}))
    config['data']['component_stats'] = str(stats)
    checkpoints = []
    for name in ('best_composite.ckpt', 'last.ckpt'):
        path = run / 'checkpoints' / name
        torch.save({'config': config, 'model': model.state_dict(), 'epoch': 1}, path)
        checkpoints.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    result = subprocess.run([
        sys.executable, str(ROOT / 'scripts/bit_sar_v2_entry.py'), 'review',
        '--pilot-root', str(run.parent), '--models', 'proposed' if model_name.startswith('proposed_') else model_name, '--device', 'cpu',
        '--output-root', str(tmp_path / 'out'),
        '--object-boundary',
    ], env={**os.environ, 'PYTHONHASHSEED': '42', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = list((tmp_path / 'out').glob('*/*/*/report.json'))
    assert len(reports) == 2
    for path in reports:
        report = json.loads(path.read_text())
        assert report['test_used'] is False
        assert report['global_pixel']['valid_pixels'] == 2048
        assert report['negative_patches'] == 1
        assert len(list((path.parent / 'images').glob('*.png'))) == 2
        detail = json.loads((path.parent / 'object_boundary_report.json').read_text())
        assert detail['counts']['patches'] == 2
        assert detail['counts']['component_target'] == 1
        assert detail['small_threshold_pixels'] == 35
    for path, digest in checkpoints:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
