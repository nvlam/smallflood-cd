import importlib.util
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import pytest
from smallflood_cd.data.connected_components import label_connected_components

spec = importlib.util.spec_from_file_location('mask_audit', Path(__file__).parents[2] / 'scripts/audit_proposed_train_masks.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_invalid_bridge_and_weights():
    y = np.zeros((9, 9), dtype=np.float32)
    y[4, 2:7] = 1
    v = np.ones_like(y); v[4, 4] = 0
    c, h, mh = audit.inspect(y, v)
    assert h == {5: 1} and mh == {2: 2}
    assert c['invalid_foreground'] == 1
    assert c['valid_foreground_weight_changed_by_mask'] == 4
    assert c['boundary_supervised_near_invalid_r1'] > 0


def test_all_valid_and_all_invalid():
    y = np.zeros((9, 9), dtype=np.float32); y[3:6, 3:6] = 1
    c, _, _ = audit.inspect(y, np.ones_like(y))
    assert c['invalid_foreground'] == c['boundary_supervised_near_invalid_r1'] == 0
    c, _, mh = audit.inspect(y, np.zeros_like(y))
    assert c['boundary_supervised'] == c['valid_foreground_weight_mass'] == 0
    assert not mh


def test_label_histogram_matches_cache_algorithm():
    y = np.random.default_rng(42).random((32, 32)) > .8
    _, h, _ = audit.inspect(y, np.ones_like(y))
    labels = label_connected_components(y, 8)
    from collections import Counter
    assert h == Counter(map(int, np.bincount(labels.ravel())[1:]))
    with pytest.raises(ValueError):
        audit.inspect(y.astype(float) * float('nan'), np.ones_like(y))


def test_full_scan_reads_train_masks_only_and_preserves_files(tmp_path):
    y = np.zeros((1, 256, 256), dtype=np.float32); y[:, 10:12, 10:12] = 1
    label, valid = tmp_path/'label.npy', tmp_path/'valid.npy'
    np.save(label, y); np.save(valid, np.ones_like(y))
    manifest = tmp_path/'manifest.csv'
    fields = ['patch_id','event_id','split','pre_path','post_path','label_path','valid_mask_path']
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for split in ['train','validation','test']:
            writer.writerow(dict(patch_id=split,event_id='event',split=split,
                pre_path='/DO_NOT_READ.npy',post_path='/DO_NOT_READ.npy',
                label_path=str(label) if split=='train' else '/DO_NOT_READ.npy',
                valid_mask_path=str(valid) if split=='train' else '/DO_NOT_READ.npy'))
    stats=tmp_path/'stats.json'
    stats.write_text(json.dumps(dict(fitted_split='train',connectivity=8,reference_area=65,
                                    small_area_threshold=35,area_histogram={'4':1})))
    paths=[label,valid,manifest,stats]
    hashes=[audit.sha(p) for p in paths]
    audit.run(manifest,stats,tmp_path/'output')
    r=json.loads((tmp_path/'output/report.json').read_text())
    assert r['target_histogram_matches_stats']
    assert r['counts']['patches']==1 and not r['test_used']
    assert hashes==[audit.sha(p) for p in paths]
    with pytest.raises(FileExistsError):
        audit.run(manifest,stats,tmp_path/'output')
