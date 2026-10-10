import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smallflood_cd.models.registry import build_model

ROOT = Path(__file__).parents[2]


@pytest.fixture
def eff(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    monkeypatch.syspath_prepend(str(ROOT/'src'))
    monkeypatch.setenv('PYTHONHASHSEED', '42')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    spec = importlib.util.spec_from_file_location('efficiency_t', ROOT/'scripts/efficiency.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny():
    return build_model({'name': 'fc_siam_diff', 'input_channels': 2, 'widths': [2, 4, 8, 16]}).eval()


def test_architecture_table_matches_protocol(eff):
    assert {k: v[1] for k, v in eff.ARCHITECTURES.items()} == {
        'smallflood_cdnet': 1079865, 'smallflood_cdnet_boundary': 1080805,
        'fc_siam_diff': 487857, 'bit_sar_v2': 3492642}
    assert eff.GPU_TIMING == {'warmup': 100, 'timed': 500, 'repetitions': 3}
    assert eff.CPU_TIMING == {'warmup': 20, 'timed': 100, 'repetitions': 3}
    assert eff.CPU_THREADS == (1, 4) and eff.ONNX_PATCHES == 64


def test_flops_sizes_and_parameter_bytes(eff, tmp_path):
    model = tiny()
    counted = eff.count_flops(model)
    with torch.inference_mode():  # the smoke stage calls it inside inference_mode
        assert eff.count_flops(model) == counted
    assert counted['flops'] > 0 and counted['macs'] == counted['flops'] // 2
    assert 'convolutions' in counted['flop_scope']
    sizes = eff.serialized_sizes(model, tmp_path/'s', 'tiny')
    assert sizes['state_dict_fp16_bytes'] < sizes['state_dict_fp32_bytes']
    assert not list((tmp_path/'s').glob('*.pt'))
    parameters = sum(p.numel() for p in model.parameters())
    assert eff.parameter_bytes(model) >= 4*parameters
    assert eff.example(1)[0].shape == (1, 2, 256, 256)
    assert torch.equal(eff.example(1)[0], eff.example(1)[0])
    assert eff.example(8, half=True)[1].dtype == torch.float16


def test_latency_summary_and_gpu_process_filter(eff):
    s = eff.summarize_latency([[1, 2, 3, 4, 100], [2, 2, 2, 2, 2], [5, 5, 5, 5, 5]])
    assert s['repetition_medians_ms'] == [3, 2, 5] and s['median_ms'] == 3
    assert s['p90_ms'] == pytest.approx(5.0)
    assert eff.other_gpu_processes('123\n456\n', 123) == [456]
    assert eff.other_gpu_processes('\n', 1) == []


def test_agreement_and_metrics_from_predictions(eff):
    a = np.array([[[[2.0, -1.0], [0.5, -3.0]]]])
    b = np.array([[[[1.5, 0.2], [0.4, -3.0]]]])
    valid = np.array([[[[True, True], [True, False]]]])
    assert eff.agreement(a, b, valid) == (2, 3, pytest.approx(1.2))
    truth = np.zeros((32, 32), bool); truth[8:16, 8:16] = True
    valid = np.ones_like(truth)
    perfect = eff.metrics_from_predictions([truth], [truth], [valid], ['e'], 35)
    assert perfect['event_macro_f1'] == 100 and perfect['f1'] == 100
    assert perfect['object_global_component_f1'] == 100
    empty = eff.metrics_from_predictions([np.zeros_like(truth)], [truth], [valid], ['e'], 35)
    assert empty['f1'] == 0 and empty['object_global_component_f1'] in (0, None)


def test_fp16_edits_are_the_only_changes(eff):
    import difflib
    import inspect
    source = inspect.getsource(eff.rpv.evaluate)
    edited = source
    for old, new, count in eff.FP16_EDITS:
        assert source.count(old) == count
        edited = edited.replace(old, new)
    changed = [x for x in difflib.ndiff(source.splitlines(), edited.splitlines()) if x[:2] in ('- ', '+ ')]
    # Signature and forward line replaced (2 removed, 2 added) plus one added half() line.
    assert sum(x.startswith('- ') for x in changed) == 2
    assert sum(x.startswith('+ ') for x in changed) == 3


def synthetic_run(tmp_path):
    rows = []
    for i in range(2):
        row = {'patch_id': f'v{i}', 'event_id': 'val_event', 'split': 'validation'}
        target = np.zeros((1, 32, 32), dtype='float32'); target[:, 10:14, 10:14] = i
        for field, array in {'pre': np.full((2, 32, 32), i, dtype='float32'),
                             'post': np.ones((2, 32, 32), dtype='float32'),
                             'label': target, 'valid_mask': np.ones_like(target)}.items():
            path = tmp_path/f'{i}_{field}.npy'
            np.save(path, array)
            row[field + '_path'] = str(path)
        rows.append(row)
    rows.append({**rows[0], 'split': 'test', 'patch_id': 'test',
                 **{k: '/not-present.npy' for k in rows[0] if k.endswith('_path')}})
    manifest = tmp_path/'manifest.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stats = tmp_path/'stats.json'
    stats.write_text(json.dumps({'fitted_split': 'train', 'small_area_threshold': 35}))
    config = {'base': {'seed': 42}, 'data': {'manifest': str(manifest), 'component_stats': str(stats)},
              'model': {'name': 'fc_siam_diff', 'input_channels': 2, 'widths': [2, 4, 8, 16]}}
    run = tmp_path/'run_fc_siam_diff'
    (run/'checkpoints').mkdir(parents=True)
    torch.manual_seed(0)
    torch.save({'config': config, 'model': build_model(config['model']).state_dict(), 'epoch': 14},
               run/'checkpoints'/'last.ckpt')
    (run/'environment.json').write_text(json.dumps({
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'validation_patches': 2}))
    return run


def test_fp32_reproduction_check_and_validation_only_loader(eff, tmp_path):
    run = synthetic_run(tmp_path)
    args = SimpleNamespace(device='cpu', batch_size=8, object_boundary=True)
    eff.rpv.evaluate(run, 'last.ckpt', tmp_path/'a', args)
    eff.rpv.evaluate(run, 'last.ckpt', tmp_path/'b', args)
    eff.check_fp32_reproduces(tmp_path/'a'/run.name/'last', tmp_path/'b'/run.name/'last')
    report = tmp_path/'b'/run.name/'last'/'report.json'
    changed = json.loads(report.read_text()); changed['global_pixel']['tp'] += 1
    report.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='does not reproduce'):
        eff.check_fp32_reproduces(tmp_path/'a'/run.name/'last', tmp_path/'b'/run.name/'last')
    dataset, loader, small = eff.validation_dataset(run)
    assert len(dataset) == 2 and small == 35
    assert {r.split for r in dataset.records} == {'validation'}


def test_complete_requires_all_outputs(eff, tmp_path):
    with (tmp_path/'costs.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=['architecture']); w.writeheader()
        w.writerows([{'architecture': 'x'}]*16)
    (tmp_path/'fidelity.json').write_text(json.dumps({'architectures': dict.fromkeys(eff.ARCHITECTURES, {}),
                                                      'test_used': False}))
    eff.complete(tmp_path)
    assert json.loads((tmp_path/'COMPLETE.json').read_text())['cost_rows'] == 16
    (tmp_path/'fidelity.json').write_text(json.dumps({'architectures': {}, 'test_used': False}))
    with pytest.raises(RuntimeError, match='Incomplete'):
        eff.complete(tmp_path)


def test_export_restores_eval_and_training_models_are_refused(eff, tmp_path):
    pytest.importorskip('onnx')  # present on the server; the local venv has no onnx
    model = tiny()
    eff.require_eval(model)
    pre, post = eff.example(1)
    with torch.no_grad():
        before = model(pre, post).change_logits.clone()
    eff.export_eval(model, tmp_path/'tiny.onnx')
    assert not model.training and not any(m.training for m in model.modules())
    with torch.no_grad():
        assert torch.equal(model(pre, post).change_logits, before)
    with pytest.raises(RuntimeError, match='training mode'):
        eff.require_eval(model.train())
