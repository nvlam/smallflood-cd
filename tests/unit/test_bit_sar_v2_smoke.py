from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('smoke_bit_sar_v2', SCRIPTS / 'smoke_bit_sar_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_rejects_self(smoke, tmp_path):
    with pytest.raises(ValueError, match='different'):
        smoke.compare(tmp_path, tmp_path)


def test_missing_gradient_rejected(smoke):
    with pytest.raises(RuntimeError, match='gradient'):
        smoke.check_gradients(torch.nn.Linear(2, 1))


def test_state_normalization_preserves_exact_values_and_types(smoke):
    original = {'state': {0: {'step': torch.tensor(8.),
                              'exp_avg': torch.tensor([0.125], dtype=torch.float64)}},
                'groups': [({'lr': 1e-4},)], 'enabled': False}
    normalized = smoke.state_on_cpu(original)
    smoke.exact_equal(original, normalized)
    assert isinstance(normalized['groups'][0], tuple)
    assert normalized['state'][0]['exp_avg'].dtype == torch.float64
    changed = deepcopy(normalized)
    changed['state'][0]['exp_avg'] += 1e-10
    with pytest.raises(AssertionError):
        smoke.exact_equal(normalized, changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA regression requires server GPU')
def test_mixed_cpu_cuda_optimizer_state_comparison(smoke):
    left = {'state': {0: {'step': torch.tensor(8.),
                          'exp_avg': torch.tensor([0.125], device='cuda')}}}
    right = {'state': {0: {'step': torch.tensor(8., device='cuda'),
                           'exp_avg': torch.tensor([0.125])}}}
    smoke.exact_equal(smoke.state_on_cpu(left), smoke.state_on_cpu(right))
    right['state'][0]['step'].add_(1)
    with pytest.raises(AssertionError):
        smoke.exact_equal(smoke.state_on_cpu(left), smoke.state_on_cpu(right))


@pytest.mark.parametrize('batch_size', [2, 8])
def test_two_cpu_smokes_and_corruption_detection(smoke, monkeypatch, tmp_path, batch_size):
    """Synthetic fixtures only: exercise real model/trainer without server data or CUDA."""
    monkeypatch.setenv('PYTHONHASHSEED', '42')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    class TinyDataset:
        def __init__(self, split):
            self.records = [SimpleNamespace(patch_id=f'{split}_{i}') for i in range(16)]
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            generator = torch.Generator().manual_seed(i + 100)
            target = torch.zeros(1, 32, 32)
            if i % 2:
                target[:, 8:12, 8:12] = 1
            return {'pre': torch.randn(2, 32, 32, generator=generator),
                    'post': torch.randn(2, 32, 32, generator=generator),
                    'target': target, 'valid_mask': torch.ones_like(target), 'event_id': 'synthetic'}
    calls = []
    def dataset(config, split, loss):
        assert split in ('train', 'validation')
        calls.append(split)
        return TinyDataset(split), None
    monkeypatch.setattr(smoke, '_dataset', dataset)
    config = smoke.load_experiment_config('configs/experiment/bit_sar_v2.yaml')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('patch_id,flood_pixels\n' + ''.join(
        f'{split}_{i},{16 if i % 2 else 0}\n' for split in ('train', 'validation') for i in range(16)))
    config['data']['manifest'] = str(manifest)
    metadata = tmp_path / 'normalization.json'
    metadata.write_text(json.dumps({'pre_bands_1based': [5,6], 'post_bands_1based': [7,8],
        'fitted_split': 'train', 'shared_pre_post': True, 'units': 'dB', 'already_applied_to_npy': True}))
    config['data']['normalization_stats'] = str(metadata)
    a, b = tmp_path / 'a', tmp_path / 'b'
    try:
        for path in (a, b):
            smoke.run_smoke(path, config=deepcopy(config), device='cpu', batch_size=batch_size)
        smoke.compare(a, b)
        assert calls == ['train', 'validation', 'train', 'validation']
        assert json.loads((a / 'checks.json').read_text())['optimizer_steps'] == 16 // batch_size
        with pytest.raises(FileExistsError):
            smoke.run_smoke(a, config=deepcopy(config), device='cpu')
        original = (b / 'checks.json').read_text()
        (b / 'checks.json').write_text(original.replace('"PASS"', '"FAIL"'))
        with pytest.raises(AssertionError):
            smoke.compare(a, b)
        (b / 'checks.json').write_text(original)
        checkpoint_path = b / 'checkpoints/last.ckpt'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint['global_step'] += 1
        torch.save(checkpoint, checkpoint_path)
        with pytest.raises(AssertionError):
            smoke.compare(a, b)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
