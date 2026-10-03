import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('pilot_v2_test', SCRIPTS / 'pilot_bit_sar_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate(pilot, monkeypatch, tmp_path):
    checks = {'status': 'PASS', 'model': 'bit_sar_v2', 'batch_size': 8,
              'optimizer_steps': 2, 'checkpoint_reload_exact': True,
              'test_used': False, 'parameter_count': 3492642}
    (tmp_path / 'checks.json').write_text(json.dumps(checks))
    (tmp_path / 'COMPLETE.json').write_text(json.dumps({'status': 'PASS', 'test_used': False}))
    (tmp_path / 'environment.json').write_text(json.dumps({'device': 'cuda'}))
    (tmp_path / 'batch8_gate.json').write_text(json.dumps({'fingerprint': 'same'}))
    monkeypatch.setattr(pilot, 'fingerprint', lambda: {'fingerprint': 'same'})
    return tmp_path


def test_valid_gate(pilot, gate):
    assert pilot.validate_gate(gate) == {'fingerprint': 'same'}


@pytest.mark.parametrize('key,value', [('batch_size', 2), ('optimizer_steps', 8),
                                       ('status', 'FAIL'), ('test_used', True)])
def test_invalid_gate_rejected(pilot, gate, key, value):
    path = gate / 'checks.json'
    checks = json.loads(path.read_text())
    checks[key] = value
    path.write_text(json.dumps(checks))
    with pytest.raises(ValueError):
        pilot.validate_gate(gate)


def test_changed_source_rejected(pilot, gate, monkeypatch):
    monkeypatch.setattr(pilot, 'fingerprint', lambda: {'fingerprint': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        pilot.validate_gate(gate)


def test_cpu_gate_rejected(pilot, gate):
    (gate / 'environment.json').write_text(json.dumps({'device': 'cpu'}))
    with pytest.raises(ValueError, match='CUDA'):
        pilot.validate_gate(gate)


@pytest.mark.parametrize('fail', [False, True])
def test_fixed_pilot_arguments_and_no_persistent_registration(pilot, gate, monkeypatch, fail):
    old = dict(pilot.smoke_train.CONFIGS)
    calls = []
    def fake_run(args):
        calls.append(args)
        assert pilot.smoke_train.CONFIGS['bit_sar_v2'] == pilot.CONFIG
        if fail:
            raise RuntimeError('simulated training failure')
    monkeypatch.setattr(pilot.next_steps, 'pilot', fake_run)
    if fail:
        with pytest.raises(RuntimeError, match='simulated'):
            pilot.run_pilot(gate, gate / 'output')
    else:
        pilot.run_pilot(gate, gate / 'output')
    assert len(calls) == 1
    args = calls[0]
    assert (args.model, args.epochs, args.batch_size, args.seed, args.engineering_only) == (
        'bit_sar_v2', 15, 8, 42, True)
    assert pilot.smoke_train.CONFIGS == old


def test_missing_gate_blocks_training(pilot, gate, monkeypatch):
    (gate / 'batch8_gate.json').unlink()
    monkeypatch.setattr(pilot.next_steps, 'pilot', lambda args: pytest.fail('Must not train'))
    with pytest.raises(FileNotFoundError):
        pilot.run_pilot(gate, gate / 'output')
