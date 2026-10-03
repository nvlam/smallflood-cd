import hashlib
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from smallflood_cd.engine.checkpointing import load_model_state, save_checkpoint
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.baselines.bit import BITBaseline
from smallflood_cd.models.baselines.bit_sar_v2 import BITSARV2
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config

REFERENCE = Path(__file__).resolve().parents[1] / 'reference/bit_upstream'
HASHES = {
    'help_funcs.py': 'ab6ae7776652435f11ad91a2bbdb374a8dc3c21c1f8a380ac5a7bd6c9f7af084',
    'networks.py': 'd4dc9a1551d25efda4d66492ca1958054bb72e4a19dd2d049440bc4d1bea055e',
    'resnet.py': 'c306949839ae411bfe472904fd06ad7bab117f19953f84c352466df233e62de2',
}


@pytest.fixture(autouse=True)
def small_cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(threads)


def _read_reference(name):
    raw = (REFERENCE / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[name]
    module = types.ModuleType('reference_' + name[:-3])
    exec(compile(raw, str(REFERENCE / name), 'exec'), module.__dict__)
    return module


def _official_reference(monkeypatch):
    """Only compatibility shims: imports, no downloads, SAR stem, fixed weights later."""
    legacy = types.ModuleType('torchvision.models.utils')
    def forbidden_download(*args, **kwargs):
        raise AssertionError('Reference test must never download weights')
    legacy.load_state_dict_from_url = forbidden_download
    monkeypatch.setitem(sys.modules, 'torchvision.models.utils', legacy)
    backbone = _read_reference('resnet.py')
    helpers = _read_reference('help_funcs.py')
    namespace = types.ModuleType('models')
    namespace.resnet18 = lambda pretrained=True, **kw: backbone.resnet18(pretrained=False, **kw)
    monkeypatch.setitem(sys.modules, 'models', namespace)
    monkeypatch.setitem(sys.modules, 'models.help_funcs', helpers)
    reference = _read_reference('networks.py').BASE_Transformer(
        input_nc=2, output_nc=2, with_pos='learned', resnet_stages_num=4,
        token_len=4, enc_depth=1, dec_depth=8, dim_head=64, decoder_dim_head=64)
    reference.resnet.conv1 = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
    return reference


def test_registered_version_does_not_replace_old_bit():
    config = load_experiment_config('configs/experiment/bit_sar_v2.yaml')
    assert isinstance(build_model(config['model']), BITSARV2)
    assert isinstance(build_model({'name': 'bit'}), BITBaseline)
    assert config['loss']['use_size_aware'] is False


def test_structure_and_full_resolution_classifier():
    model = BITSARV2().eval()
    assert len(model.core.transformer.layers) == 1
    assert len(model.core.transformer_decoder.layers) == 8
    assert model.core.pos_embedding.shape == (1, 8, 32)
    assert isinstance(model.core.resnet.layer4, nn.Identity)
    assert isinstance(model.core.resnet.fc, nn.Identity)
    x = torch.randn(1, 2, 256, 256)
    seen = []
    handle = model.core.classifier.register_forward_pre_hook(
        lambda module, args: seen.append(tuple(args[0].shape)))
    with torch.no_grad():
        features = model.core.forward_single(x)
        out = model(x, x + 0.1)
    handle.remove()
    assert features.shape == (1, 32, 64, 64)
    assert seen == [(1, 32, 256, 256)]
    assert out.change_logits.shape == (1, 1, 256, 256)
    assert out.boundary_logits is None
    assert not hasattr(model.core, 'tokens')
    assert not hasattr(model.core, 'tokens_')


def test_decoder_preserves_pixel_residual():
    decoder = BITSARV2().core.transformer_decoder
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
    pixels, tokens = torch.randn(1, 17, 32), torch.randn(1, 4, 32)
    torch.testing.assert_close(decoder(pixels, tokens), pixels, rtol=0, atol=0)


@pytest.mark.parametrize('size', [32, 64, 256])
def test_matches_pinned_official_forward(monkeypatch, size):
    model = BITSARV2().eval()
    reference = _official_reference(monkeypatch).eval()
    # Original retains unused layer4/fc; verify these are the ONLY absent keys.
    result = reference.load_state_dict(model.core.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(k.startswith(('resnet.layer4.', 'resnet.fc.')) for k in result.missing_keys)
    pre, post = torch.randn(1, 2, size, size), torch.randn(1, 2, size, size)
    with torch.no_grad():
        expected = reference(pre, post)
        actual = model.forward_two_class(pre, post)
        binary = model(pre, post).change_logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(binary.sigmoid(), expected.softmax(1)[:, 1:2],
                               atol=1e-6, rtol=1e-5)


def test_official_gradient_parity(monkeypatch):
    model = BITSARV2().eval()
    reference = _official_reference(monkeypatch).eval()
    reference.load_state_dict(model.core.state_dict(), strict=False)
    pre, post = torch.randn(1, 2, 32, 32), torch.randn(1, 2, 32, 32)
    target = torch.randint(0, 2, (1, 32, 32))
    torch.nn.functional.cross_entropy(model.forward_two_class(pre, post), target).backward()
    torch.nn.functional.cross_entropy(reference(pre, post), target).backward()
    upstream = dict(reference.named_parameters())
    for name, parameter in model.core.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, upstream[name].grad, atol=1e-6, rtol=1e-5)


def test_loss_optimizer_checkpoint_roundtrip(tmp_path):
    model = BITSARV2().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = torch.randn(2, 2, 32, 32), torch.randn(2, 2, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:12, 9:13] = 1
    valid = torch.ones_like(target)
    before = model.core.resnet.conv1.weight.detach().clone()
    output = model(x, y)
    loss = SmallFloodLoss(0.5, 0.5, 0.0)(output, target, valid, valid, target, valid).total
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
    assert model.core.pos_embedding.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.core.resnet.conv1.weight)
    path = tmp_path / 'bit_sar_v2.ckpt'
    save_checkpoint(path, model, optimizer, None, 0, 1, 0.0, {'model': {'name': 'bit_sar_v2'}})
    clone = BITSARV2().eval()
    checkpoint = load_model_state(path, clone, torch.device('cpu'))
    assert checkpoint['config']['model']['name'] == 'bit_sar_v2'
    restored_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    assert len(restored_optimizer.state) == len(optimizer.state)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, y).change_logits, clone(x, y).change_logits,
                                   rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [{'pretrained': True}, {'input_channels': 3}])
def test_unsupported_initialization_or_channels_fail(kwargs):
    with pytest.raises(ValueError):
        BITSARV2(**kwargs)


def test_invalid_input_fails():
    model = BITSARV2()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, 33, 32), torch.zeros(1, 2, 33, 32))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, 32, 32), torch.zeros(1, 2, 64, 64))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, 32, 32), torch.zeros(1, 2, 32, 32), torch.ones(1))
