"""efficiency_v1 (A3): cost set for the four benchmark architectures.

docs/efficiency_protocol_v1_20261010.md. Measures parameters, counted FLOPs/MACs, serialized
size, peak GPU memory, PyTorch FP32/FP16 latency and throughput on the server GPU and ONNX
Runtime FP32 CPU latency; checks FP16 and ONNX fidelity on the validation split only.
Never trains and never reads the test split.

Stages:
  test                       synthetic CPU tests
  smoke    --output DIR      engineering check of export/FP16 paths (no timing kept)
  measure  --output ROOT     costs.csv, latency_raw/, onnx/, environment.json
  fidelity --output ROOT     fidelity.json (run with PYTHONHASHSEED=201)
  complete --output ROOT     COMPLETE.json after both stages
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from bit_sar_v2_entry import pin_source
import review_pilot_validation as rpv
from review_pilot_validation import sha256, write_csv
from smallflood_cd.deployment.onnx_export import export_onnx
from smallflood_cd.metrics.object_boundary_v2 import add_counts, patch_counts, summarize
from smallflood_cd.models.registry import build_model

PROTOCOL_ID = 'efficiency_v1'
RUN = Path('runs/recipe_confirmation_v1/20261007T234436Z')
REVIEW = Path('artifacts/recipe_confirmation_review/20261007T234436Z/s201')
SEED = 201
# architecture -> (arms, parameters, run directory glob under RUN)
ARCHITECTURES = {
    'smallflood_cdnet': ('A, B', 1079865, 'factorial_s201/A_proposed'),
    'smallflood_cdnet_boundary': ('C, D', 1080805, 'factorial_s201/C_proposed'),
    'fc_siam_diff': ('FC', 487857, 'pilots_s201/*_fc_siam_diff'),
    'bit_sar_v2': ('BIT', 3492642, 'pilots_s201/*_bit_sar_v2'),
}
INPUT_SHAPE = (2, 256, 256)
GPU_TIMING = {'warmup': 100, 'timed': 500, 'repetitions': 3}
CPU_TIMING = {'warmup': 20, 'timed': 100, 'repetitions': 3}
THROUGHPUT = {'batch': 8, 'warmup': 20, 'timed': 100, 'repetitions': 3}
CPU_THREADS = (1, 4)
GPU_WAIT_MINUTES = 30
SIZE_BUDGET_MB = 20.0
ONNX_PATCHES = 64
METRICS = ('event_macro_f1', 'f1', 'object_global_component_f1',
           'object_global_small_component_recall_interior', 'object_global_boundary_f1',
           'object_global_boundary_inner_band_iou')


def run_directory(architecture):
    matches = sorted(RUN.glob(ARCHITECTURES[architecture][2]))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one run directory for {architecture}')
    return matches[0]


def load_model(architecture):
    """Model built from the checkpoint's own config, with the reviewed weights."""
    directory = run_directory(architecture)
    path = directory/'checkpoints'/'last.ckpt'
    reference = json.loads((REVIEW/directory.name/'last'/'report.json').read_text())
    if sha256(path) != reference['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint differs from its validation record: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(dict(checkpoint['config']['model'], pretrained=False)).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != ARCHITECTURES[architecture][1]:
        raise RuntimeError(f'Parameter count mismatch for {architecture}: {count}')
    return model, directory


def export_eval(model, path):
    """export_onnx leaves the wrapped model in training mode (torch.onnx restores the fresh
    wrapper's default mode); the exported graph itself is in eval mode. Restore eval here."""
    path = export_onnx(model, path)
    model.eval()
    return path


def require_eval(model):
    if model.training or any(m.training for m in model.modules()):
        raise RuntimeError('Model is in training mode; refuse to measure')
    return model


def inference_settings():
    """Timing uses inference settings, not the training determinism settings."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def example(batch, device='cpu', half=False):
    generator = torch.Generator().manual_seed(0)
    pre, post = (torch.randn(batch, *INPUT_SHAPE, generator=generator) for _ in range(2))
    if half:
        pre, post = pre.half(), post.half()
    return pre.to(device), post.to(device)


def count_flops(model):
    from torch.utils.flop_counter import FlopCounterMode
    # FlopCounterMode counts nothing under inference_mode on torch 2.5; use no_grad instead.
    with torch.inference_mode(False), torch.no_grad():
        pre, post = example(1)
        with FlopCounterMode(display=False) as counter:
            model(pre, post)
    flops = int(counter.get_total_flops())
    if flops <= 0:
        raise RuntimeError('FLOP counter returned zero')
    return {'flops': flops, 'macs': flops // 2,
            'flop_scope': 'convolutions and matrix multiplications only (FlopCounterMode)'}


def state_bytes(model, half):
    state = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    return state


def serialized_sizes(model, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for label, half in (('fp32', False), ('fp16', True)):
        path = directory/f'{name}_{label}.pt'
        torch.save(state_bytes(model, half), path)
        sizes[f'state_dict_{label}_bytes'] = path.stat().st_size
        path.unlink()
    return sizes


def parameter_bytes(model):
    return (sum(p.numel()*p.element_size() for p in model.parameters())
            + sum(b.numel()*b.element_size() for b in model.buffers()))


def summarize_latency(repetitions):
    """Median and 90th percentile per repetition; reported value = median of medians."""
    medians = [statistics.median(r) for r in repetitions]
    p90 = [float(np.percentile(r, 90)) for r in repetitions]
    return {'median_ms': statistics.median(medians), 'p90_ms': statistics.median(p90),
            'repetition_medians_ms': medians, 'repetition_p90_ms': p90}


def other_gpu_processes(output, own_pid):
    return [int(x) for x in output.split() if x.strip().isdigit() and int(x) != own_pid]


def wait_for_free_gpu(log):
    """Protocol section 6: no other compute process; wait up to 30 minutes, then stop."""
    for minute in range(GPU_WAIT_MINUTES + 1):
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout
        others = other_gpu_processes(out, os.getpid())
        log.append({'time': time.time(), 'other_gpu_pids': others, 'loadavg': os.getloadavg()})
        if not others:
            return
        print(f'GPU busy (pids {others}); waiting, minute {minute}', flush=True)
        time.sleep(60)
    raise RuntimeError('GPU still shared after 30 minutes; stop and report')


@torch.inference_mode()
def gpu_latency(model, half, log):
    device = torch.device('cuda')
    model = require_eval(model.half() if half else model.float()).to(device)
    pre, post = example(1, device, half)
    repetitions = []
    for _ in range(GPU_TIMING['repetitions']):
        wait_for_free_gpu(log)
        for _ in range(GPU_TIMING['warmup']):
            model(pre, post)
        torch.cuda.synchronize()
        values = []
        for _ in range(GPU_TIMING['timed']):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(pre, post).change_logits
            end.record()
            torch.cuda.synchronize()
            values.append(start.elapsed_time(end))
        if not torch.isfinite(out).all():
            raise RuntimeError('Nonfinite output')
        repetitions.append(values)
    return repetitions


@torch.inference_mode()
def gpu_memory(model, half):
    device = torch.device('cuda')
    model = require_eval(model.half() if half else model.float()).to(device)
    pre, post = example(1, device, half)
    model(pre, post)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    model(pre, post)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    return {'peak_allocated_bytes': peak, 'parameter_bytes': parameter_bytes(model),
            'peak_minus_parameters_bytes': peak - parameter_bytes(model)}


@torch.inference_mode()
def gpu_throughput(model, half, log):
    device = torch.device('cuda')
    model = require_eval(model.half() if half else model.float()).to(device)
    pre, post = example(THROUGHPUT['batch'], device, half)
    rates = []
    for _ in range(THROUGHPUT['repetitions']):
        wait_for_free_gpu(log)
        for _ in range(THROUGHPUT['warmup']):
            model(pre, post)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for _ in range(THROUGHPUT['timed']):
            model(pre, post)
        torch.cuda.synchronize()
        rates.append(THROUGHPUT['batch']*THROUGHPUT['timed']/(time.perf_counter() - tick))
    return rates


def onnx_session(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def cpu_latency(path, threads, log):
    session = onnx_session(path, threads)
    pre, post = (x.numpy() for x in example(1))
    feed = {'pre': pre, 'post': post}
    repetitions = []
    for _ in range(CPU_TIMING['repetitions']):
        before = os.getloadavg()
        for _ in range(CPU_TIMING['warmup']):
            session.run(None, feed)
        values = []
        for _ in range(CPU_TIMING['timed']):
            tick = time.perf_counter_ns()
            out = session.run(None, feed)[0]
            values.append((time.perf_counter_ns() - tick)/1e6)
        if not np.isfinite(out).all():
            raise RuntimeError('Nonfinite ONNX output')
        log.append({'time': time.time(), 'threads': threads, 'loadavg_before': before,
                    'loadavg_after': os.getloadavg()})
        repetitions.append(values)
    return repetitions


def environment():
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    import onnx
    import onnxruntime as ort
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(), 'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
            'onnxruntime_providers': ort.get_available_providers(), 'python': sys.version,
            'nvidia_smi': run('nvidia-smi'), 'lscpu': run('lscpu'), 'loadavg': os.getloadavg(),
            'torch_threads': torch.get_num_threads()}


def measure(root):
    inference_settings()
    root.mkdir(parents=True, exist_ok=False)
    (root/'latency_raw').mkdir()
    pin = environment()
    rows, load_log = [], []
    for name, (arms, parameters, _) in ARCHITECTURES.items():
        model, directory = load_model(name)
        base = {'architecture': name, 'arms': arms, 'parameters': parameters, **count_flops(model),
                **serialized_sizes(model, root/'tmp', name)}
        onnx_path = export_eval(model, root/'onnx'/f'{name}.onnx')
        base['onnx_fp32_bytes'] = onnx_path.stat().st_size
        base['fits_20mb_budget_fp32'] = base['state_dict_fp32_bytes']/1e6 <= SIZE_BUDGET_MB
        base['fits_20mb_budget_fp16'] = base['state_dict_fp16_bytes']/1e6 <= SIZE_BUDGET_MB
        for precision in ('fp32', 'fp16'):
            half = precision == 'fp16'
            fresh, _ = load_model(name)
            raw = gpu_latency(fresh, half, load_log)
            rpv_dump(root/'latency_raw'/f'{name}_gpu_{precision}.json', raw)
            rates = gpu_throughput(fresh, half, load_log)
            rows.append({**base, 'device': 'gpu', 'backend': 'pytorch', 'precision': precision, 'threads': '',
                         **summarize_latency(raw), **gpu_memory(fresh, half),
                         'throughput_batch8_pairs_per_s': statistics.median(rates),
                         'throughput_repetitions': rates})
            del fresh
            torch.cuda.empty_cache()
            print(f'MEASURED {name} gpu {precision}', flush=True)
        for threads in CPU_THREADS:
            raw = cpu_latency(onnx_path, threads, load_log)
            rpv_dump(root/'latency_raw'/f'{name}_cpu_fp32_{threads}t.json', raw)
            rows.append({**base, 'device': 'cpu', 'backend': 'onnxruntime', 'precision': 'fp32',
                         'threads': threads, **summarize_latency(raw), 'peak_allocated_bytes': '',
                         'parameter_bytes': '', 'peak_minus_parameters_bytes': '',
                         'throughput_batch8_pairs_per_s': '', 'throughput_repetitions': ''})
            print(f'MEASURED {name} cpu {threads} thread(s)', flush=True)
        rpv_dump(root/'onnx'/f'{name}.sha256.json', {'sha256': sha256(onnx_path), 'run': str(directory)})
    (root/'tmp').rmdir()
    write_csv(root/'costs.csv', rows)
    rpv_dump(root/'environment.json', {**pin, 'load_log': load_log, 'gpu_timing': GPU_TIMING,
                                      'cpu_timing': CPU_TIMING, 'throughput': THROUGHPUT,
                                      'script_sha256': sha256(__file__)})
    print(f'EFFICIENCY MEASURE COMPLETE: {root}', flush=True)


def rpv_dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


# ---------- fidelity (validation only) ----------

# Edits to review_pilot_validation.evaluate() for FP16 inference: (old, new, expected count).
FP16_EDITS = (
    ('def evaluate(directory, checkpoint_name, output, args):',
     'def evaluate_fp16(directory, checkpoint_name, output, args):', 1),
    ("    model.load_state_dict(checkpoint['model'], strict=True)\n",
     "    model.load_state_dict(checkpoint['model'], strict=True)\n    model = model.half()\n", 1),
    ('logits = model(pre, post).change_logits',
     'logits = model(pre.half(), post.half()).change_logits.float()', 1),
)


def build_evaluate_fp16():
    source = inspect.getsource(rpv.evaluate)
    for old, new, count in FP16_EDITS:
        if source.count(old) != count:
            raise RuntimeError(f'evaluate() changed; expected {count} x {old!r}')
        source = source.replace(old, new)
    namespace = dict(vars(rpv))
    exec(compile(source, rpv.__file__, 'exec'), namespace)
    return namespace['evaluate_fp16']


evaluate_fp16 = build_evaluate_fp16()
COMPARED_REPORT = ('global_pixel', 'event_macro', 'event_macro_defined_counts', 'by_patch_category')
COMPARED_DETAIL = ('counts', 'global', 'events', 'event_macro')


def check_fp32_reproduces(new, old):
    for name, keys in (('report.json', COMPARED_REPORT), ('object_boundary_report.json', COMPARED_DETAIL)):
        a, b = json.loads((new/name).read_text()), json.loads((old/name).read_text())
        for key in keys:
            if a[key] != b[key]:
                raise RuntimeError(f'FP32 does not reproduce the validation report: {name}:{key}')


def validation_dataset(directory):
    from torch.utils.data import DataLoader
    from smallflood_cd.data.datasets import ManifestDataset
    from smallflood_cd.data.preprocessing import load_array
    config = torch.load(directory/'checkpoints'/'last.ckpt', map_location='cpu', weights_only=False)['config']
    dataset = ManifestDataset(Path(config['data']['manifest']), 'validation', load_array, use_uncertain_mask=False)
    dataset.records.sort(key=lambda r: (r.event_id, r.patch_id))
    if any(r.split != 'validation' for r in dataset.records):
        raise RuntimeError('Non-validation record in fidelity loader')
    stats = json.loads(Path(config['data'].get('component_stats', 'data/metadata/component_stats_v1.json')).read_text())
    return dataset, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0), int(stats['small_area_threshold'])


def agreement(a, b, valid):
    """Mask agreement over valid pixels and max |logit difference| (numpy arrays)."""
    pa, pb = a >= 0, b >= 0  # sigmoid(logit) >= 0.5
    return int(((pa == pb) & valid).sum()), int(valid.sum()), float(np.abs(a - b).max())


@torch.inference_mode()
def fp16_agreement(architecture):
    model32, directory = load_model(architecture)
    model16, _ = load_model(architecture)
    model32, model16 = require_eval(model32).cuda(), require_eval(model16).half().cuda()
    _, loader, _ = validation_dataset(directory)
    same = total = 0
    worst = 0.0
    for batch in loader:
        pre, post = batch['pre'].cuda(), batch['post'].cuda()
        a = model32(pre, post).change_logits.float().cpu().numpy()
        b = model16(pre.half(), post.half()).change_logits.float().cpu().numpy()
        s, t, w = agreement(a, b, batch['valid_mask'].numpy() > 0.5)
        same, total, worst = same + s, total + t, max(worst, w)
    return {'mask_agreement': same/total, 'disagreeing_pixels': total - same, 'valid_pixels': total,
            'max_abs_logit_difference': worst}


def metrics_from_predictions(predictions, targets, valids, events, small_threshold):
    """Review metrics (same functions as evaluate) from boolean masks, in percent."""
    by_event = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    object_events = defaultdict(dict)
    for p, t, v, e in zip(predictions, targets, valids, events):
        by_event[e] += rpv.confusion(p, t, v)
        add_counts(object_events[e], patch_counts(p, t, v, small_threshold))
    rows = {e: rpv.scores(c) for e, c in by_event.items()}
    f1s = [r['f1'] for r in rows.values() if r['f1'] is not None]
    detail = summarize(object_events)['global']
    pooled = rpv.scores(sum(by_event.values(), np.zeros(4, dtype=np.int64)))
    values = {'event_macro_f1': sum(f1s)/len(f1s) if f1s else None, 'f1': pooled['f1'],
              'object_global_component_f1': detail['component_f1'],
              'object_global_small_component_recall_interior': detail['small_component_recall_interior'],
              'object_global_boundary_f1': detail['boundary_f1'],
              'object_global_boundary_inner_band_iou': detail['boundary_inner_band_iou']}
    return {k: None if v is None else 100*v for k, v in values.items()}


@torch.inference_mode()
def onnx_agreement(architecture, onnx_path):
    model, directory = load_model(architecture)
    model = require_eval(model).cuda()
    dataset, _, small = validation_dataset(directory)
    session = onnx_session(onnx_path, 4)
    same = total = 0
    worst = 0.0
    torch_masks, onnx_masks, targets, valids, events = [], [], [], [], []
    for index in range(ONNX_PATCHES):
        item = dataset[index]
        pre, post = item['pre'][None].float(), item['post'][None].float()
        a = model(pre.cuda(), post.cuda()).change_logits.cpu().numpy()
        b = session.run(None, {'pre': pre.numpy(), 'post': post.numpy()})[0]
        valid = item['valid_mask'][None].numpy() > 0.5
        s, t, w = agreement(a, b, valid)
        same, total, worst = same + s, total + t, max(worst, w)
        torch_masks.append(a[0, 0] >= 0); onnx_masks.append(b[0, 0] >= 0)
        targets.append(item['target'][0].numpy() > 0.5); valids.append(valid[0, 0]); events.append(item['event_id'])
    m_torch = metrics_from_predictions(torch_masks, targets, valids, events, small)
    m_onnx = metrics_from_predictions(onnx_masks, targets, valids, events, small)
    return {'patches': ONNX_PATCHES, 'mask_agreement': same/total, 'disagreeing_pixels': total - same,
            'valid_pixels': total, 'max_abs_logit_difference': worst,
            'metrics_pytorch_fp32': m_torch, 'metrics_onnxruntime': m_onnx,
            'metric_differences_pp': {k: None if m_torch[k] is None or m_onnx[k] is None else m_onnx[k] - m_torch[k]
                                      for k in m_torch}}


def fidelity(root):
    if not (root/'costs.csv').is_file():
        raise RuntimeError('Run measure first')
    out = root/'fidelity'
    out.mkdir()
    args = SimpleNamespace(device='cuda', batch_size=8, object_boundary=True)
    result = {}
    for name in ARCHITECTURES:
        directory = run_directory(name)
        load_model(name)  # checkpoint identity
        r32 = rpv.evaluate(directory, 'last.ckpt', out/'fp32', args)
        check_fp32_reproduces(out/'fp32'/directory.name/'last', REVIEW/directory.name/'last')
        r16 = evaluate_fp16(directory, 'last.ckpt', out/'fp16', args)
        onnx_path = root/'onnx'/f'{name}.onnx'
        if sha256(onnx_path) != json.loads((root/'onnx'/f'{name}.sha256.json').read_text())['sha256']:
            raise RuntimeError('ONNX file changed since measurement')
        result[name] = {
            'checkpoint': str(directory/'checkpoints'/'last.ckpt'),
            'fp32_reproduces_validation_report': True,
            'fp16_vs_fp32': {**fp16_agreement(name),
                             'metrics_fp32': {k: 100*r32[k] for k in METRICS},
                             'metrics_fp16': {k: 100*r16[k] for k in METRICS},
                             'metric_differences_pp': {k: 100*(r16[k] - r32[k]) for k in METRICS}},
            'onnxruntime_vs_pytorch_fp32': onnx_agreement(name, onnx_path)}
        print(f'FIDELITY {name}: {json.dumps(result[name]["fp16_vs_fp32"]["metric_differences_pp"])}', flush=True)
    rpv_dump(root/'fidelity.json', {'protocol': PROTOCOL_ID, 'split': 'validation', 'test_used': False,
                                   'architectures': result, 'script_sha256': sha256(__file__)})
    print(f'EFFICIENCY FIDELITY COMPLETE: {root}', flush=True)


def smoke(root):
    """Engineering check, no timing kept: FLOPs, ONNX export, one ORT run and one FP16 forward."""
    root.mkdir(parents=True, exist_ok=False)
    for name in ARCHITECTURES:
        model, _ = load_model(name)
        flops = count_flops(model)['flops']
        path = export_eval(model, root/f'{name}.onnx')
        with torch.inference_mode():
            pre, post = example(1)
            reference = model(pre, post).change_logits.numpy()
            out = onnx_session(path, 1).run(None, {'pre': pre.numpy(), 'post': post.numpy()})[0]
            half = model.half().cuda()(*example(1, 'cuda', True)).change_logits.float().cpu().numpy()
        print(f'SMOKE {name}: flops={flops} onnx_max_diff={float(np.abs(out - reference).max()):.2e} '
              f'fp16_max_diff={float(np.abs(half - reference).max()):.2e}', flush=True)
    print(f'EFFICIENCY SMOKE PASSED: {root}', flush=True)


def complete(root):
    with (root/'costs.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    fid = json.loads((root/'fidelity.json').read_text())
    if len(rows) != 16 or set(fid['architectures']) != set(ARCHITECTURES) or fid['test_used'] is not False:
        raise RuntimeError('Incomplete efficiency outputs')
    rpv_dump(root/'COMPLETE.json', {'protocol': PROTOCOL_ID, 'cost_rows': len(rows),
                                   'architectures': list(ARCHITECTURES), 'fidelity_split': 'validation',
                                   'test_used': False, 'training_performed': False,
                                   'script_sha256': sha256(__file__)})
    print(f'EFFICIENCY COMPLETE: {root}', flush=True)


def main():
    pin_source()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['test', 'smoke', 'measure', 'fidelity', 'complete'])
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.stage == 'test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_efficiency.py']))
    if a.output is None:
        p.error('--output is required')
    if a.stage == 'complete':
        complete(a.output)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if a.stage == 'smoke':
        smoke(a.output)
    elif a.stage == 'measure':
        measure(a.output)
    else:
        if os.environ.get('PYTHONHASHSEED') != str(SEED):
            raise RuntimeError(f'Run fidelity with PYTHONHASHSEED={SEED}')
        fidelity(a.output)


if __name__ == '__main__':
    main()
