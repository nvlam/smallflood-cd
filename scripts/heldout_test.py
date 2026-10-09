"""heldout_test_v1 (A4): single pre-specified evaluation of the sealed test set.

docs/heldout_test_protocol_v1_20261009.md. Evaluates best and last of 36 existing runs
(6 arms x 2 recipes x 3 seeds) with review_pilot_validation.evaluate(); the split-specific
lines are the only change (see SPLIT_EDITS). This is the only script allowed to read the test
split, and only with --approve-test. Never trains. One process per seed, because evaluate()
requires PYTHONHASHSEED to equal the run seed.

Stages:
  test                     synthetic CPU tests
  regress  --seed S        validation mode; must reproduce existing validation reports exactly
  evaluate --seed S --output ROOT --approve-test [--resume]
  analyze  --output ROOT   T1/T2/T3 and benchmark tables from the 72 evaluations
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from bit_sar_v2_entry import pin_source
import review_pilot_validation as rpv
from review_pilot_validation import sha256, write_csv

PROTOCOL_ID = 'heldout_test_v1'
OUTPUT_PARENT = Path('artifacts/heldout_test_v1')
SPLIT_FILE = 'data/splits/split_v1.yaml'
SPLIT_SHA256 = '88d8c3a7d1fd652b870c0dc8c97348eabd5b1b70a71f052568a11e7213431472'
TEST_EVENTS = ('20210727_Weihui', '20230609_NovaKakhovka', '20231201_Jubba')
ARMS = ('A', 'B', 'C', 'D', 'FC', 'BIT')
RECIPES = {'old': (42, 1337, 2026), 'R1': (201, 202, 203)}
CHECKPOINTS = ('last.ckpt', 'best_composite.ckpt')
PARAMETERS = {'A': 1079865, 'B': 1079865, 'C': 1080805, 'D': 1080805, 'FC': 487857, 'BIT': 3492642}
# T3 cut points (protocol section 5, adopted before any test access).
T3_GENERALIZES, T3_NOT = 7, 4
LOW_F1 = 0.20

# Old recipe, seed 42 (original runners and reviews).
S42_RUNS = {
    **{c: (f'runs/proposed_factorial_v1/20260927T075837Z/{c}_proposed',
           f'artifacts/proposed_factorial_review/20260928T004225Z/results/{c}_proposed') for c in 'ABCD'},
    'FC': ('runs/pilot15/20260924T032911Z/20260924T051127718020Z_fc_siam_diff',
           'artifacts/pilot15_review/20260924T082109094745Z/20260924T051127718020Z_fc_siam_diff'),
    'BIT': ('runs/bit_sar_v2_pilot15/20260925T015023Z/pilot/20260925T015034589216Z_bit_sar_v2',
            'artifacts/bit_sar_v2_validation_review/20260925T081025Z/results/'
            '20260925T081029198595Z/20260925T015034589216Z_bit_sar_v2'),
}
CONFIRMATION = {  # seed -> (run root, validation review root)
    **{s: ('runs/direction_b_confirmation_v1/20261003T135430Z',
           f'artifacts/direction_b_confirmation_review/20261003T135430Z/s{s}') for s in (1337, 2026)},
    **{s: ('runs/recipe_confirmation_v1/20261007T234436Z',
           f'artifacts/recipe_confirmation_review/20261007T234436Z/s{s}') for s in (201, 202, 203)},
}
# Validation-mode regression targets: existing reports written by the current evaluate().
REGRESSION = {42: ('A', 'BIT'), 201: ('A', 'FC')}

# The only edits to review_pilot_validation.evaluate(): (old, new, expected count).
SPLIT_EDITS = (
    ('def evaluate(directory, checkpoint_name, output, args):',
     'def evaluate_split(directory, checkpoint_name, output, args, split):', 1),
    ("dataset = ManifestDataset(manifest, 'validation', load_array, use_uncertain_mask=False)",
     'dataset = ManifestDataset(manifest, split, load_array, use_uncertain_mask=False)', 1),
    ("if not dataset.records or len(dataset) != original['validation_patches']:",
     "if not dataset.records or (split == 'validation' and len(dataset) != original['validation_patches']):", 1),
    ("'split': 'validation', 'test_used': False,", "'split': split, 'test_used': split == 'test',", 2),
)


def build_evaluate_split():
    """review_pilot_validation.evaluate with SPLIT_EDITS applied, in that module's globals."""
    source = inspect.getsource(rpv.evaluate)
    for old, new, count in SPLIT_EDITS:
        if source.count(old) != count:
            raise RuntimeError(f'evaluate() changed; expected {count} x {old!r}')
        source = source.replace(old, new)
    namespace = dict(vars(rpv))
    exec(compile(source, rpv.__file__, 'exec'), namespace)
    function = namespace['evaluate_split']

    def evaluate_split(directory, checkpoint_name, output, args, split):
        if split not in ('validation', 'test'):
            raise ValueError(f'Unknown split {split}')
        return function(directory, checkpoint_name, output, args, split)
    return evaluate_split


evaluate_split = build_evaluate_split()


def runs():
    """The fixed 36-run list of protocol section 3."""
    result = []
    for recipe, seeds in RECIPES.items():
        for seed in seeds:
            for arm in ARMS:
                if seed == 42:
                    directory, review = map(Path, S42_RUNS[arm])
                else:
                    root, review_root = map(Path, CONFIRMATION[seed])
                    if arm in 'ABCD':
                        directory = root/f'factorial_s{seed}'/f'{arm}_proposed'
                    else:
                        model = {'FC': 'fc_siam_diff', 'BIT': 'bit_sar_v2'}[arm]
                        matches = sorted((root/f'pilots_s{seed}').glob(f'*_{model}'))
                        if len(matches) != 1:
                            raise RuntimeError(f'Expected one {model} run for seed {seed}')
                        directory = matches[0]
                    review = review_root/directory.name
                result.append({'recipe': recipe, 'seed': seed, 'arm': arm,
                               'directory': directory, 'review': review})
    if len(result) != 36 or len({str(r['directory']) for r in result}) != 36:
        raise RuntimeError('Run list is not the fixed 36 runs')
    return result


def validated_sha(run, checkpoint):
    """Checkpoint SHA256 must equal the one in its existing validation review report."""
    path = run['directory']/'checkpoints'/checkpoint
    report = json.loads((run['review']/Path(checkpoint).stem/'report.json').read_text())
    digest = sha256(path)
    if report.get('split') != 'validation' or report.get('checkpoint_sha256') != digest:
        raise RuntimeError(f'Checkpoint differs from its validation record: {path}')
    return digest, report


GT_COUNT_KEYS = ('patches', 'component_target', 'target_touching_rim', 'small_target_clipped',
                 'small_target_interior', 'boundary_target', 'boundary_valid_pixels')


def denominators(destination):
    """Ground-truth-only quantities; identical for every checkpoint on the same split."""
    report = json.loads((destination/'report.json').read_text())
    detail = json.loads((destination/'object_boundary_report.json').read_text())
    with (destination/'events.csv').open(newline='') as handle:
        events = {r['event']: {'valid_pixels': int(r['valid_pixels']), 'positives': int(r['tp'])+int(r['fn'])}
                  for r in csv.DictReader(handle)}
    g = report['global_pixel']
    return {'patches': report['validation_patches'], 'valid_pixels': g['valid_pixels'],
            'positives': g['tp']+g['fn'], 'events': events,
            'object_counts': {k: detail['counts'][k] for k in GT_COUNT_KEYS},
            'object_events': {e['event']: {k: e[k] for k in GT_COUNT_KEYS} for e in detail['events']}}


def check_denominators(root, current):
    path = root/'denominators.json'
    if not path.exists():
        rpv_dump(path, current)
        return
    if json.loads(path.read_text()) != json.loads(json.dumps(current)):
        raise RuntimeError('Test denominators changed between evaluations')


def rpv_dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def single_use(root):
    """Refuse a second heldout_test_v1 evaluation (any other output root)."""
    others = [p for p in OUTPUT_PARENT.glob('*') if p.resolve() != Path(root).resolve()] if OUTPUT_PARENT.exists() else []
    if others:
        raise RuntimeError(f'heldout_test_v1 already has outputs: {others}; single use only')


def recorded(seed_dir):
    progress = seed_dir/'progress.jsonl'
    rows = [json.loads(x) for x in progress.read_text().splitlines()] if progress.exists() else []
    return {(r['run'], r['checkpoint']) for r in rows}


def prepare_resume(seed_dir, directory, checkpoint):
    """True if this evaluation is recorded as complete; moves any unrecorded output aside."""
    destination = seed_dir/directory.name/Path(checkpoint).stem
    if (directory.name, checkpoint) in recorded(seed_dir):
        return True
    if destination.exists():
        index = len(list(destination.parent.glob(destination.name + '.incomplete_*')))
        shutil.move(str(destination), str(destination.parent/f'{destination.name}.incomplete_{index}'))
    return False


def evaluate_seed(seed, root, resume):
    seed_dir = root/f's{seed}'
    if seed_dir.exists() and not resume:
        raise RuntimeError(f'{seed_dir} exists; use --resume to continue only unevaluated checkpoints')
    seed_dir.mkdir(parents=True, exist_ok=resume)
    progress = seed_dir/'progress.jsonl'
    args = SimpleNamespace(device='cuda', batch_size=8, object_boundary=True)
    selected = [r for r in runs() if r['seed'] == seed]
    for run in selected:
        for checkpoint in CHECKPOINTS:
            digest, validation = validated_sha(run, checkpoint)
            ckpt = torch.load(run['directory']/'checkpoints'/checkpoint, map_location='cpu', weights_only=False)
            if int(ckpt['epoch'])+1 != validation['checkpoint_epoch_1based']:
                raise RuntimeError('Checkpoint epoch differs from its validation record')
            if checkpoint == 'last.ckpt' and int(ckpt['epoch']) != 14:
                raise RuntimeError('Last checkpoint is not epoch 15')
            del ckpt
            if resume and prepare_resume(seed_dir, run['directory'], checkpoint):
                print(f'ALREADY EVALUATED: {run["directory"].name}/{checkpoint}', flush=True)
                continue
            result = evaluate_split(run['directory'], checkpoint, seed_dir, args, 'test')
            destination = seed_dir/run['directory'].name/Path(checkpoint).stem
            report = json.loads((destination/'report.json').read_text())
            if report['checkpoint_sha256'] != digest or sha256(run['directory']/'checkpoints'/checkpoint) != digest:
                raise RuntimeError('Checkpoint changed during evaluation')
            if report['split'] != 'test' or report['training_performed'] is not False:
                raise RuntimeError('Report is not a test-only evaluation')
            current = denominators(destination)
            if set(current['events']) != set(TEST_EVENTS):
                raise RuntimeError(f'Unexpected test events: {sorted(current["events"])}')
            check_denominators(root, current)
            row = {'recipe': run['recipe'], 'seed': seed, 'arm': run['arm'], 'run': run['directory'].name,
                   'directory': str(run['directory']), 'checkpoint_sha256': digest, **result}
            with progress.open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            print(f'TEST EVALUATED: {run["directory"].name}/{checkpoint}', flush=True)
    rows = [json.loads(x) for x in progress.read_text().splitlines()]
    keys = {(r['run'], r['checkpoint']) for r in rows}
    if len(keys) != 12 or len(rows) != 12:
        raise RuntimeError(f'Seed {seed}: expected 12 distinct evaluations, found {len(rows)}')
    write_csv(seed_dir/'summary.csv', rows)
    rpv_dump(seed_dir/'COMPLETE.json', {'protocol': PROTOCOL_ID, 'seed': seed, 'evaluations': 12,
                                       'split': 'test', 'training_performed': False,
                                       'checkpoints_match_validation_records': True,
                                       'denominators_consistent': True, 'script_sha256': sha256(__file__)})
    print(f'HELDOUT TEST s{seed} COMPLETE: {seed_dir}', flush=True)


COMPARED_REPORT = ('global_pixel', 'event_macro', 'event_macro_defined_counts', 'by_patch_category',
                   'negative_patches', 'negative_patches_with_any_false_positive', 'all_invalid_patches',
                   'validation_patches', 'checkpoint_sha256', 'checkpoint_epoch_1based')
COMPARED_DETAIL = ('counts', 'global', 'events', 'event_macro', 'event_macro_defined_counts')


def regress(seed, output):
    """Validation mode must reproduce existing validation reports exactly (no test access)."""
    args = SimpleNamespace(device='cuda', batch_size=8, object_boundary=True)
    checked = []
    for run in (r for r in runs() if r['seed'] == seed and r['arm'] in REGRESSION[seed]):
        digest, _ = validated_sha(run, 'last.ckpt')
        evaluate_split(run['directory'], 'last.ckpt', output, args, 'validation')
        new = output/run['directory'].name/'last'
        old = run['review']/'last'
        for name, keys in (('report.json', COMPARED_REPORT), ('object_boundary_report.json', COMPARED_DETAIL)):
            a, b = json.loads((new/name).read_text()), json.loads((old/name).read_text())
            for key in keys:
                if a[key] != b[key]:
                    raise RuntimeError(f'Regression: {run["arm"]} s{seed} {name}:{key} differs')
        if (new/'events.csv').read_text() != (old/'events.csv').read_text():
            raise RuntimeError(f'Regression: {run["arm"]} s{seed} events.csv differs')
        checked.append({'arm': run['arm'], 'seed': seed, 'checkpoint_sha256': digest,
                        'reference': str(old)})
    rpv_dump(output/'REGRESSION_PASS.json', {'protocol': PROTOCOL_ID, 'seed': seed, 'split': 'validation',
                                            'test_used': False, 'checked': checked,
                                            'script_sha256': sha256(__file__)})
    print(f'HELDOUT TEST REGRESSION s{seed} PASSED: {output}', flush=True)


# ---------- analysis (protocol section 5) ----------

def pct(value):
    return None if value in (None, '') else 100*float(value)


def load_results(root):
    """values[(recipe, arm, seed, checkpoint)] = metrics in percent, with per-event dicts."""
    values = {}
    for seeds in RECIPES.values():
        for seed in seeds:
            base = root/f's{seed}'
            complete = json.loads((base/'COMPLETE.json').read_text())
            if complete.get('evaluations') != 12 or complete.get('split') != 'test':
                raise RuntimeError(f'Seed {seed} incomplete')
            with (base/'summary.csv').open(newline='') as handle:
                for r in csv.DictReader(handle):
                    with (base/r['run']/Path(r['checkpoint']).stem/'events.csv').open(newline='') as h:
                        events = {e['event']: e for e in csv.DictReader(h)}
                    values[(r['recipe'], r['arm'], int(r['seed']), r['checkpoint'])] = {
                        'event_macro_f1': pct(r['event_macro_f1']), 'pixel_f1': pct(r['f1']),
                        'component_f1': pct(r['object_global_component_f1']),
                        'interior': pct(r['object_global_small_component_recall_interior']),
                        'boundary_f1': pct(r['object_global_boundary_f1']),
                        'inner_band': pct(r['object_global_boundary_inner_band_iou']),
                        'event_f1': {e: pct(events[e]['f1']) for e in TEST_EVENTS},
                        'event_pred_positive': {e: pct(events[e]['pred_positive_fraction']) for e in TEST_EVENTS},
                        'event_gt_positive': {e: pct(events[e]['gt_positive_fraction']) for e in TEST_EVENTS}}
    if len(values) != 72:
        raise RuntimeError(f'Expected 72 evaluations, found {len(values)}')
    return values


def sign_rule(pairs, kind):
    """recipe_confirmation_v1 section 8 rules; None anywhere -> not evaluable."""
    if any(x is None for pair in pairs.values() for x in pair):
        return 'not evaluable'
    if kind == 'paired':
        if all(a <= 0 and b <= 0 for a, b in pairs.values()):
            return 'supported'
        if all(a > 0 and b > 0 for a, b in pairs.values()):
            return 'contradicted'
        return 'inconsistent'
    if all(a >= 0 and b >= 0 for a, b in pairs.values()):
        return 'supported'
    if all(a < 0 for a, _ in pairs.values()) or all(b < 0 for _, b in pairs.values()):
        return 'contradicted'
    return 'inconsistent'


def diff(a, b):
    return None if a is None or b is None else a - b


def t1(values, recipe):
    seeds = RECIPES[recipe]
    v = lambda arm, s, k: values[(recipe, arm, s, 'last.ckpt')][k]  # noqa: E731
    size = {s: (diff(v('B', s, 'interior'), v('A', s, 'interior')), diff(v('D', s, 'interior'), v('C', s, 'interior')))
            for s in seeds}
    boundary = {s: (diff(v('C', s, 'inner_band'), v('A', s, 'inner_band')),
                    diff(v('D', s, 'inner_band'), v('B', s, 'inner_band'))) for s in seeds}
    fc = {s: (diff(v('FC', s, 'event_macro_f1'), v('A', s, 'event_macro_f1')),
              diff(v('FC', s, 'event_macro_f1'), v('BIT', s, 'event_macro_f1'))) for s in seeds}
    return {'size_weighting': {'contrasts_pp': {s: {'B-A': a, 'D-C': b} for s, (a, b) in size.items()},
                               'outcome': sign_rule(size, 'paired')},
            'boundary_package': {'contrasts_pp': {s: {'C-A': a, 'D-B': b} for s, (a, b) in boundary.items()},
                                 'outcome': sign_rule(boundary, 'paired')},
            'fc_not_worse': {'contrasts_pp': {s: {'FC-A': a, 'FC-BIT': b} for s, (a, b) in fc.items()},
                             'outcome': sign_rule(fc, 'fc')}}


def spread(xs):
    defined = [x for x in xs if x is not None]
    if len(defined) != len(xs):
        return {'values': xs, 'mean': None, 'range': None}
    return {'values': xs, 'mean': sum(xs)/len(xs), 'range': max(xs) - min(xs)}


def t2(values, recipe):
    seeds = RECIPES[recipe]
    groups = {}
    for arm in ARMS:
        rows = [values[(recipe, arm, s, 'last.ckpt')] for s in seeds]
        groups[arm] = {'event_macro_f1': spread([r['event_macro_f1'] for r in rows]),
                       'event_f1': {e: spread([r['event_f1'][e] for r in rows]) for e in TEST_EVENTS}}
    pairs = {}
    for i, a in enumerate(ARMS):
        for b in ARMS[i+1:]:
            signs = []
            for s in seeds:
                d = values[(recipe, a, s, 'last.ckpt')]['event_macro_f1'] - values[(recipe, b, s, 'last.ckpt')]['event_macro_f1']
                signs.append((d > 0) - (d < 0))
            pairs[f'{a}-{b}'] = len(set(signs)) == 1
    return {'by_arm': groups, 'rank_pairs_stable': pairs,
            'stable_pairs': sum(pairs.values()), 'pairs': len(pairs)}


def t3(values, gt_fraction):
    low = min(TEST_EVENTS, key=lambda e: gt_fraction[e])
    groups, hits = {}, 0
    for recipe, seeds in RECIPES.items():
        for arm in ARMS:
            ranges = {}
            for e in TEST_EVENTS:
                xs = [values[(recipe, arm, s, 'last.ckpt')]['event_f1'][e] for s in seeds]
                if any(x is None for x in xs):
                    raise RuntimeError(f'T3 not evaluable: undefined F1 for {e}')
                ranges[e] = max(xs) - min(xs)
            top = max(ranges.values())
            hit = ranges[low] == top
            hits += hit
            groups[f'{recipe}:{arm}'] = {'ranges_pp': ranges, 'largest': [e for e in ranges if ranges[e] == top],
                                         'low_fraction_event_largest': hit}
    outcome = ('generalizes' if hits >= T3_GENERALIZES else
               'does not generalize' if hits <= T3_NOT else 'partial')
    return {'low_fraction_event': low, 'gt_positive_percent': gt_fraction, 'groups': groups,
            'groups_with_low_event_largest': hits, 'groups_total': len(groups), 'outcome': outcome}


def low_events(values):
    return {f'{r}:{a}:s{s}': sum(1 for e in TEST_EVENTS
                                 if values[(r, a, s, 'last.ckpt')]['event_f1'][e] is not None
                                 and values[(r, a, s, 'last.ckpt')]['event_f1'][e] < 100*LOW_F1)
            for (r, a, s, c) in values if c == 'last.ckpt'}


def benchmark(values, checkpoint):
    table = {}
    for recipe, seeds in RECIPES.items():
        for arm in ARMS:
            rows = [values[(recipe, arm, s, checkpoint)] for s in seeds]
            table[f'{recipe}:{arm}'] = {'parameters': PARAMETERS[arm], **{
                k: spread([r[k] for r in rows]) for k in
                ('event_macro_f1', 'pixel_f1', 'component_f1', 'interior', 'boundary_f1', 'inner_band')}}
    return table


def analyze(root):
    values = load_results(root)
    den = json.loads((root/'denominators.json').read_text())
    gt_fraction = {e: 100*den['events'][e]['positives']/den['events'][e]['valid_pixels'] for e in TEST_EVENTS}
    return {'protocol': PROTOCOL_ID, 'split': 'test', 'primary_checkpoint': 'last (epoch 15)',
            'T1': {recipe: t1(values, recipe) for recipe in RECIPES},
            'T2': {recipe: t2(values, recipe) for recipe in RECIPES},
            'T3': t3(values, gt_fraction),
            'events_below_20_f1_last': low_events(values),
            'event_pred_vs_gt_positive_last': {f'{r}:{a}:s{s}': {'pred': v['event_pred_positive'], 'gt': v['event_gt_positive']}
                                               for (r, a, s, c), v in values.items() if c == 'last.ckpt'},
            'benchmark_last': benchmark(values, 'last.ckpt'),
            'benchmark_best_descriptive': benchmark(values, 'best_composite.ckpt'),
            'training_performed': False}


def main():
    pin_source()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['test', 'regress', 'evaluate', 'analyze'])
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--approve-test', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.stage == 'test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_heldout_test.py',
                                      'tests/unit/test_review_pilot_validation.py']))
    if sha256(ROOT/SPLIT_FILE) != SPLIT_SHA256:
        raise RuntimeError('Split file changed')
    if a.stage == 'analyze':
        if a.output is None:
            p.error('analyze needs --output')
        result = analyze(a.output)
        rpv_dump(a.output/'RESULTS.json', result)
        rpv_dump(a.output/'COMPLETE.json', {'protocol': PROTOCOL_ID, 'evaluations': 72, 'split': 'test',
                                           'training_performed': False,
                                           'checkpoints_match_validation_records': True,
                                           'denominators_consistent': True, 'script_sha256': sha256(__file__)})
        print(json.dumps({k: result[k] for k in ('T1', 'T3')}, indent=2), flush=True)
        return
    seeds = [s for ss in RECIPES.values() for s in ss]
    if a.seed not in seeds:
        p.error(f'--seed must be one of {seeds}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if a.stage == 'regress':
        if a.seed not in REGRESSION:
            p.error(f'regress runs for seeds {sorted(REGRESSION)}')
        output = a.output or Path('artifacts/heldout_test_regression')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/f's{a.seed}'
        output.mkdir(parents=True, exist_ok=False)
        regress(a.seed, output)
        return
    if not a.approve_test or a.output is None:
        p.error('evaluate needs --approve-test and --output (the protocol output root)')
    if a.output.parent.resolve() != OUTPUT_PARENT.resolve():
        p.error(f'--output must be directly under {OUTPUT_PARENT}')
    single_use(a.output)
    evaluate_seed(a.seed, a.output, a.resume)


if __name__ == '__main__':
    main()
