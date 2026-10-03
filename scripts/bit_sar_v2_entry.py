"""Pin project src imports before pytest or BIT-SAR smoke imports; never delete old code."""
from __future__ import annotations

import importlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_loaded_modules():
    expected = (ROOT / 'src/smallflood_cd').resolve()
    for name, module in list(sys.modules.items()):
        if name != 'smallflood_cd' and not name.startswith('smallflood_cd.'):
            continue
        filename = getattr(module, '__file__', None)
        locations = list(getattr(module, '__path__', []))
        if filename:
            locations.append(filename)
        if not locations or any(
            Path(p).resolve() != expected and expected not in Path(p).resolve().parents
            for p in locations
        ):
            raise RuntimeError(f'STOP: wrong source already loaded: {name}: {locations}')


def pin_source():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f'Run from project root: {ROOT}')
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'src'))
    verify_loaded_modules()
    for name in ('smallflood_cd', 'smallflood_cd.engine.validator',
                 'smallflood_cd.engine.trainer', 'smallflood_cd.engine.experiment',
                 'smallflood_cd.models.registry'):
        module = importlib.import_module(name)
        verify_loaded_modules()
        print(f'SOURCE {name}: {module.__file__}', flush=True)
    from smallflood_cd.engine.validator import SELECTION_PROTOCOL
    if SELECTION_PROTOCOL['version'] != 'event_pixel_v2':
        raise RuntimeError('Wrong validation protocol')
    print('SOURCE IMPORT CHECK PASSED', flush=True)


def main():
    pin_source()
    if sys.argv[1:] == ['test-review']:
        import pytest
        code = pytest.main(['--import-mode=importlib', '-q',
                            'tests/unit/test_review_pilot_validation.py',
                            'tests/unit/test_bit_sar_v2_entry.py'])
        verify_loaded_modules()
        raise SystemExit(code)
    if len(sys.argv) > 1 and sys.argv[1] == 'review':
        sys.argv = [str(ROOT / 'scripts/review_pilot_validation.py'), *sys.argv[2:]]
        runpy.run_path(sys.argv[0], run_name='__main__')
        verify_loaded_modules()
        return
    if sys.argv[1:] == ['test']:
        import pytest
        code = pytest.main(['--import-mode=importlib', '-q',
                            'tests/unit/test_bit_sar_v2_smoke.py',
                            'tests/unit/test_bit_sar_v2_entry.py'])
        verify_loaded_modules()
        raise SystemExit(code)
    if sys.argv[1:] == ['test-pilot']:
        import pytest
        code = pytest.main(['--import-mode=importlib', '-q',
                            'tests/unit/test_bit_sar_v2_pilot.py',
                            'tests/unit/test_bit_sar_v2_smoke.py',
                            'tests/unit/test_bit_sar_v2_entry.py'])
        verify_loaded_modules()
        raise SystemExit(code)
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot':
        sys.argv = [str(ROOT / 'scripts/pilot_bit_sar_v2.py'), *sys.argv[2:]]
        runpy.run_path(sys.argv[0], run_name='__main__')
        verify_loaded_modules()
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        sys.argv = [str(ROOT / 'scripts/smoke_bit_sar_v2.py'), *sys.argv[2:]]
        runpy.run_path(sys.argv[0], run_name='__main__')
        verify_loaded_modules()
        return
    raise SystemExit('Usage: bit_sar_v2_entry.py test | test-pilot | test-review | smoke ... | pilot ... | review ...')


if __name__ == '__main__':
    main()
