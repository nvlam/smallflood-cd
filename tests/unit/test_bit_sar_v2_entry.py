import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

PATH = Path(__file__).resolve().parents[2] / 'scripts/bit_sar_v2_entry.py'
spec = importlib.util.spec_from_file_location('entry_test', PATH)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_correct_source_accepted():
    import smallflood_cd.engine.validator
    entry.verify_loaded_modules()


def test_stale_loaded_submodule_rejected(monkeypatch):
    stale = ModuleType('smallflood_cd.old_shadow')
    stale.__file__ = str(entry.ROOT / 'engine/validator.py')
    monkeypatch.setitem(sys.modules, stale.__name__, stale)
    with pytest.raises(RuntimeError, match='wrong source'):
        entry.verify_loaded_modules()


def test_stale_package_search_path_rejected(monkeypatch):
    import smallflood_cd
    monkeypatch.setattr(smallflood_cd, '__path__', [str(entry.ROOT)])
    with pytest.raises(RuntimeError, match='wrong source'):
        entry.verify_loaded_modules()
