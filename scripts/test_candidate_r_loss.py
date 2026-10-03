"""Source-pinned synthetic tests only; no data loading or training run."""
from pathlib import Path
import sys

from bit_sar_v2_entry import pin_source, verify_loaded_modules


if __name__ == '__main__':
    pin_source()
    import pytest
    result = pytest.main(['--import-mode=importlib', '-q',
                         'tests/unit/test_local_component_loss.py',
                         'tests/unit/test_losses.py',
                         'tests/unit/test_size_aware_tversky.py'])
    verify_loaded_modules()
    raise SystemExit(result)
