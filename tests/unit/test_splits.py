import csv

import pytest

from smallflood_cd.data.splits import validate_event_isolation


def _write_manifest(path, rows) -> None:
    fields = [
        "patch_id",
        "event_id",
        "pre_path",
        "post_path",
        "label_path",
        "valid_mask_path",
        "coherence_path",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_event_leakage_is_rejected(tmp_path) -> None:
    common = {
        "event_id": "same-event",
        "pre_path": "a.npy",
        "post_path": "b.npy",
        "label_path": "y.npy",
        "valid_mask_path": "v.npy",
        "coherence_path": "",
    }
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {**common, "patch_id": "one", "split": "train"},
            {**common, "patch_id": "two", "split": "test"},
        ],
    )
    with pytest.raises(ValueError, match="Event leakage"):
        validate_event_isolation(manifest)
