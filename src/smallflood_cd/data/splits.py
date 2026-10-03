from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from smallflood_cd.data.datasets import read_manifest


def validate_event_isolation(manifest: str | Path) -> None:
    """Reject any event assigned to more than one split."""
    event_splits: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        for record in read_manifest(manifest, split):
            event_splits[record.event_id].add(split)
    leaked = {event: sorted(splits) for event, splits in event_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Event leakage detected: {leaked}")
