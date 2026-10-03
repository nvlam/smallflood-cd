from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from smallflood_cd.engine.experiment import (
    expand_grid,
    load_matrix,
    run_experiment,
    write_combined_csv,
)
from smallflood_cd.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen SmallFlood-CD experiment matrix")
    parser.add_argument(
        "--matrix", default="configs/experiment/experiment_matrix.yaml"
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment", action="append", help="Run only selected names")
    parser.add_argument("--seed", action="append", type=int, help="Override matrix seeds")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    seeds, entries, matrix_directory = load_matrix(arguments.matrix)
    if arguments.seed:
        seeds = arguments.seed
    selected = set(arguments.experiment or [])
    if selected:
        entries = [entry for entry in entries if entry["name"] in selected]
        missing = selected - {entry["name"] for entry in entries}
        if missing:
            raise ValueError(f"Unknown experiment names: {sorted(missing)}")
    output_root = Path(arguments.output_root or (
        Path("runs") / datetime.now(timezone.utc).strftime("matrix-%Y%m%d-%H%M%S")
    ))
    jobs = []
    for entry in entries:
        config_path = matrix_directory / entry["config"]
        base_config = load_experiment_config(config_path)
        for variant, config in expand_grid(base_config):
            for seed in seeds:
                jobs.append((entry["name"], variant, seed, config))
    output_root.mkdir(parents=True, exist_ok=False)
    plan = [
        {"experiment": name, "variant": variant, "seed": seed}
        for name, variant, seed, _ in jobs
    ]
    (output_root / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )
    if arguments.dry_run:
        print(f"planned_jobs={len(jobs)} output_root={output_root}")
        return
    results = []
    for index, (name, variant, seed, config) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {name} variant={variant} seed={seed}", flush=True)
        results.append(run_experiment(name, variant, seed, config, output_root))
        write_combined_csv(results, output_root / "all_results.csv")
    print(f"completed_jobs={len(results)} results={output_root / 'all_results.csv'}")


if __name__ == "__main__":
    main()
