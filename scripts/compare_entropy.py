import argparse
import csv
from pathlib import Path

import numpy as np
from score_entropy import score_checkpoint

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Compare saved flat and patch checkpoint entropy.")
parser.add_argument("--runs", type=Path, default=ROOT / "data/raw/runs")
parser.add_argument("--output", type=Path, default=ROOT / "results/week7")
args = parser.parse_args()
RUNS = args.runs.resolve()
OUTPUT = args.output.resolve()

RUN_IDS = {
    "flat": [
        "174414", "174438", "174457", "174510", "174523"
    ],
    "patches": [
        "174547", "174601", "174623", "174639", "174653"
    ],
}

METRICS = [
    "alive_cells",
    "occupied_genome_ids",
    "genome_shannon_bits",
    "energy_binned_entropy_bits",
    "infrastructure_binned_entropy_bits",
]

jobs = []
for condition, run_ids in RUN_IDS.items():
    for seed, run_id in zip(range(42, 47), run_ids):
        checkpoint = (
            RUNS / f"coral_dev_20261001_{run_id}"
            / "checkpoint_0001000/substrate.pt"
        )
        jobs.append((condition, seed, checkpoint))

missing = [str(path) for _, _, path in jobs if not path.is_file()]
if missing:
    raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))

rows = []
for condition, seed, checkpoint in jobs:
    result = score_checkpoint(checkpoint, bin_width=1.0)
    row = {
        "condition": condition,
        "seed": seed,
        "checkpoint_label": 1000,
        "checkpoint": (
            checkpoint.relative_to(ROOT).as_posix()
            if checkpoint.is_relative_to(ROOT) else checkpoint.as_posix()
        ),
        "bin_width": 1.0,
    }
    row.update({metric: result[metric] for metric in METRICS})
    rows.append(row)

OUTPUT.mkdir(parents=True, exist_ok=True)

details_path = OUTPUT / "entropy_by_run.csv"
with details_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for metric in METRICS:
    display = []
    for condition in RUN_IDS:
        values = np.array([
            row[metric] for row in rows
            if row["condition"] == condition
        ], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        summary.append({
            "condition": condition,
            "metric": metric,
            "n_runs": len(values),
            "mean": mean,
            "sample_sd": sd,
        })
        display.append(f"{condition}: {mean:.3f} ± {sd:.3f}")
    print(f"{metric}\n  " + "; ".join(display))

summary_path = OUTPUT / "entropy_summary.csv"
with summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print("\nCheckpoint label: 1000; five seeds per condition.")
print("Resource entropy: whole grid, fixed one-unit bins.")
print("Genome entropy: living cells only.")
print("Saved:", details_path)
print("Saved:", summary_path)
