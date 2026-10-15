import argparse
import json
from pathlib import Path

import numpy as np
import torch


def shannon_bits(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def binned_entropy(values, width):
    # Fixed-width bins anchored at zero, without clipping values.
    bins = np.floor(values.astype(np.float64) / width)
    _, counts = np.unique(bins, return_counts=True)
    return shannon_bits(counts)


def score_checkpoint(path, bin_width=1.0):
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("Bin width must be finite and positive.")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(state) or state.ndim != 4:
        raise ValueError("Expected a saved Coralai cell tensor.")
    if state.shape[0] != 1 or state.shape[1] != 14:
        raise ValueError("Expected the 14-channel coral_dev format.")

    cells = state.detach().cpu().numpy()[0]
    if not np.isfinite(cells).all():
        raise ValueError("Checkpoint contains invalid values.")

    genome = cells[13]
    if not np.equal(genome, np.floor(genome)).all():
        raise ValueError("Genome IDs must be integers.")

    alive = genome >= 0
    _, genome_counts = np.unique(genome[alive], return_counts=True)

    return {
        "checkpoint": str(path),
        "grid_shape": list(cells.shape[1:]),
        "alive_cells": int(alive.sum()),
        "occupied_genome_ids": int(genome_counts.size),
        "genome_shannon_bits": shannon_bits(genome_counts),
        "energy_binned_entropy_bits": binned_entropy(cells[0], bin_width),
        "infrastructure_binned_entropy_bits": binned_entropy(cells[1], bin_width),
        "bin_width": bin_width,
        "resource_scope": "all grid cells",
        "genome_scope": "living cells only",
        "method": "Shannon entropy in bits; fixed-width resource bins",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/week7/coralai_entropy.json"),
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        pattern = (
            "**/coral_dev_20261001_174547/"
            "checkpoint_0001000/substrate.pt"
        )
        matches = sorted({
            path.resolve()
            for root in [Path("data/raw"), Path("external/coralai")]
            for path in root.glob(pattern)
        })
        if len(matches) != 1:
            parser.error(
                "Supply a substrate.pt path; expected one matching patch checkpoint."
            )
        args.checkpoint = matches[0]

    result = score_checkpoint(args.checkpoint, args.bin_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("Saved:", args.output)
