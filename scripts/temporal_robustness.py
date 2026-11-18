"""Seed-robustness helpers for the temporal-novelty search (issue #37).

Everything here is plain numpy, with no jax, torch or Coralai imports, so the
logic can be unit-tested on any laptop. The loss is ASAL's open-endedness loss,
so LOWER is better (more novel / keeps changing).

These mirror the issue #26/#31 ideas for the supervised search: average a
candidate over several simulation seeds, charge it for disagreeing with itself
across seeds, then check the winner against the hand-made world on seeds the
search never saw.
"""

import math

import numpy as np

MAX_SEED = 4294967295
TIE = 1e-7  # same tie tolerance the single-seed pilot already uses


def _is_seed(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_SEED


def validate_seed_plan(simulation_seed, search_seeds, comparison_seeds, spread_penalty):
    """Reject seed plans that would make the held-out check meaningless."""
    if not isinstance(search_seeds, list) or not all(_is_seed(s) for s in search_seeds):
        raise ValueError("search_seeds must be a list of integer seeds")
    if len(search_seeds) < 2 or len(set(search_seeds)) != len(search_seeds):
        raise ValueError("Use at least two distinct search seeds so scores can be averaged")
    if search_seeds[0] != simulation_seed:
        raise ValueError("The first search seed must equal simulation_seed (it is the seed the reference runs and figures use)")
    if not isinstance(comparison_seeds, list) or not all(_is_seed(s) for s in comparison_seeds):
        raise ValueError("comparison_seeds must be a list of integer seeds")
    if len(comparison_seeds) < 3 or len(set(comparison_seeds)) != len(comparison_seeds):
        raise ValueError("Use at least three distinct held-out comparison seeds")
    if set(comparison_seeds) & set(search_seeds):
        raise ValueError("Held-out comparison seeds must not overlap the search seeds")
    if isinstance(spread_penalty, bool) or not isinstance(spread_penalty, (int, float)) \
            or not math.isfinite(spread_penalty) or spread_penalty < 0:
        raise ValueError("spread_penalty must be a finite number >= 0")


def aggregate_losses(losses, spread_penalty=0.0):
    """Summarize one world's loss on several seeds.

    selection_loss = mean + spread_penalty * sample_std. With penalty 0 this is
    the plain average; a positive penalty lets a steady world beat an erratic
    one that is only good on a lucky seed.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite per-seed losses")
    if spread_penalty < 0:
        raise ValueError("spread_penalty must be >= 0")
    std = float(values.std(ddof=1))
    mean = float(values.mean())
    return {"n_seeds": len(values), "mean_loss": mean, "loss_std": std,
            "spread_penalty": float(spread_penalty),
            "selection_loss": mean + float(spread_penalty) * std,
            "per_seed_loss": [float(v) for v in values]}


def paired_heldout(hand_losses, selected_losses, tolerance=TIE):
    """Compare hand-made vs search-selected on the same unseen seeds.

    difference = hand_loss - selected_loss, so POSITIVE means the selected world
    is more novel on that seed. Pairing by seed removes seed-to-seed luck.
    """
    hand = np.asarray(hand_losses, dtype=float)
    selected = np.asarray(selected_losses, dtype=float)
    if hand.shape != selected.shape or hand.ndim != 1 or len(hand) < 2:
        raise ValueError("Need matching per-seed losses for at least two seeds")
    if not (np.isfinite(hand).all() and np.isfinite(selected).all()):
        raise ValueError("Losses must be finite")
    diff = hand - selected
    improved = int((diff > tolerance).sum())
    mean = float(diff.mean())
    held_up = improved * 2 > len(diff) and mean > tolerance
    return {"n_seeds": len(diff), "positive_means": "selected world is more novel than the hand-made one",
            "improved_seeds": improved, "paired_differences": [float(d) for d in diff],
            "paired_difference_mean": mean, "paired_difference_sample_std": float(diff.std(ddof=1)),
            "hand_mean_loss": float(hand.mean()), "selected_mean_loss": float(selected.mean()),
            "held_up": bool(held_up),
            "rule": "held_up = selected better on a majority of unseen seeds AND better on average"}


def changed_fraction(before, after):
    """Fraction of cells whose occupied/empty status differs between two frames."""
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("Frames must be 2-D and the same shape")
    return float((before != after).mean())


def change_summary(occupancy_frames):
    """Plain-language 'how much does this world keep changing' numbers.

    occupancy_frames is the ordered list of boolean occupancy grids at the
    sampled updates (including update 0). Returns:
      changed_from_start  fraction of cells that differ between first and last frame
      ongoing_change      mean fraction of cells that differ between consecutive frames
      late_change         the same, over the second half of the run only. It tells a
                          world that keeps moving from one that settles and freezes.
    These are reading aids for the story, not a second search objective.
    """
    if len(occupancy_frames) < 3:
        raise ValueError("Need at least three frames")
    steps = [changed_fraction(a, b) for a, b in zip(occupancy_frames[:-1], occupancy_frames[1:])]
    return {"changed_from_start": changed_fraction(occupancy_frames[0], occupancy_frames[-1]),
            "ongoing_change": float(np.mean(steps)),
            "late_change": float(np.mean(steps[len(steps) // 2:])),
            "per_step_change": steps}
