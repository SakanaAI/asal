"""
Replay utility functions — importable without Taichi or argparse.

Used by coralai/replay.py (the GUI tool) and tests.
"""

import glob
import os
import re
import sys


def discover_checkpoints(run_dir):
    """Return checkpoint dirs sorted by step number."""
    dirs = glob.glob(os.path.join(run_dir, "checkpoint_*"))
    dirs = [d for d in dirs if re.search(r"checkpoint_(\d+)$", d)]
    return sorted(dirs, key=lambda d: int(re.search(r"checkpoint_(\d+)$", d).group(1)))


def load_experiment_from_snapshot(snapshot_dir):
    """Import the EXPERIMENT instance from snapshot/experiment.py.

    Adds snapshot_dir to sys.path so deferred imports inside run_physics and
    run_evolution resolve to the snapshot's frozen code at call time.
    """
    import importlib.util

    exp_path = os.path.join(snapshot_dir, "experiment.py")
    if not os.path.exists(exp_path):
        raise FileNotFoundError(
            f"No experiment.py found in snapshot at {snapshot_dir!r}.\n"
            "This run pre-dates the Experiment class. Use the old "
            "experiment-specific replay.py in the snapshot folder instead.")

    if snapshot_dir not in sys.path:
        sys.path.insert(0, snapshot_dir)

    spec = importlib.util.spec_from_file_location("snapshot_experiment", exp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXPERIMENT
