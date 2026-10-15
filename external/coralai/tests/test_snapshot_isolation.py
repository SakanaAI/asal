"""
Snapshot isolation test.

Verifies that coralai/replay.py uses the code frozen in a run's snapshot/
directory — not the live experiment files. This ensures old runs can be
replayed even after the experiment source code is modified.

Also checks that the run meta.json records the git hash so the framework
code version can be recovered manually if needed.

Run from the repo root:
    python tests/test_snapshot_isolation.py
"""

import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY   = sys.executable
PASS_COUNT = 0
FAIL_COUNT = 0


def ok(label):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS  {label}")


def fail(label, detail=""):
    global FAIL_COUNT
    FAIL_COUNT += 1
    lines = [f"  FAIL  {label}"]
    if detail:
        for line in detail.strip().splitlines()[-15:]:
            lines.append(f"        {line}")
    print("\n".join(lines))


def run_cmd(args, timeout=300):
    return subprocess.run([PY] + args, capture_output=True, text=True,
                         cwd=REPO, timeout=timeout)


def newest_run_dir():
    dirs = glob.glob(os.path.join(REPO, "runs", "coral_dev_*"))
    dirs = [d for d in dirs if re.search(r"\d{8}_\d{6}$", d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


if __name__ == "__main__":
    print("=" * 60)
    print("Snapshot Isolation Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Create a run with at least one checkpoint
    # ------------------------------------------------------------------
    print("  [1] Running 150 steps with checkpoint at step 100...")
    r = run_cmd(["experiments/coral_dev/run.py",
                 "--no-gui", "--steps", "150", "--seed", "42",
                 "--checkpoint-interval", "100"])
    if r.returncode != 0:
        fail("create test run", r.stderr)
        sys.exit(1)

    run_dir = newest_run_dir()
    if not run_dir:
        fail("find run dir")
        sys.exit(1)
    print(f"  Run dir: {run_dir}")

    ckpt_dirs = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*")))
    non_zero  = [c for c in ckpt_dirs if not c.endswith("checkpoint_0000000")]
    if not non_zero:
        fail("find non-zero checkpoint", f"found: {ckpt_dirs}")
        sys.exit(1)
    ckpt     = non_zero[0]
    snap_dir = os.path.join(run_dir, "snapshot")
    print(f"  Checkpoint: {os.path.basename(ckpt)}")
    ok("test run created with checkpoint")

    # ------------------------------------------------------------------
    # Step 2: Check git_hash in run meta.json
    # ------------------------------------------------------------------
    with open(os.path.join(run_dir, "meta.json")) as f:
        run_meta = json.load(f)
    if "git_hash" in run_meta and run_meta["git_hash"] not in ("", "unknown"):
        ok(f"run meta.json has git_hash: {run_meta['git_hash'][:12]}…")
    else:
        fail("run meta.json missing git_hash", str(run_meta))

    # ------------------------------------------------------------------
    # Step 3: Inject a breaking change into the LIVE physics.py
    # ------------------------------------------------------------------
    physics_path = os.path.join(REPO, "experiments", "coral_dev", "physics.py")
    with open(physics_path) as f:
        original_physics = f.read()

    sentinel = "SNAPSHOT_ISOLATION_TEST_SENTINEL"
    broken   = f'raise RuntimeError("{sentinel}: live physics.py was modified")\n'

    try:
        with open(physics_path, "w") as f:
            f.write(broken + original_physics)

        # ------------------------------------------------------------------
        # Step 4: Confirm the live code is actually broken
        # ------------------------------------------------------------------
        helper_live = os.path.join(REPO, "tests", "_tmp_live_step.py")
        with open(helper_live, "w") as hf:
            hf.write(f"""\
import sys, json, os
sys.path.insert(0, {REPO!r})
sys.path.insert(0, os.path.join({REPO!r}, "experiments", "coral_dev"))
import taichi as ti, torch
ti.init(ti.metal)
from coralai.evolver import apply_weights_and_biases
from experiment import EXPERIMENT as exp
with open(os.path.join({ckpt!r}, "meta.json")) as f:
    meta = json.load(f)
shape = tuple(meta["shape"])
substrate = exp.make_substrate(shape, torch.device("mps"))
evolver   = exp.make_evolver(substrate)
evolver.load_checkpoint({ckpt!r})
exp.run_physics(substrate, evolver)  # should raise
print("LIVE_OK")
""")
        r_live = subprocess.run([PY, helper_live],
                                capture_output=True, text=True, timeout=60)
        os.remove(helper_live)

        combined = r_live.stdout + r_live.stderr
        if sentinel in combined and "LIVE_OK" not in r_live.stdout:
            ok("live physics.py modification is visible (direct import raises)")
        else:
            fail("live physics.py change not detected — test may be invalid",
                 combined)

        # ------------------------------------------------------------------
        # Step 5: Snapshot replay must succeed despite broken live physics.py
        # ------------------------------------------------------------------
        helper_snap = os.path.join(REPO, "tests", "_tmp_snap_step.py")
        with open(helper_snap, "w") as hf:
            hf.write(f"""\
import sys, json, os
sys.path.insert(0, {REPO!r})
import taichi as ti, torch
ti.init(ti.metal)
from coralai.replay_utils import load_experiment_from_snapshot, discover_checkpoints
from coralai.evolver import apply_weights_and_biases

snap_dir = {snap_dir!r}
ckpt     = {ckpt!r}

# load_experiment_from_snapshot inserts snap_dir first in sys.path,
# so deferred 'from physics import ...' resolves to the snapshot copy.
exp = load_experiment_from_snapshot(snap_dir)

with open(os.path.join(ckpt, "meta.json")) as f:
    meta = json.load(f)
shape = tuple(meta["shape"])

substrate = exp.make_substrate(shape, torch.device("mps"))
evolver   = exp.make_evolver(substrate)
evolver.load_checkpoint(ckpt)

cw, cb = evolver.get_combined_weights()
out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                         cw, cb, evolver.dir_kernel, evolver.dir_order,
                         substrate.ti_indices, evolver._get_key_to_local())
substrate.mem[0, evolver.act_chinds] = out_mem
exp.run_physics(substrate, evolver)   # must use snapshot physics, not live
exp.run_evolution(substrate, evolver, evolver.timestep)
assert not torch.isnan(substrate.mem).any(), "NaN after one step"
print("SNAPSHOT_OK")
""")
        r_snap = subprocess.run([PY, helper_snap],
                                capture_output=True, text=True, timeout=120)
        os.remove(helper_snap)

        if "SNAPSHOT_OK" in r_snap.stdout:
            ok("snapshot replay succeeds despite live physics.py being broken")
        else:
            fail("snapshot replay failed", r_snap.stdout + r_snap.stderr)

    finally:
        # Always restore live physics.py — even if a test assertion fails
        with open(physics_path, "w") as f:
            f.write(original_physics)
        print("  (live physics.py restored)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  {PASS_COUNT} passed  {FAIL_COUNT} failed")
    print("=" * 60)
    print()
    print("  Note on framework code (coralai/):")
    print("  The snapshot captures experiment/ files only, not coralai/.")
    print(f"  To recover the framework version for this run:")
    print(f"    git checkout {run_meta.get('git_hash', '<hash>')} -- coralai/")
    print()
    sys.exit(0 if FAIL_COUNT == 0 else 1)
