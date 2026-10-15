"""
Smoke tests for the Experiment / StartEnvironment / replay framework.

Run from the repo root:
    python tests/smoke_test.py

Each test prints PASS or FAIL. Exit code 0 only if all pass.
Tests are ordered: unit tests first, then integration (each creates its own
run dir and passes it explicitly — no ambiguous "latest dir" lookups).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(label):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS  {label}")


def fail(label, detail=""):
    global FAIL_COUNT
    FAIL_COUNT += 1
    lines = [f"  FAIL  {label}"]
    if detail:
        for line in detail.strip().splitlines()[-10:]:
            lines.append(f"        {line}")
    print("\n".join(lines))


def run_cmd(args, timeout=300):
    """Run a command from repo root; return CompletedProcess."""
    return subprocess.run([PY] + args, capture_output=True, text=True,
                         cwd=REPO, timeout=timeout)


def newest_run_dir(prefix="coral_dev"):
    """Return the most recently created runs/prefix_* directory."""
    dirs = glob.glob(os.path.join(REPO, "runs", f"{prefix}_*"))
    # filter to actual run dirs (not snapshot subdirs)
    dirs = [d for d in dirs if re.search(r"\d{8}_\d{6}$", d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


# ---------------------------------------------------------------------------
# 1. Unit tests — no Taichi
# ---------------------------------------------------------------------------

def test_syntax():
    import ast
    files = [
        "coralai/environment.py",
        "coralai/experiment.py",
        "coralai/replay_utils.py",
        "coralai/replay.py",
        "experiments/coral_dev/environments.py",
        "experiments/coral_dev/experiment.py",
        "experiments/coral_dev/run.py",
        "experiments/coral/environments.py",
        "experiments/coral/experiment.py",
        "experiments/coral/run.py",
        "experiments/_template/experiment.py",
        "experiments/_template/environments.py",
    ]
    bad = []
    for rel in files:
        try:
            with open(os.path.join(REPO, rel)) as f:
                ast.parse(f.read())
        except SyntaxError as e:
            bad.append(f"{rel}: {e}")
    if bad:
        fail("syntax", "\n".join(bad))
    else:
        ok(f"syntax — {len(files)} files parse cleanly")


def test_replay_utils_importable():
    sys.path.insert(0, REPO)
    try:
        from coralai.replay_utils import discover_checkpoints, load_experiment_from_snapshot
        ok("coralai.replay_utils importable without Taichi")
    except Exception as e:
        fail("coralai.replay_utils import", str(e))
    finally:
        sys.path.pop(0)


def test_environment_base():
    sys.path.insert(0, REPO)
    try:
        from coralai.environment import StartEnvironment
        env = StartEnvironment()
        assert env.name == "flat"
        assert env.persist is False
        assert env.to_dict() == {"name": "flat"}
        ok("StartEnvironment base class (no Taichi)")
    except Exception as e:
        fail("StartEnvironment base class", str(e))
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# 2. CLI integration tests
# ---------------------------------------------------------------------------

def test_basic_run():
    """Run 60 steps headless; verify run dir has required files."""
    r = run_cmd(["experiments/coral_dev/run.py",
                 "--no-gui", "--steps", "60", "--seed", "42",
                 "--checkpoint-interval", "0"])
    if r.returncode != 0:
        fail("basic headless run", r.stderr)
        return None

    run_dir = newest_run_dir()
    missing = [f for f in ("meta.json", "snapshot", "step_log.csv", "initial_state.pt")
               if not os.path.exists(os.path.join(run_dir, f))]
    if missing:
        fail("basic run — missing files", str(missing))
        return None

    ok("basic headless run — run dir with meta.json + snapshot + CSV + initial_state.pt")
    return run_dir


def test_run_meta_json(run_dir):
    if run_dir is None:
        fail("run meta.json — skipped (no run dir)")
        return
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    for key in ("experiment", "shape", "seed", "env", "env_persist", "start_time"):
        if key not in meta:
            fail(f"run meta.json — missing key {key!r}")
            return
    if meta["experiment"] != "coral_dev":
        fail(f"run meta.json — experiment={meta['experiment']!r}, expected 'coral_dev'")
        return
    ok("run meta.json — all expected keys present")


def test_snapshot_contents(run_dir):
    if run_dir is None:
        fail("snapshot contents — skipped")
        return
    snap = os.path.join(run_dir, "snapshot")
    required = ("experiment.py", "environments.py", "physics.py", "evolution.py", "neat.config")
    missing = [f for f in required if not os.path.exists(os.path.join(snap, f))]
    if missing:
        fail("snapshot contents", f"missing: {missing}")
    else:
        ok(f"snapshot contains {', '.join(required)}")


def test_hole_env():
    """--env hole: should run clean and report persist=True in header."""
    r = run_cmd(["experiments/coral_dev/run.py",
                 "--no-gui", "--steps", "60", "--seed", "42",
                 "--checkpoint-interval", "0",
                 "--env", "hole", "--env-param", "0.35"])
    if r.returncode != 0:
        fail("hole env", r.stderr)
    elif "persist=True" not in r.stdout:
        fail("hole env — persist=True not in output", r.stdout)
    else:
        ok("hole env — exits 0, persist=True in header")


def test_checkpointing():
    """600-step run with checkpoint at 500; verify checkpoint files."""
    r = run_cmd(["experiments/coral_dev/run.py",
                 "--no-gui", "--steps", "600", "--seed", "42",
                 "--checkpoint-interval", "500"], timeout=300)
    if r.returncode != 0:
        fail("checkpoint run", r.stderr)
        return None

    run_dir = newest_run_dir()
    ckpt_dirs = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*")))
    if not ckpt_dirs:
        fail("checkpoint run — no checkpoint dirs found")
        return None

    for ckpt in ckpt_dirs:
        missing = [f for f in ("substrate.pt", "population.pkl", "meta.json")
                   if not os.path.exists(os.path.join(ckpt, f))]
        if missing:
            fail(f"{os.path.basename(ckpt)} — missing: {missing}")
            return None

    ok(f"checkpointing — {len(ckpt_dirs)} checkpoint(s), each has substrate + population + meta")
    return run_dir


def test_load_from_snapshot(run_dir):
    """load_experiment_from_snapshot returns EXPERIMENT with correct attrs."""
    if run_dir is None:
        fail("load_from_snapshot — skipped")
        return
    snap = os.path.join(run_dir, "snapshot")
    sys.path.insert(0, REPO)
    try:
        from coralai.replay_utils import load_experiment_from_snapshot
        exp = load_experiment_from_snapshot(snap)
        assert exp.name == "coral_dev", f"name={exp.name!r}"
        assert list(exp.channels.keys()) == ["energy", "infra", "acts", "com", "rot", "genome"]
        assert hasattr(exp, "_exp_dir"), "missing _exp_dir"
        ok("load_experiment_from_snapshot — correct name, channels, _exp_dir")
    except Exception as e:
        fail("load_experiment_from_snapshot", str(e))
    finally:
        sys.path.pop(0)


def test_replay_one_step(run_dir):
    """Load checkpoint, run one physics+evolution step via snapshot code; no NaN."""
    if run_dir is None:
        fail("replay one step — skipped")
        return

    ckpt_dirs = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*")))
    if not ckpt_dirs:
        fail("replay one step — no checkpoints in run dir")
        return

    # Write a helper script so Taichi initialises in a fresh subprocess
    helper = os.path.join(REPO, "tests", "_tmp_replay_step.py")
    try:
        with open(helper, "w") as f:
            f.write(f"""\
import sys, json, os
sys.path.insert(0, {REPO!r})

import taichi as ti, torch
_dev = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
_bk = {{"cuda": ti.cuda, "mps": ti.metal, "cpu": ti.cpu}}[_dev]
ti.init(_bk)

from coralai.replay_utils import load_experiment_from_snapshot, discover_checkpoints
from coralai.evolver import apply_weights_and_biases

run_dir  = {run_dir!r}
snap_dir = os.path.join(run_dir, "snapshot")
ckpts    = discover_checkpoints(run_dir)
assert ckpts, "no checkpoints found"

exp = load_experiment_from_snapshot(snap_dir)

with open(os.path.join(ckpts[0], "meta.json")) as f:
    meta = json.load(f)
shape = tuple(meta["shape"])

substrate = exp.make_substrate(shape, torch.device(_dev))
evolver   = exp.make_evolver(substrate)
evolver.load_checkpoint(ckpts[0])

cw, cb = evolver.get_combined_weights()
out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                         cw, cb, evolver.dir_kernel, evolver.dir_order,
                         substrate.ti_indices, evolver._get_key_to_local())
substrate.mem[0, evolver.act_chinds] = out_mem
exp.run_physics(substrate, evolver)
exp.run_evolution(substrate, evolver, evolver.timestep)

assert not torch.isnan(substrate.mem).any(), "NaN after one step"
print("OK")
""")
        r = subprocess.run([PY, helper], capture_output=True, text=True,
                           cwd=REPO, timeout=120)
        if r.returncode != 0 or "OK" not in r.stdout:
            fail("replay one step", r.stdout + r.stderr)
        else:
            ok("replay one step — load checkpoint → run_physics + run_evolution → no NaN")
    finally:
        if os.path.exists(helper):
            os.remove(helper)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CoralAI Framework Smoke Tests")
    print("=" * 60)

    # Unit tests (no Taichi)
    test_syntax()
    test_replay_utils_importable()
    test_environment_base()

    # Integration: basic run (creates run_dir_a)
    run_dir_a = test_basic_run()
    test_run_meta_json(run_dir_a)
    test_snapshot_contents(run_dir_a)

    # Integration: environments
    test_hole_env()

    # Integration: checkpointing (creates run_dir_b with checkpoints)
    run_dir_b = test_checkpointing()
    test_load_from_snapshot(run_dir_b)
    test_replay_one_step(run_dir_b)

    print("=" * 60)
    print(f"  {PASS_COUNT} passed  {FAIL_COUNT} failed")
    print("=" * 60)
    sys.exit(0 if FAIL_COUNT == 0 else 1)
