"""
Coral experiment runner — DEVELOPMENT VERSION.

Active development branch. Diverges from the thesis version (experiments/coral/)
with physics fixes and ongoing improvements. Current changes vs. thesis:

  - softmax invest/liquidate → tanh signed trade (physics.py activate_outputs)
  - uniform day/night energy removed; patches are the sole energy source
  - infra-as-defense and infra decay available via CLI flags

Usage:
    python experiments/coral_dev/run.py --env patches
    python experiments/coral_dev/run.py --env patches --env-param 12
    python experiments/coral_dev/run.py --env patches --infra-decay 0.002 --defense-coeff 0.5
    python experiments/coral_dev/run.py --no-gui --steps 300 --profile
    python experiments/coral_dev/run.py --benchmark --steps 200 --shape 400
    python experiments/coral_dev/run.py --backend cpu --device cpu --no-gui --steps 100
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import torch
import taichi as ti

# ---------------------------------------------------------------------------
# Argument parsing (before ti.init so --help works without GPU init)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Coral Dev Experiment Runner")
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--shape", type=int, default=400)
parser.add_argument("--backend", type=str, default=("cuda" if torch.cuda.is_available() else ("metal" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")),
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")),
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--radiate-interval", type=int, default=50)
parser.add_argument("--cleanup-interval", type=int, default=100,
                    help="Steps between extinct-genome sweeps (removes 0-cell genomes only).")
parser.add_argument("--emergency-pop-cap", type=int, default=1000,
                    help="Hard cap: force-cull live genomes only if count exceeds this. "
                         "Set to 0 to disable entirely. Default 1000.")
parser.add_argument("--checkpoint-interval", type=int, default=1000,
                    help="Steps between checkpoints (0 = disable)")
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--env", type=str, default="flat",
                    choices=["flat", "hole", "ring", "stripes", "corners", "patches", "oases"])
parser.add_argument("--env-param", type=float, default=None)
parser.add_argument("--infra-decay", type=float, default=0.0,
                    help="Infra lost per step as a fraction (e.g. 0.002). 0=disabled.")
parser.add_argument("--defense-coeff", type=float, default=0.0,
                    help="Infra-as-defense multiplier in explore kernel. 0=disabled.")
parser.add_argument("--resume-from", type=str, default=None,
                    help="Path to a checkpoint dir; resume run from that state. "
                         "--steps counts additional steps from the checkpoint.")
args = parser.parse_args()

import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

backend_map = {"cpu": ti.cpu, "metal": ti.metal,
               "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.evolver import apply_weights_and_biases
from experiment import EXPERIMENT as exp

# Apply CLI-specified physics parameters to the experiment
exp.infra_decay   = args.infra_decay
exp.defense_coeff = args.defense_coeff


# ---------------------------------------------------------------------------
# Device sync helper
# ---------------------------------------------------------------------------
def sync():
    if args.device == "mps":
        torch.mps.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import json, shutil

    shape = (args.shape, args.shape)
    substrate = exp.make_substrate(shape, DEVICE)

    if args.benchmark:
        inds = substrate.ti_indices[None]
        substrate.mem[0, inds.energy] = torch.rand(shape, device=DEVICE)
        substrate.mem[0, inds.infra]  = torch.rand(shape, device=DEVICE)
        env = exp.make_env("flat")
    else:
        env = exp.make_env(args.env, args.env_param)
        env.seed(substrate)

    evolver = exp.make_evolver(substrate)

    # Load checkpoint if resuming; otherwise this is step 0
    if args.resume_from:
        evolver.load_checkpoint(args.resume_from)
        print(f"  [resume] loaded {args.resume_from}  (step {evolver.timestep})")
    start_step = evolver.timestep

    headless = args.no_gui or args.benchmark
    vis = None if headless else exp.make_vis(substrate, evolver)

    inds = substrate.ti_indices[None]
    timings = defaultdict(float)
    step_times = []
    fps_window = []
    fps_print_interval = 2.0
    last_fps_time = time.time()
    last_fps = 0.0
    t_session = time.time()
    # end_step is absolute; --steps counts additional steps from wherever we start
    end_step = start_step + (args.steps if args.steps > 0 else 10 ** 9)

    # ---- Run directory ---------------------------------------------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    run_dir = os.path.join(repo_root, "runs",
                           f"coral_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    shutil.copytree(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(run_dir, "snapshot"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    try:
        import subprocess as _sp
        _git_hash = _sp.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL,
            cwd=repo_root).decode().strip()
    except Exception:
        _git_hash = "unknown"

    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "experiment":   exp.name,
            "shape":        list(shape),
            "seed":         args.seed,
            "env":          env.to_dict(),
            "env_persist":  env.persist,
            "start_step":   start_step,
            "resumed_from": args.resume_from,
            "start_time":   datetime.now().isoformat(),
            "git_hash":     _git_hash,
            "infra_decay":       args.infra_decay,
            "defense_coeff":     args.defense_coeff,
            "cleanup_interval":  args.cleanup_interval,
            "emergency_pop_cap": args.emergency_pop_cap,
        }, f, indent=2)

    if not args.benchmark:
        torch.save(substrate.mem.cpu(), os.path.join(run_dir, "initial_state.pt"))

    log_path = os.path.join(run_dir, "step_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "total_energy", "total_infra", "energy_pct",
                         "n_alive", "n_genomes", "energy_offset", "fps"])
    # ---------------------------------------------------------------------------

    print(f"Run dir: {run_dir}")
    print(f"Coral-dev | {shape[0]}x{shape[1]} | {args.backend}/{args.device} | "
          f"env={args.env} persist={env.persist} | GUI={'OFF' if headless else 'ON'} | "
          f"{'benchmark' if args.benchmark else 'profile' if args.profile else 'run'}")
    if start_step > 0:
        print(f"  Resuming from step {start_step}")
    print("-" * 60)

    # Save step-0 checkpoint on fresh runs so replay can always start from the beginning
    if not args.benchmark and args.checkpoint_interval > 0 and start_step == 0:
        evolver.save_checkpoint(run_dir, 0)

    from evolution import apply_radiation_mutation

    for step in range(start_step, end_step):
        t_step = time.perf_counter()

        sync(); t0 = time.perf_counter()
        cw, cb = evolver.get_combined_weights()
        out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                 cw, cb, evolver.dir_kernel, evolver.dir_order,
                                 substrate.ti_indices, evolver._get_key_to_local())
        substrate.mem[0, evolver.act_chinds] = out_mem
        sync(); timings["0_nn_forward"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        exp.run_physics(substrate, evolver)
        sync(); timings["1_physics"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        exp.run_evolution(substrate, evolver, step)
        sync(); timings["2_evolution"] += time.perf_counter() - t0

        if env.persist:
            env.step(substrate)

        if vis:
            sync(); t0 = time.perf_counter()
            vis.update()
            sync(); timings["3_render"] += time.perf_counter() - t0

        if step % args.radiate_interval == 0 and step > 0:
            sync(); t0 = time.perf_counter()
            apply_radiation_mutation(evolver, 5)
            sync(); timings["4_radiation"] += time.perf_counter() - t0

        # Periodic extinct-only cleanup: remove genomes with 0 cells (never kills live ones)
        if step % args.cleanup_interval == 0 and step > 0:
            evolver.remove_extinct_genomes()

        # Emergency hard cap: only fires if natural extinction isn't keeping up
        if (args.emergency_pop_cap > 0
                and len(evolver.genomes) > args.emergency_pop_cap
                and (step - evolver.time_last_cull) > 50):
            print(f"  [emergency] population {len(evolver.genomes)} > cap {args.emergency_pop_cap}")
            evolver.reduce_population_to_threshold(args.emergency_pop_cap)

        evolver.timestep = step + 1
        dt = time.perf_counter() - t_step
        step_times.append(dt)

        now = time.time()
        fps_window.append(dt)
        if now - last_fps_time >= fps_print_interval:
            last_fps = len(fps_window) / sum(fps_window)
            alive_n = (substrate.mem[0, inds.genome] >= 0).sum().item()
            if vis:
                vis.fps_history.append(last_fps)
            print(f"  step {step+1:5d} | {last_fps:5.1f} FPS | "
                  f"{1000*sum(fps_window)/len(fps_window):5.1f}ms/step | "
                  f"alive: {alive_n:,} | genomes: {len(evolver.genomes)}")
            fps_window = []
            last_fps_time = now

        if not args.benchmark and step % args.log_interval == 0:
            tot_e = float(substrate.mem[0, inds.energy].sum())
            tot_i = float(substrate.mem[0, inds.infra].sum())
            alive_n = int((substrate.mem[0, inds.genome] >= 0).sum())
            log_writer.writerow([
                step, f"{tot_e:.4f}", f"{tot_i:.4f}",
                f"{100 * tot_e / (tot_e + tot_i + 1e-8):.2f}",
                alive_n, len(evolver.genomes),
                f"{evolver.energy_offset:.4f}", f"{last_fps:.2f}",
            ])
            log_file.flush()

        if (not args.benchmark and args.checkpoint_interval > 0
                and step > 0 and step % args.checkpoint_interval == 0):
            evolver.save_checkpoint(run_dir, step)

        if vis and not vis.window.running:
            break

    log_file.close()

    total = time.time() - t_session
    n = len(step_times)
    avg_fps = n / sum(step_times)
    median_ms = sorted(step_times)[n // 2] * 1000
    alive_final = (substrate.mem[0, inds.genome] >= 0).sum().item()

    print(f"\n{'='*60}")
    print(f"  {n} steps in {total:.2f}s  |  avg {avg_fps:.1f} FPS  |  "
          f"median {median_ms:.1f}ms/step")
    print(f"  alive: {alive_final:,}  |  genomes: {len(evolver.genomes)}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    if args.profile or args.benchmark:
        total_p = sum(timings.values())
        print("\n  Per-category breakdown:")
        for name, t in sorted(timings.items()):
            pct = 100 * t / total_p if total_p > 0 else 0
            print(f"    {name:25s}  {1000*t/n:6.2f}ms/step  {pct:5.1f}%")
        print(f"    {'TOTAL':25s}  {1000*total_p/n:6.2f}ms/step")


if __name__ == "__main__":
    main()
