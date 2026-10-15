"""
coralai/replay.py — generic checkpoint replay tool.

Loads the Experiment class from a run's snapshot directory so replay always
uses the exact physics and evolution code that generated the run. Works with
any experiment that follows the Experiment base-class interface.

Usage:
    # Open GUI at latest checkpoint of a run:
    python -m coralai.replay --run-dir runs/coral_dev_20260228_174936

    # Start at a specific checkpoint step:
    python -m coralai.replay --run-dir runs/coral_dev_... --step 2000

    # CPU backend for bitwise-identical replay:
    python -m coralai.replay --run-dir runs/coral_dev_... \\
        --backend cpu --device cpu
"""

import argparse
import json
import os
import sys
import time

import torch
import taichi as ti

# ---------------------------------------------------------------------------
# Argument parsing — before ti.init so --help works without GPU init
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CoralAI generic checkpoint replay")
parser.add_argument("--run-dir", type=str, required=True,
                    help="Path to runs/experiment_TIMESTAMP/ directory")
parser.add_argument("--step", type=int, default=None,
                    help="Checkpoint step to load (default: latest)")
parser.add_argument("--backend", type=str, default="metal",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "mps", "cuda"])
args = parser.parse_args()

backend_map = {"cpu": ti.cpu, "metal": ti.metal,
               "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.evolver import apply_weights_and_biases
from coralai.visualization import Visualization
from coralai.replay_utils import load_experiment_from_snapshot, discover_checkpoints


# ---------------------------------------------------------------------------
# GenericReplayVis
# ---------------------------------------------------------------------------

class GenericReplayVis(Visualization):
    """Replay visualisation with checkpoint navigation and generic stats panel.

    Experiment-agnostic: no hard-coded channel button presets. Channel slots
    can be changed via sliders. The stats panel shows evolver state only.
    """

    def __init__(self, substrate, evolver, vis_chs, run_dir, ckpt_dirs, ckpt_idx):
        super().__init__(substrate, vis_chs, panel_width=300)
        self.evolver     = evolver
        self.run_dir     = run_dir
        self.ckpt_dirs   = ckpt_dirs
        self.ckpt_idx    = ckpt_idx
        self.fps_history = []
        self.genome_stats = []
        self.running      = False   # start paused
        self._load_pending = None

    def load_ckpt(self, idx):
        """Queue a checkpoint load — executed between frames to avoid mid-render state."""
        self._load_pending = idx

    def render_opt_window(self):
        inds  = self.substrate.ti_indices[None]
        px    = self.panel_x
        pw    = self.panel_wfrac
        n_chs = self.substrate.mem.shape[1]
        run_name = os.path.basename(self.run_dir)

        # -- Panel 1: Replay navigator — y=0.01..0.20 ------------------------
        with self.gui.sub_window("Replay", px, 0.01, pw, 0.20) as sw:
            label = run_name if len(run_name) <= 28 else "..." + run_name[-25:]
            sw.text(label)
            sw.text(f"Ckpt {self.ckpt_idx+1}/{len(self.ckpt_dirs)}: "
                    f"step {self.evolver.timestep:,}")
            if sw.button("Reset (step 0)"):
                self.load_ckpt(0)
            if sw.button("< Prev") and self.ckpt_idx > 0:
                self.load_ckpt(self.ckpt_idx - 1)
            if sw.button("> Next") and self.ckpt_idx < len(self.ckpt_dirs) - 1:
                self.load_ckpt(self.ckpt_idx + 1)
            self.running = not sw.checkbox("Paused", not self.running)

        # -- Panel 2: Display (sliders only) — y=0.22..0.52 ------------------
        with self.gui.sub_window("Display", px, 0.22, pw, 0.31) as sw:
            self._draw_norm_controls(sw)
            self.chinds[0] = sw.slider_int(
                f"R: {self.substrate.index_to_chname(int(self.chinds[0]))}",
                int(self.chinds[0]), 0, n_chs - 1)
            self.chinds[1] = sw.slider_int(
                f"G: {self.substrate.index_to_chname(int(self.chinds[1]))}",
                int(self.chinds[1]), 0, n_chs - 1)
            self.chinds[2] = sw.slider_int(
                f"B: {self.substrate.index_to_chname(int(self.chinds[2]))}",
                int(self.chinds[2]), 0, n_chs - 1)
            self.paused = sw.checkbox("Pause render", self.paused)

        # -- Panel 3: Stats — y=0.54..0.99 ------------------------------------
        with self.gui.sub_window("Stats", px, 0.54, pw, 0.45) as sw:
            # Cursor-position readout (experiment-agnostic)
            pos = self.window.get_cursor_pos()
            sim_frac = self.sim_w / self.image.shape[0]
            cx = int((pos[0] / sim_frac) * self.w) % self.w
            cy = int(pos[1] * self.h) % self.h
            genome_val = self.substrate.mem[0, inds.genome, cx, cy]
            sw.text(f"Cursor ({cx},{cy})\nGenome: {genome_val:.0f}")

            sw.text(f"Step: {self.evolver.timestep}")
            sw.text(f"E offset: {self.evolver.energy_offset:.3f}")
            sw.text(f"Genomes: {len(self.evolver.genomes)}")
            if self.fps_history:
                sw.text(f"FPS: {self.fps_history[-1]:.1f}")

            # Genome cell-count leaderboard (refreshed every 20 steps)
            if self.evolver.timestep % 20 == 0:
                self.genome_stats = []
                for i in range(len(self.evolver.genomes)):
                    n = self.substrate.mem[0, inds.genome].eq(i).sum().item()
                    self.genome_stats.append((i, n, self.evolver.ages[i]))
                self.genome_stats.sort(key=lambda x: x[1], reverse=True)
            for i, n_cells, age in self.genome_stats[:6]:
                sw.text(f"  G{i}: {n_cells:.0f}c  {age}t")


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
    run_dir = args.run_dir

    ckpt_dirs = discover_checkpoints(run_dir)
    if not ckpt_dirs:
        print(f"ERROR: No checkpoints found in {run_dir}")
        sys.exit(1)

    # Determine starting checkpoint
    if args.step is not None:
        target = f"checkpoint_{args.step:07d}"
        matching = [d for d in ckpt_dirs if os.path.basename(d) == target]
        if not matching:
            available = [os.path.basename(d) for d in ckpt_dirs]
            print(f"ERROR: step {args.step} not found. Available: {available}")
            sys.exit(1)
        start_idx = ckpt_dirs.index(matching[0])
    else:
        start_idx = len(ckpt_dirs) - 1  # latest

    print(f"  [replay] {len(ckpt_dirs)} checkpoint(s) in {run_dir}")
    print(f"  [replay] Starting at: {os.path.basename(ckpt_dirs[start_idx])}")

    # Read shape from first checkpoint's meta.json
    with open(os.path.join(ckpt_dirs[0], "meta.json")) as f:
        meta = json.load(f)
    shape = tuple(meta["shape"])

    # Load experiment from snapshot
    snapshot_dir = os.path.join(run_dir, "snapshot")
    exp = load_experiment_from_snapshot(snapshot_dir)
    print(f"  [replay] Experiment: {exp.name!r} from {snapshot_dir}")

    substrate = exp.make_substrate(shape, DEVICE)
    evolver   = exp.make_evolver(substrate)

    evolver.load_checkpoint(ckpt_dirs[start_idx])

    # Use first 3 sense channels for default RGB display
    vis_chs = exp.sense_chs[:3]
    vis = GenericReplayVis(substrate, evolver, vis_chs,
                           run_dir, ckpt_dirs, start_idx)

    inds = substrate.ti_indices[None]
    step_times = []
    fps_window = []
    fps_print_interval = 2.0
    last_fps_time = time.time()

    print(f"\nReplay | {shape[0]}x{shape[1]} | {args.backend}/{args.device}")
    print(f"  Run : {run_dir}")
    print(f"  Ckpt: {os.path.basename(ckpt_dirs[start_idx])}")
    print("-" * 60)
    print("  < Prev / > Next to navigate checkpoints")
    print("  Uncheck 'Paused' in the Replay panel to run forward")
    print("-" * 60)

    vis.running = False
    step = evolver.timestep

    while vis.window.running:
        # Apply any pending checkpoint load before physics
        if vis._load_pending is not None:
            new_idx = vis._load_pending
            vis._load_pending = None
            evolver.load_checkpoint(ckpt_dirs[new_idx])
            vis.ckpt_idx = new_idx
            step = evolver.timestep
            print(f"  [loaded] {os.path.basename(ckpt_dirs[new_idx])}")

        if vis.running:
            t_step = time.perf_counter()

            sync()
            cw, cb = evolver.get_combined_weights()
            out_mem = evolver._get_scratch("out_mem",
                                           substrate.mem[0, evolver.act_chinds])
            apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                     cw, cb, evolver.dir_kernel, evolver.dir_order,
                                     substrate.ti_indices, evolver._get_key_to_local())
            substrate.mem[0, evolver.act_chinds] = out_mem
            sync()

            exp.run_physics(substrate, evolver)
            exp.run_evolution(substrate, evolver, step)

            # Radiation on the same schedule as training (every 50 steps)
            from evolution import apply_radiation_mutation
            if step % 50 == 0 and step > 0:
                apply_radiation_mutation(evolver, 5)

            evolver.timestep = step + 1
            step += 1
            dt = time.perf_counter() - t_step
            step_times.append(dt)
            fps_window.append(dt)

            now = time.time()
            if now - last_fps_time >= fps_print_interval:
                fps = len(fps_window) / sum(fps_window)
                alive_n = (substrate.mem[0, inds.genome] >= 0).sum().item()
                vis.fps_history.append(fps)
                print(f"  step {step:5d} | {fps:5.1f} FPS | "
                      f"alive: {alive_n:,} | genomes: {len(evolver.genomes)}")
                fps_window = []
                last_fps_time = now

        vis.update()


if __name__ == "__main__":
    main()
