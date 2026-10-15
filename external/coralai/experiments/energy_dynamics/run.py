"""
Energy Dynamics experiment.

Demonstrates pure energy-flow physics (no organisms, no NEAT) across six
environment types and five physics modes, all switchable live in the GUI.

The GUI is split: simulation on the left, control panels on the right.
Normalization and view modes live in the base Visualization class and are
available to every experiment — this file only adds the experiment-specific
Physics and Environment panels on top.

Usage:
    python experiments/energy_dynamics/run.py
    python experiments/energy_dynamics/run.py --env gradient --shape 300
    python experiments/energy_dynamics/run.py --no-gui --steps 200
"""

import os
import sys
import time
import argparse

import torch
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="patch",
                    choices=["patch", "gradient", "ring", "rivers", "volcano", "random"])
parser.add_argument("--shape", type=int, default=200)
parser.add_argument("--backend", type=str, default=("cuda" if torch.cuda.is_available() else ("metal" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")),
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")),
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true",
                    help="Headless smoke-test: run --steps then exit")
parser.add_argument("--steps", type=int, default=0,
                    help="Steps before exit (0 = run until window closed)")
args = parser.parse_args()

ti.init({"cpu": ti.cpu, "metal": ti.metal,
         "cuda": ti.cuda, "vulkan": ti.vulkan}[args.backend])
DEVICE = torch.device(args.device)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coralai.substrate import Substrate          # noqa: E402
from coralai.visualization import Visualization  # noqa: E402
from physics import (                            # noqa: E402
    init_env, inject_energy,
    physics_infra_attract, physics_infra_attract_cap,
    physics_diffuse, physics_infra_repel, physics_infra_decay,
)

# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy": ti.f32,
    "infra":  ti.f32,
    "source": ti.f32,
}

KERNEL = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]

# ---------------------------------------------------------------------------
# Physics / environment registries
# ---------------------------------------------------------------------------
PHYSICS_MODES = [
    ("infra_attract",     "Infra Attract"),
    ("infra_attract_cap", "Infra + Cap"),
    ("diffuse",           "Equalize"),
    ("infra_repel",       "Infra Repel"),
    ("infra_decay",       "Infra Decay"),
]

PHYSICS_FNS = {
    "infra_attract":     physics_infra_attract,
    "infra_attract_cap": physics_infra_attract_cap,
    "diffuse":           physics_diffuse,
    "infra_repel":       physics_infra_repel,
    "infra_decay":       physics_infra_decay,
}

ENVS = [
    ("patch",    "Patches"),
    ("gradient", "Gradient"),
    ("ring",     "Ring"),
    ("rivers",   "Rivers"),
    ("volcano",  "Volcano"),
    ("random",   "Random"),
]


# ---------------------------------------------------------------------------
# Visualization subclass — adds Physics + Environment panels
# ---------------------------------------------------------------------------
class EnergyVis(Visualization):
    """
    Extends Visualization with experiment-specific control panels.
    View/norm modes come from the base class for free.
    """

    def __init__(self, substrate, channels):
        super().__init__(substrate, channels, panel_width=300)

        # Physics / environment state (read by the main loop each frame)
        self.current_physics = "infra_attract_cap"
        self.current_env     = args.env
        self.inject_rate     = 0.02
        self.max_energy      = 1.0
        self._pending_env    = None   # set by GUI, applied before next step

    def render_opt_window(self):
        """Three non-overlapping panels: Display | Physics & Controls | Environment."""
        px = self.panel_x
        pw = self.panel_wfrac

        # -- Panel 1: Display (norm + view) — y=0.01..0.34 ------------------
        with self.gui.sub_window("Display", px, 0.01, pw, 0.34) as sw:
            self._norm_view_window(sw)

        # -- Panel 2: Physics & Controls — y=0.36..0.73 ----------------------
        with self.gui.sub_window("Physics & Controls", px, 0.36, pw, 0.38) as sw:
            for key, label in PHYSICS_MODES:
                marker = "* " if self.current_physics == key else "  "
                if sw.button(marker + label):
                    self.current_physics = key

            self.inject_rate = sw.slider_float(
                "Inject rate", self.inject_rate, 0.0, 0.1)
            self.max_energy  = sw.slider_float(
                "Max energy",  self.max_energy,  0.05, 5.0)

            ch_name = (self.chids[self.channel_to_paint]
                       if self.channel_to_paint < len(self.chids)
                       else str(self.channel_to_paint))
            self.channel_to_paint = sw.slider_int(
                f"Paint ch: {ch_name}", self.channel_to_paint,
                0, self.substrate.mem.shape[1] - 1)
            self.val_to_paint = sw.slider_float(
                "Paint value", self.val_to_paint, -1.0, 1.0)
            self.val_to_paint = round(self.val_to_paint * 10) / 10
            self.brush_radius = sw.slider_int(
                "Brush radius", self.brush_radius, 1, 200)
            self.paused = sw.checkbox("Pause", self.paused)

        # -- Panel 3: Environment — y=0.75..0.99 -----------------------------
        with self.gui.sub_window("Environment", px, 0.75, pw, 0.24) as sw:
            for key, label in ENVS:
                marker = "* " if self.current_env == key else "  "
                if sw.button(marker + label):
                    self._pending_env = key
            if sw.button("Reset current"):
                self._pending_env = self.current_env
            if sw.button("Clear energy"):
                inds = self.substrate.ti_indices[None]
                self.substrate.mem[0, inds.energy] *= 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    shape = (args.shape, args.shape)
    substrate = Substrate(shape, torch.float32, DEVICE, CHANNELS)
    substrate.malloc()
    kernel = torch.tensor(KERNEL, dtype=torch.int32, device=DEVICE)

    inds      = substrate.ti_indices[None]
    ch_energy = int(inds.energy)

    # ---- Headless smoke-test ------------------------------------------------
    if args.no_gui:
        init_env(substrate, args.env)
        n           = args.steps if args.steps > 0 else 200
        inject_rate = 0.02
        max_energy  = 1.0
        t0 = time.time()
        for step in range(n):
            inject_energy(substrate, inject_rate)
            PHYSICS_FNS["infra_attract_cap"](substrate, kernel, max_energy)
            if step % 50 == 0:
                e = substrate.mem[0, ch_energy]
                print(f"step {step:4d}  energy: mean={e.mean():.4f}  max={e.max():.4f}")
        elapsed = time.time() - t0
        print(f"\n{n} steps in {elapsed:.2f}s ({n / elapsed:.0f} steps/s)")
        return

    # ---- GUI ----------------------------------------------------------------
    vis = EnergyVis(substrate, ["energy", "infra", "source"])
    init_env(substrate, vis.current_env)

    max_steps = args.steps if args.steps > 0 else 10 ** 9
    step = 0

    while vis.window.running and step < max_steps:

        # Apply any environment change queued by the GUI last frame
        if vis._pending_env is not None:
            vis.current_env  = vis._pending_env
            vis._pending_env = None
            init_env(substrate, vis.current_env)

        if not vis.paused:
            inject_energy(substrate, vis.inject_rate)
            PHYSICS_FNS[vis.current_physics](substrate, kernel, vis.max_energy)
            step += 1

        vis.update()

    if args.steps > 0:
        print(f"Completed {step} steps.")


if __name__ == "__main__":
    main()
