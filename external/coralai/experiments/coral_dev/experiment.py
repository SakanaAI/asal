"""
CoralDevExperiment — development version of the coral spatial-NEAT experiment.

Bundles the channel layout, spatial kernel, and physics/evolution pipelines
so run.py and coralai/replay.py can reconstruct a run without hard-coding
any experiment-specific details.

The module-level EXPERIMENT instance is the canonical entry point for both
the runner and the generic replay tool:

    from experiment import EXPERIMENT as exp
    substrate = exp.make_substrate(shape, device)
    evolver   = exp.make_evolver(substrate)
    env       = exp.make_env("hole", param=0.35)
"""

import os

import taichi as ti
import torch

from coralai.experiment import Experiment
from coralai.visualization import Visualization

# Captured at module load time so make_evolver finds the right neat.config
# whether this file is run in-place or loaded from a snapshot.
_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Channel / kernel configuration
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy": ti.f32,
    "infra":  ti.f32,
    "acts": ti.types.struct(
        invest=ti.f32,
        liquidate=ti.f32,
        explore=ti.types.vector(n=4, dtype=ti.f32),  # no, fwd, left, right
    ),
    "com": ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
    "rot":    ti.f32,
    "genome": ti.f32,
}

KERNEL    = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
             [-1, 0], [-1, -1], [0, -1], [1, -1]]
DIR_ORDER = [0, -1, 1]
SENSE_CHS = ["energy", "infra", "com"]
ACT_CHS   = ["acts", "com"]


# ---------------------------------------------------------------------------
# CoralVis — experiment-specific GUI overlay
# ---------------------------------------------------------------------------

class CoralVis(Visualization):
    def __init__(self, substrate, evolver, vis_chs):
        super().__init__(substrate, vis_chs, panel_width=300)
        self.evolver = evolver
        self.genome_stats = []
        self.fps_history = []

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        px = self.panel_x
        pw = self.panel_wfrac
        n_chs = self.substrate.mem.shape[1]

        with self.gui.sub_window("Display", px, 0.01, pw, 0.46) as sw:
            self._draw_norm_controls(sw)
            if sw.button("E | I | Rot"):
                self.chinds[0] = int(inds.energy)
                self.chinds[1] = int(inds.infra)
                self.chinds[2] = int(inds.rot)
                self.view_mode = 0
            if sw.button("Genome | E | I"):
                self.chinds[0] = int(inds.genome)
                self.chinds[1] = int(inds.energy)
                self.chinds[2] = int(inds.infra)
                self.view_mode = 0
            if sw.button("Energy only"):
                self.chinds[0] = int(inds.energy)
                self.view_mode = 1
            if sw.button("Infra only"):
                self.chinds[1] = int(inds.infra)
                self.view_mode = 2
            if sw.button("Genome only"):
                self.chinds[0] = int(inds.genome)
                self.view_mode = 1
            self.chinds[0] = sw.slider_int(
                f"R: {self.substrate.index_to_chname(int(self.chinds[0]))}",
                int(self.chinds[0]), 0, n_chs - 1)
            self.chinds[1] = sw.slider_int(
                f"G: {self.substrate.index_to_chname(int(self.chinds[1]))}",
                int(self.chinds[1]), 0, n_chs - 1)
            self.chinds[2] = sw.slider_int(
                f"B: {self.substrate.index_to_chname(int(self.chinds[2]))}",
                int(self.chinds[2]), 0, n_chs - 1)
            self.paused = sw.checkbox("Pause", self.paused)

        with self.gui.sub_window("Stats", px, 0.47, pw, 0.52) as sw:
            pos = self.window.get_cursor_pos()
            sim_frac = self.sim_w / self.image.shape[0]
            cx = int((pos[0] / sim_frac) * self.w) % self.w
            cy = int(pos[1] * self.h) % self.h
            sw.text(
                f"GENOME: {self.substrate.mem[0, inds.genome, cx, cy]:.0f}\n"
                f"Energy: {self.substrate.mem[0, inds.energy, cx, cy]:.2f}\n"
                f"Infra:  {self.substrate.mem[0, inds.infra, cx, cy]:.2f}")
            sw.text(f"Step: {self.evolver.timestep}")
            tot_e = torch.sum(self.substrate.mem[0, inds.energy])
            tot_i = torch.sum(self.substrate.mem[0, inds.infra])
            sw.text(f"Total E+I: {tot_e + tot_i:.1f}")
            sw.text(f"Energy%: {100 * tot_e / (tot_e + tot_i):.1f}%")
            sw.text(f"Genomes: {len(self.evolver.genomes)}")
            if self.fps_history:
                sw.text(f"FPS: {self.fps_history[-1]:.1f}")
            if self.evolver.timestep % 20 == 0:
                self.genome_stats = []
                for i in range(len(self.evolver.genomes)):
                    n = self.substrate.mem[0, inds.genome].eq(i).sum().item()
                    self.genome_stats.append((i, n, self.evolver.ages[i]))
                self.genome_stats.sort(key=lambda x: x[1], reverse=True)
            for i, n_cells, age in self.genome_stats[:8]:
                sw.text(f"  G{i}: {n_cells:.0f}c  {age}t")


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class CoralDevExperiment(Experiment):
    name      = "coral_dev"
    channels  = CHANNELS
    kernel    = KERNEL
    dir_order = DIR_ORDER
    sense_chs = SENSE_CHS
    act_chs   = ACT_CHS
    _exp_dir  = _EXPERIMENT_DIR

    # Tunable physics parameters (overridable via run.py CLI args)
    infra_decay   = 0.0   # fraction of infra lost per step; 0 = disabled
    defense_coeff = 0.0   # infra-as-defense multiplier; 0 = disabled

    def make_env(self, env_name: str = "flat", param=None):
        from environments import make_env
        return make_env(env_name, param)

    def make_vis(self, substrate, evolver):
        return CoralVis(substrate, evolver, ["energy", "infra", "rot"])

    def run_physics(self, substrate, evolver):
        """Activate NN outputs → invest/liquidate → explore → energy flow → death."""
        from physics import (activate_outputs, invest_liquidate,
                             explore_physics, energy_physics)
        activate_outputs(substrate)
        invest_liquidate(substrate)
        explore_physics(substrate, evolver.kernel, evolver.dir_order,
                        defense_coeff=self.defense_coeff)
        energy_physics(substrate, evolver.kernel, max_infra=10, max_energy=1.5)

        inds = substrate.ti_indices[None]
        alive = (substrate.mem[0, inds.infra] + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)

    def run_evolution(self, substrate, evolver, step: int):
        """Age tracking, infra decay, and periodic chunk death.

        Energy injection is now handled entirely by the environment (patches).
        No global day/night cycle.
        """
        from evolution import kill_random_chunk
        evolver.ages = [a + 1 for a in evolver.ages]
        inds = substrate.ti_indices[None]
        if self.infra_decay > 0:
            substrate.mem[0, inds.infra] *= (1.0 - self.infra_decay)
        substrate.mem[0, inds.energy].clamp_(0.0, 100)
        substrate.mem[0, inds.infra].clamp_(0.0, 100)
        if step % 50 == 0:
            kill_random_chunk(evolver, 5)


# Module-level instance — imported by run.py and coralai/replay.py
EXPERIMENT = CoralDevExperiment()
