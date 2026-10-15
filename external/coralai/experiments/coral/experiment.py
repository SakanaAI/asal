"""
CoralExperiment — thesis version (frozen, 2024).

Physics and channel layout match the Master's Thesis exactly. Do not modify
for new experiments — use experiments/coral_dev/ instead.
"""

import os

import taichi as ti
import torch

from coralai.experiment import Experiment

_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Channel / kernel configuration (thesis frozen)
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy": ti.f32,
    "infra":  ti.f32,
    "acts": ti.types.struct(
        invest=ti.f32,
        liquidate=ti.f32,
        explore=ti.types.vector(n=4, dtype=ti.f32),
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


class CoralExperiment(Experiment):
    name      = "coral"
    channels  = CHANNELS
    kernel    = KERNEL
    dir_order = DIR_ORDER
    sense_chs = SENSE_CHS
    act_chs   = ACT_CHS
    _exp_dir  = _EXPERIMENT_DIR

    def make_env(self, env_name: str = "flat", param=None):
        from environments import make_env
        return make_env(env_name, param)

    def make_vis(self, substrate, evolver):
        from run import CoralVis
        return CoralVis(substrate, evolver, ["energy", "infra", "rot"])

    def run_physics(self, substrate, evolver):
        from physics import (activate_outputs, invest_liquidate,
                             explore_physics, energy_physics)
        activate_outputs(substrate)
        invest_liquidate(substrate)
        explore_physics(substrate, evolver.kernel, evolver.dir_order)
        energy_physics(substrate, evolver.kernel, max_infra=10, max_energy=1.5)

        inds = substrate.ti_indices[None]
        alive = (substrate.mem[0, inds.infra] + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)

    def run_evolution(self, substrate, evolver, step: int):
        from evolution import kill_random_chunk, get_energy_offset
        evolver.energy_offset = get_energy_offset(step)
        evolver.ages = [a + 1 for a in evolver.ages]
        inds = substrate.ti_indices[None]
        offset = evolver.energy_offset
        substrate.mem[0, inds.energy].add_(
            torch.randn_like(substrate.mem[0, inds.energy]).add_(offset).mul_(0.1))
        substrate.mem[0, inds.infra].add_(
            torch.randn_like(substrate.mem[0, inds.infra]).add_(offset).mul_(0.1))
        substrate.mem[0, inds.energy].clamp_(0.01, 100)
        substrate.mem[0, inds.infra].clamp_(0.01, 100)
        if step % 50 == 0:
            kill_random_chunk(evolver, 5)


# Module-level instance
EXPERIMENT = CoralExperiment()
