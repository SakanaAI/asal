"""
Template Experiment class.

Copy this file to your experiment directory and implement run_physics and
run_evolution. Export EXPERIMENT at module level so coralai/replay.py can
import it from the snapshot.

See experiments/coral_dev/experiment.py for a complete working example.
"""

import os

import taichi as ti
import torch

from coralai.experiment import Experiment

_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Define your channel layout
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy":  ti.f32,
    "infra":   ti.f32,
    "acts": ti.types.struct(
        invest=ti.f32,
        liquidate=ti.f32,
        explore=ti.types.vector(n=4, dtype=ti.f32),
    ),
    "com":     ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
    "rot":     ti.f32,
    "genome":  ti.f32,
}

# 2. Define the spatial kernel (neighbourhood shape)
KERNEL    = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]  # von Neumann
DIR_ORDER = [0, -1, 1]

# 3. Which channels the network reads and writes
SENSE_CHS = ["energy", "infra", "com"]
ACT_CHS   = ["acts", "com"]


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class TemplateExperiment(Experiment):
    name      = "template"
    channels  = CHANNELS
    kernel    = KERNEL
    dir_order = DIR_ORDER
    sense_chs = SENSE_CHS
    act_chs   = ACT_CHS
    _exp_dir  = _EXPERIMENT_DIR

    def run_physics(self, substrate, evolver):
        """Apply one step of physics. Import from your physics.py."""
        from physics import activate_outputs, step_physics
        activate_outputs(substrate)
        step_physics(substrate)

        inds = substrate.ti_indices[None]
        alive = (substrate.mem[0, inds.infra] + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)

    def run_evolution(self, substrate, evolver, step: int):
        """Apply one step of evolution: noise, ages, chunk death."""
        evolver.ages = [a + 1 for a in evolver.ages]
        inds = substrate.ti_indices[None]
        substrate.mem[0, inds.energy].add_(
            torch.randn_like(substrate.mem[0, inds.energy]).mul_(0.1))
        substrate.mem[0, inds.infra].add_(
            torch.randn_like(substrate.mem[0, inds.infra]).mul_(0.1))
        substrate.mem[0, inds.energy].clamp_(0.01, 100)
        substrate.mem[0, inds.infra].clamp_(0.01, 100)


# Module-level instance — imported by run.py and coralai/replay.py
EXPERIMENT = TemplateExperiment()
