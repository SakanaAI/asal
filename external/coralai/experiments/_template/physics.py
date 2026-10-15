"""
Template physics pipeline.

Copy this file to your experiment directory and define the physics that
constitute your research question. The functions here are called each step
by your run.py.

All physics functions receive a Substrate instance and read/write
substrate.mem (shape [1, C, W, H]) via substrate.ti_indices[None].
"""

import torch
import taichi as ti

from coralai.nn_lib import ch_norm


def activate_outputs(substrate):
    """Normalize raw network outputs into valid action values."""
    inds = substrate.ti_indices[None]
    mem = substrate.mem
    # Example: apply sigmoid to all output channels
    # Replace with your own activation logic.
    raise NotImplementedError("Define activate_outputs for your experiment.")


def step_physics(substrate):
    """Apply one step of physics to the substrate.

    Typically: energy flow, colonization, death checks, etc.
    Replace with your own physics logic.
    """
    raise NotImplementedError("Define step_physics for your experiment.")
