"""
Template experiment runner.

Copy this directory to experiments/<your_experiment>/, then:
1. Define your channel layout in experiment.py (CHANNELS, KERNEL, etc.)
2. Implement run_physics() and run_evolution() in experiment.py
3. Implement activate_outputs() and step_physics() in physics.py
4. Tune neat.config for your experiment
5. Run: python experiments/<your_experiment>/run.py --no-gui --steps 100

See experiments/coral_dev/ for a complete working example.
"""

import os
import time
import argparse

import torch
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=0,
                    help="Steps to run then exit (0 = run until window closed)")
parser.add_argument("--shape", type=int, default=100)
parser.add_argument("--backend", type=str, default=("cuda" if torch.cuda.is_available() else ("metal" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")),
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")),
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

ti.init({"cpu": ti.cpu, "metal": ti.metal,
         "cuda": ti.cuda, "vulkan": ti.vulkan}[args.backend])
DEVICE = torch.device(args.device)

from coralai.evolver import apply_weights_and_biases

# Import the experiment — all configuration and physics live in experiment.py
from experiment import EXPERIMENT as exp


def main():
    shape = (args.shape, args.shape)

    substrate = exp.make_substrate(shape, DEVICE)
    env       = exp.make_env("flat")
    env.seed(substrate)

    evolver = exp.make_evolver(substrate)

    vis = None if args.no_gui else exp.make_vis(substrate, evolver)
    inds = substrate.ti_indices[None]
    max_steps = args.steps if args.steps > 0 else 10 ** 9

    from evolution import apply_radiation_mutation

    for step in range(max_steps):
        cw, cb = evolver.get_combined_weights()
        out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                 cw, cb, evolver.dir_kernel, evolver.dir_order,
                                 substrate.ti_indices, evolver._get_key_to_local())
        substrate.mem[0, evolver.act_chinds] = out_mem

        exp.run_physics(substrate, evolver)
        exp.run_evolution(substrate, evolver, step)

        if env.persist:
            env.step(substrate)

        if step % 50 == 0 and step > 0:
            apply_radiation_mutation(evolver, 5)

        evolver.timestep = step + 1

        if vis:
            vis.update()
            if not vis.window.running:
                break


if __name__ == "__main__":
    main()
