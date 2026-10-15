"""
Headless benchmark for coral_runner_space.

Measures step throughput (FPS) with no GUI overhead.
Seeds everything for deterministic, reproducible runs.

IMPORTANT: By default, initializes ALL cells as alive (genome >= 0)
to benchmark worst-case (most computation) scenario.

Usage:
    # On Mac with Metal GPU:
    python benchmark_coral_space.py --steps 200 --shape 400 --backend metal --device mps

    # On CPU (any platform):
    python benchmark_coral_space.py --steps 100 --shape 100 --backend cpu --device cpu

    # With profiling:
    python benchmark_coral_space.py --steps 100 --shape 100 --profile

    # With sparse population (80% dead, like default init):
    python benchmark_coral_space.py --steps 100 --shape 100 --occupancy 0.2
"""

import argparse
import os
import time
import random

import numpy as np
import torch
import taichi as ti
import neat

parser = argparse.ArgumentParser(description="Coral Space Benchmark")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--shape", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--backend", type=str, default="cpu",
                    choices=["cpu", "metal", "cuda", "vulkan"],
                    help="Taichi backend")
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "mps", "cuda"],
                    help="PyTorch device")
parser.add_argument("--profile", action="store_true",
                    help="Print per-function timing breakdown")
parser.add_argument("--warmup", type=int, default=5,
                    help="Warmup steps (excluded from timing)")
parser.add_argument("--occupancy", type=float, default=1.0,
                    help="Fraction of cells alive (1.0 = all alive, 0.2 = 20%%)")
parser.add_argument("--no-radiation", action="store_true",
                    help="Disable radiation/mutation (pure step timing)")
args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(args.seed)

backend_map = {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda,
               "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.substrate import Substrate
from coralai.evolver import SpaceEvolver, apply_weights_and_biases
from experiments.coral.physics import (
    activate_outputs, invest_liquidate, explore_physics, energy_physics)


def build_evolver(shape, device, occupancy=1.0):
    """Create a SpaceEvolver identical to coral_runner_space.py."""
    channels = {
        "energy": ti.f32,
        "infra": ti.f32,
        "acts": ti.types.struct(
            invest=ti.f32,
            liquidate=ti.f32,
            explore=ti.types.vector(n=4, dtype=ti.f32),
        ),
        "com": ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
        "rot": ti.f32,
        "genome": ti.f32,
    }
    substrate = Substrate(shape, torch.float32, device, channels)
    substrate.malloc()

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir,
                               "experiments/coral/neat.config")
    kernel = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
              [-1, 0], [-1, -1], [0, -1], [1, -1]]
    dir_order = [0, -1, 1]
    sense_chs = ["energy", "infra", "com"]
    act_chs = ["acts", "com"]

    evolver = SpaceEvolver(config_path, substrate, kernel, dir_order,
                           sense_chs, act_chs)

    # Override occupancy: set fraction of cells to alive genomes
    if occupancy != 0.2:  # 0.2 is the default from init_substrate
        inds = substrate.ti_indices[None]
        n_genomes = len(evolver.genomes)
        if n_genomes > 0:
            alive_mask = torch.rand(shape, device=device) < occupancy
            substrate.mem[0, inds.genome] = torch.where(
                alive_mask,
                torch.randint(0, n_genomes, shape,
                              device=device).float(),
                torch.full(shape, -1.0, device=device),
            )

    return evolver, substrate


def run_benchmark():
    shape = (args.shape, args.shape)
    cells = shape[0] * shape[1]
    print(f"Benchmark: {shape[0]}x{shape[1]} ({cells:,} cells), "
          f"{args.steps} steps, seed={args.seed}")
    print(f"Backend: {args.backend} | Device: {args.device} | "
          f"Occupancy: {args.occupancy*100:.0f}%")

    seed_everything(args.seed)
    evolver, substrate = build_evolver(shape, DEVICE, args.occupancy)

    inds = substrate.ti_indices[None]
    alive = (substrate.mem[0, inds.genome] >= 0).sum().item()
    print(f"Alive cells: {alive:,} / {cells:,} "
          f"({100*alive/cells:.1f}%)")
    print(f"Genomes: {len(evolver.genomes)}")

    # Warmup (JIT compilation for Taichi kernels)
    print(f"Warming up ({args.warmup} steps)...")
    for _ in range(args.warmup):
        cw = torch.stack(evolver.combined_weights, dim=0)
        cb = torch.stack(evolver.combined_biases, dim=0)
        evolver.step_sim(cw, cb)

    # Reset for clean benchmark
    seed_everything(args.seed)
    evolver, substrate = build_evolver(shape, DEVICE, args.occupancy)
    inds = substrate.ti_indices[None]

    if args.profile:
        timings = {}

        def timed(name, fn, *a, **kw):
            if args.device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            result = fn(*a, **kw)
            if args.device == "mps":
                torch.mps.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            timings[name] = timings.get(name, 0.0) + dt
            return result

        def profiled_step(evolver, cw, cb):
            inds = evolver.substrate.ti_indices[None]

            # forward: apply_weights_and_biases
            out_mem = torch.zeros_like(
                evolver.substrate.mem[0, evolver.act_chinds])
            timed("1_apply_weights_biases",
                  apply_weights_and_biases,
                  evolver.substrate.mem, out_mem,
                  evolver.sense_chinds, cw, cb,
                  evolver.dir_kernel, evolver.dir_order,
                  evolver.substrate.ti_indices)
            evolver.substrate.mem[0, evolver.act_chinds] = out_mem

            timed("2_activate_outputs",
                  activate_outputs, evolver.substrate)
            timed("3_invest_liquidate",
                  invest_liquidate, evolver.substrate)
            timed("4_explore_physics",
                  explore_physics, evolver.substrate,
                  evolver.kernel, evolver.dir_order)
            timed("5_energy_physics",
                  energy_physics, evolver.substrate,
                  evolver.kernel, max_infra=10, max_energy=1.5)

            evolver.substrate.mem[0, inds.genome] = torch.where(
                (evolver.substrate.mem[0, inds.infra]
                 + evolver.substrate.mem[0, inds.energy]) > 0.05,
                evolver.substrate.mem[0, inds.genome], -1)

            # step_sim rest
            def step_rest():
                evolver.energy_offset = evolver.get_energy_offset(
                    evolver.timestep)
                evolver.ages = [age + 1 for age in evolver.ages]
                evolver.substrate.mem[0, inds.energy] += (
                    torch.randn_like(evolver.substrate.mem[0, inds.energy])
                    + evolver.energy_offset) * 0.1
                evolver.substrate.mem[0, inds.infra] += (
                    torch.randn_like(evolver.substrate.mem[0, inds.energy])
                    + evolver.energy_offset) * 0.1
                evolver.substrate.mem[0, inds.energy] = torch.clamp(
                    evolver.substrate.mem[0, inds.energy], 0.01, 100)
                evolver.substrate.mem[0, inds.infra] = torch.clamp(
                    evolver.substrate.mem[0, inds.infra], 0.01, 100)
                if evolver.timestep % 50 == 0:
                    evolver.kill_random_chunk(5)
            timed("6_step_rest", step_rest)

        def profiled_main_loop():
            for step in range(args.steps):
                t_stack = time.perf_counter()
                cw = torch.stack(evolver.combined_weights, dim=0)
                cb = torch.stack(evolver.combined_biases, dim=0)
                timings["0_stack_weights"] = timings.get(
                    "0_stack_weights", 0) + time.perf_counter() - t_stack

                profiled_step(evolver, cw, cb)
                evolver.timestep = step + 1

        print(f"Running {args.steps} steps (profiled)...")
        t_start = time.perf_counter()
        profiled_main_loop()
        total = time.perf_counter() - t_start

        fps = args.steps / total
        print(f"\n{'='*65}")
        print(f"  Total: {total:.3f}s  |  Avg: {total/args.steps*1000:.2f}"
              f"ms/step  |  FPS: {fps:.1f}")
        print(f"{'='*65}")
        print(f"\n  Per-function breakdown ({args.steps} steps):")
        total_profiled = sum(timings.values())
        for name, t in sorted(timings.items()):
            pct = 100 * t / total_profiled
            per_step = 1000 * t / args.steps
            print(f"    {name:35s}  {t:8.3f}s  {per_step:7.2f}ms/step  "
                  f"{pct:5.1f}%")
        print(f"    {'TOTAL':35s}  {total_profiled:8.3f}s")
    else:
        print(f"Running {args.steps} steps...")
        t_start = time.perf_counter()
        step_times = []
        for step in range(args.steps):
            t0 = time.perf_counter()
            cw = torch.stack(evolver.combined_weights, dim=0)
            cb = torch.stack(evolver.combined_biases, dim=0)
            evolver.step_sim(cw, cb)
            evolver.timestep = step + 1
            dt = time.perf_counter() - t0
            step_times.append(dt)

        total = time.perf_counter() - t_start
        avg = total / args.steps
        fps = 1.0 / avg
        print(f"\n{'='*65}")
        print(f"  Total: {total:.3f}s  |  Avg: {avg*1000:.2f}ms/step  |  "
              f"FPS: {fps:.1f}")
        print(f"  Min: {min(step_times)*1000:.2f}ms  "
              f"Max: {max(step_times)*1000:.2f}ms  "
              f"Median: {sorted(step_times)[len(step_times)//2]*1000:.2f}ms")
        print(f"{'='*65}")

    # Final state fingerprint
    alive_final = (substrate.mem[0, inds.genome] >= 0).sum().item()
    fp = {
        "energy_sum": float(substrate.mem[0, inds.energy].sum()),
        "infra_sum": float(substrate.mem[0, inds.infra].sum()),
        "alive_cells": alive_final,
    }
    print(f"\n  Final: {alive_final:,} alive cells | {fp}")
    return fp


if __name__ == "__main__":
    run_benchmark()
