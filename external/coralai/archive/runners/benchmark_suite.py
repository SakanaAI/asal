"""
Multi-configuration benchmark suite for coral_runner_space.

Runs multiple configurations and repeated trials for reliable data.

Usage:
    python benchmark_suite.py --backend metal --device mps
    python benchmark_suite.py --backend cpu --device cpu --quick
"""

import argparse
import os
import time
import random
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import taichi as ti

parser = argparse.ArgumentParser(description="Coral Space Benchmark Suite")
parser.add_argument("--backend", type=str, default="cpu",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--quick", action="store_true",
                    help="Run fewer configs for a quick check")
parser.add_argument("--output", type=str, default=None,
                    help="Save results JSON to this path")
args = parser.parse_args()

backend_map = {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda,
               "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.substrate.substrate import Substrate
from coralai.evolution.space_evolver import SpaceEvolver
from coralai.instances.coral.coral_physics import (
    activate_outputs, invest_liquidate, explore_physics, energy_physics,
    apply_weights_and_biases)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync():
    if args.device == "mps":
        torch.mps.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()


def build_evolver(shape, occupancy=0.2):
    channels = {
        "energy": ti.f32,
        "infra": ti.f32,
        "acts": ti.types.struct(
            invest=ti.f32, liquidate=ti.f32,
            explore=ti.types.vector(n=4, dtype=ti.f32),
        ),
        "com": ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
        "rot": ti.f32,
        "genome": ti.f32,
    }
    substrate = Substrate(shape, torch.float32, DEVICE, channels)
    substrate.malloc()

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir,
                               "coralai/instances/coral/coral_neat.config")
    kernel = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
              [-1, 0], [-1, -1], [0, -1], [1, -1]]
    dir_order = [0, -1, 1]

    evolver = SpaceEvolver(config_path, substrate, kernel, dir_order,
                           ["energy", "infra", "com"], ["acts", "com"])

    if occupancy != 0.2:
        inds = substrate.ti_indices[None]
        n_genomes = len(evolver.genomes)
        if n_genomes > 0:
            alive_mask = torch.rand(shape, device=DEVICE) < occupancy
            substrate.mem[0, inds.genome] = torch.where(
                alive_mask,
                torch.randint(0, n_genomes, shape, device=DEVICE).float(),
                torch.full(shape, -1.0, device=DEVICE))

    return evolver, substrate


def run_trial(shape, steps, occupancy, seed, warmup=3):
    seed_everything(seed)
    evolver, substrate = build_evolver(shape, occupancy)
    inds = substrate.ti_indices[None]

    for _ in range(warmup):
        cw, cb = evolver.get_combined_weights()
        evolver.step_sim(cw, cb)

    seed_everything(seed + 1000)
    evolver2, substrate2 = build_evolver(shape, occupancy)
    evolver = evolver2
    substrate = substrate2
    inds = substrate.ti_indices[None]

    timings = defaultdict(float)
    step_times = []

    for step in range(steps):
        t0 = time.perf_counter()

        sync()
        t_w = time.perf_counter()
        cw, cb = evolver.get_combined_weights()
        sync()
        timings["weights"] += time.perf_counter() - t_w

        sync()
        t_f = time.perf_counter()
        out_mem = evolver._get_scratch("out_mem",
                                       substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(
            substrate.mem, out_mem, evolver.sense_chinds,
            cw, cb, evolver.dir_kernel, evolver.dir_order,
            substrate.ti_indices)
        substrate.mem[0, evolver.act_chinds] = out_mem
        sync()
        timings["fwd_kernel"] += time.perf_counter() - t_f

        sync()
        t_a = time.perf_counter()
        activate_outputs(substrate)
        sync()
        timings["activate"] += time.perf_counter() - t_a

        sync()
        t_i = time.perf_counter()
        invest_liquidate(substrate)
        sync()
        timings["invest_liq"] += time.perf_counter() - t_i

        sync()
        t_e = time.perf_counter()
        explore_physics(substrate, evolver.kernel, evolver.dir_order)
        sync()
        timings["explore"] += time.perf_counter() - t_e

        sync()
        t_p = time.perf_counter()
        energy_physics(substrate, evolver.kernel, max_infra=10,
                       max_energy=1.5)
        sync()
        timings["energy_phys"] += time.perf_counter() - t_p

        alive = (substrate.mem[0, inds.infra]
                 + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)

        evolver.energy_offset = evolver.get_energy_offset(step)
        evolver.ages = [age + 1 for age in evolver.ages]
        offset = evolver.energy_offset
        substrate.mem[0, inds.energy].add_(
            torch.randn_like(substrate.mem[0, inds.energy]).add_(offset).mul_(0.1))
        substrate.mem[0, inds.infra].add_(
            torch.randn_like(substrate.mem[0, inds.infra]).add_(offset).mul_(0.1))
        substrate.mem[0, inds.energy].clamp_(0.01, 100)
        substrate.mem[0, inds.infra].clamp_(0.01, 100)
        if step % 50 == 0:
            evolver.kill_random_chunk(5)
        evolver.timestep = step + 1

        step_times.append(time.perf_counter() - t0)

    alive_final = (substrate.mem[0, inds.genome] >= 0).sum().item()
    avg_ms = 1000 * sum(step_times) / len(step_times)
    fps = 1000 / avg_ms

    return {
        "avg_ms": avg_ms,
        "fps": fps,
        "median_ms": 1000 * sorted(step_times)[len(step_times) // 2],
        "min_ms": 1000 * min(step_times),
        "max_ms": 1000 * max(step_times),
        "alive_final": alive_final,
        "genomes": len(evolver.genomes),
        "timings": {k: 1000 * v / steps for k, v in timings.items()},
    }


def main():
    if args.quick:
        configs = [
            {"shape": (100, 100), "steps": 50,  "occupancy": 1.0, "trials": 2},
            {"shape": (200, 200), "steps": 30,  "occupancy": 1.0, "trials": 2},
        ]
    else:
        configs = [
            {"shape": (100, 100), "steps": 100, "occupancy": 0.2, "trials": 3,
             "label": "100x100 sparse (default init)"},
            {"shape": (100, 100), "steps": 100, "occupancy": 1.0, "trials": 3,
             "label": "100x100 full"},
            {"shape": (200, 200), "steps": 50,  "occupancy": 0.2, "trials": 3,
             "label": "200x200 sparse"},
            {"shape": (200, 200), "steps": 50,  "occupancy": 1.0, "trials": 3,
             "label": "200x200 full"},
            {"shape": (400, 400), "steps": 30,  "occupancy": 0.2, "trials": 3,
             "label": "400x400 sparse"},
            {"shape": (400, 400), "steps": 30,  "occupancy": 1.0, "trials": 3,
             "label": "400x400 full"},
        ]

    print(f"Benchmark Suite | {args.backend}/{args.device}")
    print(f"{'='*75}")

    all_results = []

    for cfg in configs:
        label = cfg.get("label", f"{cfg['shape'][0]}x{cfg['shape'][1]} "
                        f"occ={cfg['occupancy']}")
        trials = cfg["trials"]
        trial_results = []

        for trial in range(trials):
            seed = 42 + trial * 1000
            result = run_trial(cfg["shape"], cfg["steps"],
                               cfg["occupancy"], seed)
            trial_results.append(result)

        fps_vals = [r["fps"] for r in trial_results]
        avg_fps = sum(fps_vals) / len(fps_vals)
        ms_vals = [r["avg_ms"] for r in trial_results]
        avg_ms = sum(ms_vals) / len(ms_vals)

        avg_timings = {}
        for key in trial_results[0]["timings"]:
            avg_timings[key] = sum(r["timings"][key]
                                   for r in trial_results) / trials

        print(f"\n  {label}")
        print(f"    FPS: {avg_fps:.1f} (±{max(fps_vals)-min(fps_vals):.1f})  "
              f"|  {avg_ms:.1f}ms/step  |  {trials} trials")
        total_t = sum(avg_timings.values())
        for k, v in sorted(avg_timings.items()):
            pct = 100 * v / total_t if total_t > 0 else 0
            print(f"      {k:15s}  {v:7.2f}ms  {pct:5.1f}%")

        all_results.append({
            "label": label,
            "shape": list(cfg["shape"]),
            "occupancy": cfg["occupancy"],
            "steps": cfg["steps"],
            "trials": trials,
            "avg_fps": avg_fps,
            "avg_ms": avg_ms,
            "fps_range": [min(fps_vals), max(fps_vals)],
            "timings": avg_timings,
        })

    print(f"\n{'='*75}")
    print(f"\n  Summary:")
    print(f"  {'Config':<30s}  {'FPS':>8s}  {'ms/step':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
    for r in all_results:
        print(f"  {r['label']:<30s}  {r['avg_fps']:>7.1f}  "
              f"  {r['avg_ms']:>7.1f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"backend": args.backend, "device": args.device,
                       "timestamp": datetime.now().isoformat(),
                       "results": all_results}, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
