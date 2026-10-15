"""
Profiled GUI runner for coral_runner_space.

Runs the full GUI simulation for a fixed number of steps, prints
FPS stats and per-function profiling to the terminal, then exits.

Usage (Mac):
    python profile_coral_space_gui.py
    python profile_coral_space_gui.py --steps 500 --shape 200
    python profile_coral_space_gui.py --no-gui   # headless, same profiling
"""

import argparse
import os
import time
from collections import defaultdict

import torch
import taichi as ti

parser = argparse.ArgumentParser(description="Profiled Coral Space GUI Runner")
parser.add_argument("--steps", type=int, default=300,
                    help="Number of steps to run before exiting")
parser.add_argument("--shape", type=int, default=400)
parser.add_argument("--backend", type=str, default="metal",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true",
                    help="Skip GUI rendering (isolate simulation cost)")
parser.add_argument("--radiate-interval", type=int, default=50)
parser.add_argument("--cull-max-pop", type=int, default=100)
args = parser.parse_args()

backend_map = {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda,
               "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.substrate import Substrate
from coralai.evolver import SpaceEvolver, apply_weights_and_biases
from coralai.visualization import Visualization
from experiments.coral.physics import (
    activate_outputs, invest_liquidate, explore_physics, energy_physics)


class ProfiledCoralVis(Visualization):
    def __init__(self, substrate, evolver, vis_chs):
        super().__init__(substrate, vis_chs)
        self.evolver = evolver
        self.next_generation = False
        self.genome_stats = []
        self.fps_history = []
        self.last_print_time = time.time()
        self.frames_since_print = 0

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(200 / self.img_w, self.img_w)
        opt_h = min(500 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.next_generation = sub_w.checkbox("Next Generation",
                                                  self.next_generation)
            self.chinds[0] = sub_w.slider_int(
                f"R: {self.substrate.index_to_chname(self.chinds[0])}",
                self.chinds[0], 0, self.substrate.mem.shape[1]-1)
            self.chinds[1] = sub_w.slider_int(
                f"G: {self.substrate.index_to_chname(self.chinds[1])}",
                self.chinds[1], 0, self.substrate.mem.shape[1]-1)
            self.chinds[2] = sub_w.slider_int(
                f"B: {self.substrate.index_to_chname(self.chinds[2])}",
                self.chinds[2], 0, self.substrate.mem.shape[1]-1)
            current_pos = self.window.get_cursor_pos()
            pos_x = int(current_pos[0] * self.w) % self.w
            pos_y = int(current_pos[1] * self.h) % self.h
            sub_w.text(
                f"GENOME: {self.substrate.mem[0, inds.genome, pos_x, pos_y]:.2f}\n"
                + f"Energy: {self.substrate.mem[0, inds.energy, pos_x, pos_y]:.2f}\n"
                + f"Infra: {self.substrate.mem[0, inds.infra, pos_x, pos_y]:.2f}\n")
            sub_w.text(f"TIMESTEP: {self.evolver.timestep}")
            tot_energy = torch.sum(self.substrate.mem[0, inds.energy])
            tot_infra = torch.sum(self.substrate.mem[0, inds.infra])
            sub_w.text(f"Total Energy+Infra: {tot_energy + tot_infra}")
            sub_w.text(f"Percent Energy: "
                       f"{(tot_energy / (tot_energy + tot_infra)) * 100}")
            sub_w.text(f"Energy Offset: {self.evolver.energy_offset}")
            sub_w.text(f"# Genomes: {len(self.evolver.genomes)}")
            if self.fps_history:
                sub_w.text(f"FPS: {self.fps_history[-1]:.1f}")

            if self.evolver.timestep % 20 == 0:
                self.genome_stats = []
                for i in range(len(self.evolver.genomes)):
                    n_cells = self.substrate.mem[0, inds.genome].eq(i).sum().item()
                    age = self.evolver.ages[i]
                    self.genome_stats.append((i, n_cells, age))
                self.genome_stats.sort(key=lambda x: x[1], reverse=True)
            for i, n_cells, age in self.genome_stats[:10]:
                sub_w.text(f"\tG{i}: {n_cells:.0f} cells, {age} age")


def sync_device():
    if args.device == "mps":
        torch.mps.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()


def main():
    shape = (args.shape, args.shape)
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
    substrate = Substrate(shape, torch.float32, DEVICE, channels)
    substrate.malloc()

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir,
                               "experiments/coral/neat.config")
    kernel = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
              [-1, 0], [-1, -1], [0, -1], [1, -1]]
    dir_order = [0, -1, 1]

    evolver = SpaceEvolver(config_path, substrate, kernel, dir_order,
                           ["energy", "infra", "com"], ["acts", "com"])

    if args.no_gui:
        vis = None
    else:
        vis = ProfiledCoralVis(substrate, evolver,
                               ["energy", "infra", "rot"])

    inds = substrate.ti_indices[None]
    timings = defaultdict(float)
    step_times = []
    fps_window = []
    fps_print_interval = 2.0

    print(f"Running {args.steps} steps | {shape[0]}x{shape[1]} | "
          f"{args.backend}/{args.device} | GUI={'ON' if vis else 'OFF'}")
    print(f"Genomes: {len(evolver.genomes)}")
    print("-" * 70)

    t_session = time.time()
    last_fps_time = time.time()

    for step in range(args.steps):
        t_step = time.perf_counter()

        # Stack weights (cached)
        sync_device()
        t0 = time.perf_counter()
        cw, cb = evolver.get_combined_weights()
        sync_device()
        timings["0_get_weights"] += time.perf_counter() - t0

        # Forward: apply_weights_and_biases
        sync_device()
        t0 = time.perf_counter()
        out_mem = evolver._get_scratch("out_mem",
                                       substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(
            substrate.mem, out_mem, evolver.sense_chinds,
            cw, cb, evolver.dir_kernel, evolver.dir_order,
            substrate.ti_indices)
        substrate.mem[0, evolver.act_chinds] = out_mem
        sync_device()
        timings["1_apply_weights"] += time.perf_counter() - t0

        # Physics
        sync_device()
        t0 = time.perf_counter()
        activate_outputs(substrate)
        sync_device()
        timings["2_activate_out"] += time.perf_counter() - t0

        sync_device()
        t0 = time.perf_counter()
        invest_liquidate(substrate)
        sync_device()
        timings["3_invest_liq"] += time.perf_counter() - t0

        sync_device()
        t0 = time.perf_counter()
        explore_physics(substrate, evolver.kernel, evolver.dir_order)
        sync_device()
        timings["4_explore"] += time.perf_counter() - t0

        sync_device()
        t0 = time.perf_counter()
        energy_physics(substrate, evolver.kernel, max_infra=10,
                       max_energy=1.5)
        sync_device()
        timings["5_energy"] += time.perf_counter() - t0

        # Genome death
        sync_device()
        t0 = time.perf_counter()
        alive = (substrate.mem[0, inds.infra]
                 + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)
        sync_device()
        timings["6_death"] += time.perf_counter() - t0

        # Step rest (noise, clamp, kill chunk)
        sync_device()
        t0 = time.perf_counter()
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
        sync_device()
        timings["7_step_rest"] += time.perf_counter() - t0

        # Rendering
        if vis:
            sync_device()
            t0 = time.perf_counter()
            vis.update()
            sync_device()
            timings["8_render"] += time.perf_counter() - t0

        # Radiation / culling
        if step % args.radiate_interval == 0 and step > 0:
            sync_device()
            t0 = time.perf_counter()
            evolver.apply_radiation_mutation(5)
            sync_device()
            timings["9_radiation"] += time.perf_counter() - t0

        if (len(evolver.genomes) > args.cull_max_pop
                and (step - evolver.time_last_cull) > 50):
            evolver.reduce_population_to_threshold(args.cull_max_pop)

        evolver.timestep = step + 1
        dt = time.perf_counter() - t_step
        step_times.append(dt)

        # Print FPS every N seconds
        now = time.time()
        fps_window.append(dt)
        if now - last_fps_time >= fps_print_interval:
            recent_fps = len(fps_window) / sum(fps_window)
            alive_count = (substrate.mem[0, inds.genome] >= 0).sum().item()
            n_genomes = len(evolver.genomes)
            if vis:
                vis.fps_history.append(recent_fps)
            print(f"  step {step+1:4d} | {recent_fps:5.1f} FPS | "
                  f"{1000*sum(fps_window)/len(fps_window):5.1f}ms/step | "
                  f"alive: {alive_count:,} | genomes: {n_genomes}")
            fps_window = []
            last_fps_time = now

        if vis and not vis.window.running:
            break

    total = time.time() - t_session
    avg_fps = len(step_times) / sum(step_times)
    median_ms = sorted(step_times)[len(step_times) // 2] * 1000

    print(f"\n{'='*70}")
    print(f"  {len(step_times)} steps in {total:.2f}s")
    print(f"  Avg: {avg_fps:.1f} FPS  |  Median: {median_ms:.1f}ms/step")
    print(f"  Min: {min(step_times)*1000:.1f}ms  "
          f"Max: {max(step_times)*1000:.1f}ms")
    print(f"{'='*70}")

    total_profiled = sum(timings.values())
    print(f"\n  Per-function breakdown:")
    for name, t in sorted(timings.items()):
        pct = 100 * t / total_profiled if total_profiled > 0 else 0
        per_step = 1000 * t / len(step_times)
        print(f"    {name:25s}  {t:8.3f}s  {per_step:7.2f}ms/step  "
              f"{pct:5.1f}%")
    print(f"    {'TOTAL':25s}  {total_profiled:8.3f}s")

    alive_final = (substrate.mem[0, inds.genome] >= 0).sum().item()
    print(f"\n  Final: {alive_final:,} alive | "
          f"{len(evolver.genomes)} genomes")


if __name__ == "__main__":
    main()
