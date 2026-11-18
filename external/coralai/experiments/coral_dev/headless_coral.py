"""Headless screenshot runner for the development Coral experiment.

From the Coralai directory, run:
    python experiments/coral_dev/headless_coral.py --config experiments/coral_dev/config/config_test.json --seed 42
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import taichi as ti
from PIL import Image


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config") / "config_test.json"
DEFAULT_CONFIG = {
    "steps": 300,
    "shape": 400,
    "backend": "cpu",
    "device": "cpu",
    "seed": 42,
    "env": "flat",
    "env_param": None,
    "env_options": {},
    "save_frame_states": False,
    "frame_state_format": "pt",
    "log_resource_metrics": False,
    "log_cell_events": False,
    "save_initial_checkpoint": False,
    "infra_decay": 0.0,
    "defense_coeff": 0.0,
    "radiate_interval": 50,
    "cleanup_interval": 100,
    "emergency_pop_cap": 1000,
    "checkpoint_interval": 1000,
    "log_interval": 10,
    "screenshot_interval": 1,
    "max_screenshots": 2000,
    "resume_from": None,
    "output_dir": None,
}

parser = argparse.ArgumentParser(description="Headless Coral Dev runner with PNG output")
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help="Path to a JSON run configuration")
parser.add_argument("--seed", type=int, help="Override the saved configuration's random seed")
parser.add_argument("--output-dir", type=Path,
                    help="Write to a new directory instead of the configured output_dir")
cli_args = parser.parse_args()

try:
    with cli_args.config.open() as file:
        config = json.load(file)
except FileNotFoundError:
    parser.error(f"config file not found: {cli_args.config}")
except json.JSONDecodeError as error:
    parser.error(f"invalid JSON in {cli_args.config}: {error}")

if not isinstance(config, dict):
    parser.error("config file must contain one JSON object")
unknown_keys = set(config) - set(DEFAULT_CONFIG)
if unknown_keys:
    parser.error(f"unknown config option(s): {', '.join(sorted(unknown_keys))}")
args = argparse.Namespace(**(DEFAULT_CONFIG | config))
if cli_args.seed is not None:
    args.seed = cli_args.seed
if cli_args.output_dir is not None:
    args.output_dir = str(cli_args.output_dir)

if args.steps < 1:
    parser.error("config option 'steps' must be positive")
if not isinstance(args.seed, int) or not 0 <= args.seed < 2 ** 32:
    parser.error("seed must be an integer from 0 to 4294967295")
if args.screenshot_interval < 1:
    parser.error("config option 'screenshot_interval' must be positive")
if args.backend not in {"cpu", "metal", "cuda", "vulkan"}:
    parser.error("config option 'backend' must be cpu, metal, cuda, or vulkan")
if args.device not in {"cpu", "mps", "cuda"}:
    parser.error("config option 'device' must be cpu, mps, or cuda")
if args.env not in {"flat", "hole", "ring", "stripes", "corners", "patches", "oases", "trail"}:
    parser.error("config option 'env' is not a supported environment")
if not isinstance(args.save_frame_states, bool):
    parser.error("save_frame_states must be a boolean")
if args.frame_state_format not in {"pt", "npz"}:
    parser.error("frame_state_format must be pt or npz")
if not isinstance(args.log_resource_metrics, bool) or (args.log_resource_metrics and args.env != "trail"):
    parser.error("log_resource_metrics must be boolean and requires trail")
if not isinstance(args.log_cell_events, bool) or not isinstance(args.save_initial_checkpoint, bool):
    parser.error("log_cell_events and save_initial_checkpoint must be boolean")
if args.env == "trail" and args.resume_from:
    parser.error("trail resume is unsupported: checkpoints do not restore the food reserve")

from environments import make_env
try:
    configured_env = make_env(args.env, args.env_param, args.env_options)
except (ValueError, TypeError) as error:
    parser.error(str(error))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

backend_map = {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(arch=backend_map[args.backend])
DEVICE = torch.device(args.device)

# This file lives in experiments/coral_dev, so these are the same imports as run.py.
from coralai.evolver import apply_weights_and_biases
from coralai.provenance import simulator_fingerprint
from experiment import EXPERIMENT as exp
from evolution import apply_radiation_mutation

exp.infra_decay = args.infra_decay
exp.defense_coeff = args.defense_coeff


def sync():
    if args.device == "mps":
        torch.mps.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()


def scalar_channel(substrate, name):
    """Return one (W, H) channel even if ``name`` names a struct/vector."""
    index = substrate.windex[name]
    if isinstance(index, (list, tuple)):
        index = index[0]
    values = substrate.mem[0, index].detach().cpu()
    if values.ndim > 2:
        values = values.reshape(-1, *values.shape[-2:]).mean(dim=0)
    return values.numpy()


def make_frame(substrate):
    """Match the normal visualizer's state view: energy / infra / genome."""
    image = np.zeros((substrate.h, substrate.w, 3), dtype=np.float32)
    for color, name in enumerate(("energy", "infra", "genome")):
        channel = scalar_channel(substrate, name)
        lo, hi = channel.min(), channel.max()
        if hi > lo:
            channel = (channel - lo) / (hi - lo)
        else:
            channel = np.zeros_like(channel)
        image[..., color] = channel.T
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def main():
    run_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[2] / "runs" /
        f"coral_dev_headless_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    # A repeated command must never overwrite an earlier experiment.
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w") as file:
        json.dump(vars(args), file, indent=2)
    shape = (args.shape, args.shape)
    substrate = exp.make_substrate(shape, DEVICE)
    env = configured_env
    # Preserve historical initialization for every existing headless config.
    # Trail is newly supported here and must seed AFTER evolver initialization,
    # as in run.py; otherwise its initial energy/founders would be overwritten.
    if args.env != "trail":
        env.seed(substrate)
    # SpaceEvolver creates history/ relative to the working directory.
    previous_directory = Path.cwd()
    try:
        os.chdir(run_dir)
        evolver = exp.make_evolver(substrate)
    finally:
        os.chdir(previous_directory)
    if args.env == "trail":
        env.seed(substrate)

    if args.resume_from:
        evolver.load_checkpoint(args.resume_from)
        print(f"[resume] loaded {args.resume_from} (step {evolver.timestep})")

    start_step = evolver.timestep
    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    if args.save_frame_states:
        (run_dir / "frame_states").mkdir()

    metadata = {
            "experiment": exp.name, "config_file": str(cli_args.config),
            "simulator_provenance": simulator_fingerprint(),
            "effective_config": vars(args),
            "shape": list(shape), "seed": args.seed,
            "env": env.to_dict(), "env_persist": env.persist,
            "start_step": start_step, "resumed_from": args.resume_from,
            "infra_decay": args.infra_decay, "defense_coeff": args.defense_coeff,
            "frame_update_rule": "screenshots and frame states follow all operations at completed update; step_log step + 1",
        }
    with (run_dir / "meta.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    torch.save(substrate.mem.cpu(), run_dir / "initial_state.pt")
    if args.save_initial_checkpoint:
        evolver.save_checkpoint(run_dir, start_step)

    log_file = (run_dir / "step_log.csv").open("w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "total_energy", "total_infra", "energy_pct", "n_alive", "n_genomes"])
    resource_file = None
    if args.log_resource_metrics:
        resource_file = (run_dir / "resource_metrics.csv").open("w", newline="")
        resource_log = csv.writer(resource_file)
        resource_log.writerow(["completed_update", "occupied_cells", "occupied_inside_source", "occupied_outside_source",
            "source_cells", "cumulative_requested", "cumulative_delivered", "cumulative_clipped", "remaining_capacity",
            "total_energy", "total_infra"])

    def log_resources():
        if resource_file is not None:
            state = env.resource_state()
            indices = substrate.ti_indices[None]
            occupied = substrate.mem[0, indices.genome] >= 0
            # Current source means remaining reserve > 0 AFTER this update's
            # translation/release. It does not mean occupied or stored energy > 0.
            source = env._capacity > 0
            inside = int((occupied & source).sum())
            count = int(occupied.sum())
            resource_log.writerow([state["completed_update"], count, inside, count - inside, int(source.sum()),
                state["cumulative_requested"], state["cumulative_delivered"], state["cumulative_clipped"], state["remaining_capacity"],
                float(substrate.mem[0, indices.energy].double().sum()), float(substrate.mem[0, indices.infra].double().sum())])
            resource_file.flush()

    screenshot_count = 0
    last_captured_step = None
    last_checkpoint_step = None

    def capture(step):
        nonlocal screenshot_count, last_captured_step
        if screenshot_count < args.max_screenshots:
            path = screenshots_dir / f"step_{step:07d}.png"
            Image.fromarray(make_frame(substrate)).save(path)
            if args.save_frame_states:
                if args.frame_state_format == "npz":
                    np.savez_compressed(run_dir / "frame_states" / f"update_{step:07d}.npz",
                                        state=substrate.mem.detach().cpu().numpy(), completed_update=step)
                    if args.env == "trail":
                        state = {k: v.numpy() if torch.is_tensor(v) else v for k, v in env.resource_state().items()}
                        np.savez_compressed(run_dir / "frame_states" / f"resource_{step:07d}.npz", **state)
                else:
                    torch.save(substrate.mem.cpu(), run_dir / "frame_states" / f"update_{step:07d}.pt")
                    if args.env == "trail":
                        torch.save(env.resource_state(), run_dir / "frame_states" / f"resource_{step:07d}.pt")
            screenshot_count += 1
            last_captured_step = step

    capture(start_step)
    log_resources()
    events_file = None
    if args.log_cell_events:
        from cell_events import CellEvents
        events = CellEvents(substrate)
        evolver.observe_cell_events = events.observe
        events_file = (run_dir / "cell_events.csv").open("w", newline="")
        events.begin(start_step)
        events_log = csv.DictWriter(events_file, fieldnames=list(events.finish()))
        events_log.writeheader()
    inds = substrate.ti_indices[None]
    end_step = start_step + args.steps
    started = time.perf_counter()
    try:
        for step in range(start_step, end_step):
            # Same step order as coral_dev/run.py.
            if events_file is not None:
                events.begin(step + 1)
            weights, biases = evolver.get_combined_weights()
            out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
            apply_weights_and_biases(
                substrate.mem, out_mem, evolver.sense_chinds, weights, biases,
                evolver.dir_kernel, evolver.dir_order, substrate.ti_indices,
                evolver._get_key_to_local(),
            )
            substrate.mem[0, evolver.act_chinds] = out_mem
            exp.run_physics(substrate, evolver)
            exp.run_evolution(substrate, evolver, step)
            if events_file is not None:
                events.observe("chunk_death")
            if env.persist:
                env.step(substrate)
            if step > 0 and step % args.radiate_interval == 0:
                apply_radiation_mutation(evolver, 5,
                    seed_empty=(getattr(env, "spawn", "spread") == "spread"))
                if events_file is not None:
                    events.observe("radiation")
            if step > 0 and step % args.cleanup_interval == 0:
                evolver.remove_extinct_genomes()
                if events_file is not None:
                    events.observe("cleanup")
            if (args.emergency_pop_cap > 0
                    and len(evolver.genomes) > args.emergency_pop_cap
                    and step - evolver.time_last_cull > 50):
                evolver.reduce_population_to_threshold(args.emergency_pop_cap)
                if events_file is not None:
                    events.observe("cull")
            evolver.timestep = step + 1
            log_resources()
            if events_file is not None:
                events_log.writerow(events.finish())
                events_file.flush()

            if evolver.timestep % args.screenshot_interval == 0:
                capture(evolver.timestep)
            if step % args.log_interval == 0:
                total_energy = float(substrate.mem[0, inds.energy].sum())
                total_infra = float(substrate.mem[0, inds.infra].sum())
                alive = int((substrate.mem[0, inds.genome] >= 0).sum())
                log.writerow([step, total_energy, total_infra,
                              100 * total_energy / (total_energy + total_infra + 1e-8),
                              alive, len(evolver.genomes)])
                log_file.flush()
            if (args.checkpoint_interval > 0
                    and evolver.timestep % args.checkpoint_interval == 0):
                evolver.save_checkpoint(run_dir, evolver.timestep)
                last_checkpoint_step = evolver.timestep

        # Always include the final state, unless it was already sampled.
        if last_captured_step != evolver.timestep:
            capture(evolver.timestep)
        if last_checkpoint_step != evolver.timestep:
            evolver.save_checkpoint(run_dir, evolver.timestep)
        metadata["completed_steps"] = evolver.timestep - start_step
        metadata["final_checkpoint"] = f"checkpoint_{evolver.timestep:07d}"
        if args.env == "trail":
            metadata["resource_budget"] = {k: v for k, v in env.resource_state().items()
                                            if k not in {"capacity", "last_injection"}}
        with (run_dir / "meta.json").open("w") as file:
            json.dump(metadata, file, indent=2)
    finally:
        log_file.close()
        if resource_file is not None:
            resource_file.close()
        if events_file is not None:
            events_file.close()

    elapsed = time.perf_counter() - started
    metadata["elapsed_seconds"] = elapsed
    with (run_dir / "meta.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    print(f"{args.steps} steps in {elapsed:.2f}s ({args.steps / elapsed:.1f} FPS)")
    print(f"Screenshots: {screenshots_dir} ({screenshot_count} PNGs)")
    print(f"Run data: {run_dir}")


if __name__ == "__main__":
    main()
