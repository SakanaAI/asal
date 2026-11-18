"""Bounded, synchronized comparison of fixed and translated finite trail food."""

import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_survival import measure_state, sha256

RUNNER = ROOT / "external/coralai/experiments/coral_dev/headless_coral.py"
SOURCE_FILES = [
    RUNNER.relative_to(ROOT).as_posix(),
    "external/coralai/experiments/coral_dev/environments.py",
    "external/coralai/experiments/coral_dev/experiment.py",
    "scripts/demo_dynamic_energy.py",
    "tests/test_dynamic_energy.py",
    "configs/dynamic_energy_demo/fixed.json",
    "configs/dynamic_energy_demo/dynamic.json",
]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def composite(update, states, resources, counts, configs, maximum):
    """Two aligned columns; occupancy and food are explicitly separate views."""
    panel = 288
    canvas = Image.new("RGB", (660, 758), (17, 22, 31))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    draw.text((18, 10), f"Finite trail energy | completed update {update:04d}", font=font, fill="white")
    settings = configs["fixed"]
    draw.text((18, 36), f"Seed {settings['seed']} | {settings['shape']} x {settings['shape']} | matched initial state and supply", font=small, fill="#c6cbd2")
    for index, name in enumerate(("fixed", "dynamic")):
        x = 18 + index * 330
        options = configs[name]["env_options"]
        interval = options["shift_interval"]
        label = "Fixed: sources stay in place" if not interval else f"Dynamic: sources y +{options['shift_cells']} every {interval} updates"
        draw.text((x, 65), label, font=small, fill="white")
        draw.text((x, 88), f"Occupied cells: {counts[name]} / {settings['shape'] ** 2}", font=small, fill="#54d9ed")
        occupied = states[name][0, 13].numpy().T >= 0
        rgb = np.zeros((*occupied.shape, 3), dtype=np.uint8)
        rgb[:] = (24, 32, 43)
        rgb[occupied] = (69, 211, 230)
        canvas.paste(Image.fromarray(rgb).resize((panel, panel), Image.Resampling.NEAREST), (x, 112))
        reserve = resources[name]["capacity"].numpy().T / maximum
        rgb = np.stack((255 * np.sqrt(reserve), 220 * reserve, 55 * reserve), axis=-1).clip(0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb).resize((panel, panel), Image.Resampling.NEAREST), (x, 435))
        draw.text((x, 410), "Food reserve (same color scale)", font=small, fill="#ffe08a")
        delivered = resources[name]["cumulative_delivered"]
        draw.text((x, 730), f"Energy delivered since start: {delivered:.2f}", font=small, fill="#c6cbd2")
    return canvas


def analyze(output, configs):
    settings = configs["fixed"]
    updates = sorted(set(range(0, settings["steps"] + 1, settings["screenshot_interval"])) | {settings["steps"]})
    rows, pairs, frames = [], [], []
    previous = {}
    capacity_max = None
    budgets = {}
    (output / "paired_frames").mkdir(exist_ok=True)
    for update in updates:
        states, resources, counts = {}, {}, {}
        for name in ("fixed", "dynamic"):
            run = output / name
            state_path = run / "frame_states" / f"update_{update:07d}.pt"
            image_path = run / "screenshots" / f"step_{update:07d}.png"
            state = states[name] = load(state_path)
            resource = resources[name] = load(run / "frame_states" / f"resource_{update:07d}.pt")
            require(resource["completed_update"] == update, "Resource/state update mismatch")
            measured, diagnostic = measure_state(state_path, [settings["shape"]] * 2, image_path=image_path)
            occupied = state[0, 13].numpy() >= 0
            food = resource["capacity"].numpy() > 0
            counts[name] = measured["living_cell_count"]
            row = {
                "configuration": name, "completed_update": update,
                "occupied_cells": counts[name], "occupied_fraction": counts[name] / occupied.size,
                "occupied_genome_ids": measured["occupied_genome_ids"],
                "occupied_on_food": int((occupied & food).sum()),
                "food_cells": int(food.sum()),
                "changed_occupancy_since_previous_sample": int((occupied != previous[name]).sum()) if name in previous else 0,
                "total_energy": float(state[0, 0].double().sum()),
                "total_infra": float(state[0, 1].double().sum()),
                **{k: resource[k] for k in ("initial_capacity", "remaining_capacity", "cumulative_requested", "cumulative_delivered", "cumulative_clipped")},
                "state_sha256": sha256(state_path), "screenshot_sha256": sha256(image_path),
                "screenshot_matches_state": diagnostic["matches_saved_screenshot_pixels"],
            }
            rows.append(row)
            previous[name] = occupied.copy()
            if update == 0:
                budgets[name] = {"initial_energy": row["total_energy"], "initial_infra": row["total_infra"], "initial_reserve": row["initial_capacity"]}
                capacity_max = max(capacity_max or 0, float(resource["capacity"].max()))
        if update == 0:
            require(torch.equal(states["fixed"], states["dynamic"]), "Initial states differ")
            require(torch.equal(resources["fixed"]["capacity"], resources["dynamic"]["capacity"]), "Initial reserves differ")
        # Translation changes spatial supply, never its nominal amount.
        for field in ("initial_capacity", "remaining_capacity", "cumulative_requested"):
            require(abs(resources["fixed"][field] - resources["dynamic"][field]) < 1e-7, f"Unmatched supply: {field} at {update}")
        delivered_difference = abs(resources["fixed"]["cumulative_delivered"] - resources["dynamic"]["cumulative_delivered"])
        require(delivered_difference < 0.002, f"Unmatched actual energy supply at {update}")
        require(all(r["cumulative_clipped"] == 0 for r in resources.values()), "Energy clipping changes the resource budget")
        pairs.append({"completed_update": update,
                      "differing_occupied_cells": int((previous["fixed"] != previous["dynamic"]).sum()),
                      "actual_supply_absolute_difference": delivered_difference})
        frame = composite(update, states, resources, counts, configs, capacity_max)
        frame.save(output / "paired_frames" / f"update_{update:07d}.png")
        frames.append(frame)
    for name in budgets:
        meta = json.loads((output / name / "meta.json").read_text())
        require(meta["completed_steps"] == configs[name]["steps"], "Run incomplete")
        budgets[name].update(meta["resource_budget"])
        budgets[name]["initial_energy_plus_actual_supply"] = budgets[name]["initial_energy"] + budgets[name]["cumulative_delivered"]
        budgets[name]["reserve_depletion_minus_requested"] = budgets[name]["initial_reserve"] - budgets[name]["remaining_capacity"] - budgets[name]["cumulative_requested"]
        budgets[name]["requested_minus_delivered"] = budgets[name]["cumulative_requested"] - budgets[name]["cumulative_delivered"]
        require(len(list((output / name / "screenshots").glob("*.png"))) == len(updates), "Unexpected screenshot coverage")
        final = load(output / name / f"checkpoint_{updates[-1]:07d}" / "substrate.pt")
        require(torch.equal(final, states[name]), "Final checkpoint does not match sampled state")
    for filename, data in (("measurements.csv", rows), ("paired_measurements.csv", pairs)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    gif = output / "comparison.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=150, loop=0, disposal=2, optimize=False)
    with Image.open(gif) as animation:
        require(animation.n_frames == len(updates) and animation.info["loop"] == 0, "Invalid animation frame count/loop")
        durations = []
        for i in range(animation.n_frames):
            animation.seek(i)
            decoded = np.asarray(animation.convert("RGB"))
            for column, name in enumerate(("fixed", "dynamic")):
                x = 18 + column * 330
                panel = decoded[112:400, x:x + 288]
                decoded_occupancy = (panel[..., 1] > 100) & (panel[..., 2] > 100) & (panel[..., 0] < 150)
                cells = load(output / name / "frame_states" / f"update_{updates[i]:07d}.pt")[0, 13].numpy().T >= 0
                expected = np.asarray(Image.fromarray(cells).resize((288, 288), Image.Resampling.NEAREST))
                require(np.array_equal(decoded_occupancy, expected), f"GIF/state synchronization failed at {updates[i]}")
            durations.append(animation.info["duration"])
        require(durations == [150] * len(updates), "Invalid animation timing")
    contact = Image.new("RGB", (1320, 1516))
    for index, update in enumerate((updates[0], updates[len(updates) // 4], updates[len(updates) // 2], updates[-1])):
        contact.paste(frames[updates.index(update)], ((index % 2) * 660, (index // 2) * 758))
    contact.save(output / "contact_sheet.png")
    save_json(output / "validation.json", {
        "completed_updates": updates, "state_screenshot_pixel_matches": len(rows),
        "identical_initial_states_and_reserves": True, "nominal_supply_matches_at_every_sample": True,
        "maximum_actual_supply_difference": max(p["actual_supply_absolute_difference"] for p in pairs),
        "actual_supply_tolerance": 0.002, "energy_clipped": 0,
        "final_state_matches_checkpoint": True, "animation_frames": len(updates),
        "animation_duration_ms": sum(durations), "frame_duration_ms": 150, "loop": 0,
        "decoded_gif_occupancy_matches_state_in_both_columns": True,
        "occupancy_definition": "genome >= 0, channel 13; no energy/infra proxy",
        "resource_timing": "remaining reserve after this completed update's shift and injection",
    })
    save_json(output / "resource_budget.json", {
        "runs": budgets,
        "supply": "Only trail reserve injects energy, after physics/evolution. No global day/night injection. Rolling remaining capacity adds no reserve or infrastructure.",
        "accounting": "Actual supply sums per-cell energy(after injection and clamp) minus energy(before), in float64. Requested supply is also recorded; no clipping occurred. Float32 addition/subtraction produces small delivered and reserve-depletion residuals, explicitly recorded here. Remaining reserves and nominal supply match across the pair.",
        "other_fluxes": "Investment/liquidation exchanges E and infrastructure; expansion has costs; infrastructure decay removes 1% per update. Radiation and periodic kill change genomes, not E/infra. Initial infrastructure is matched but is not external energy injection.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--timing-check", action="store_true", help="Only run ten updates to estimate cost")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    configs = {name: json.loads((ROOT / f"configs/dynamic_energy_demo/{name}.json").read_text()) for name in ("fixed", "dynamic")}
    # A mismatched control should fail before any simulation work.
    fixed = json.loads(json.dumps(configs["fixed"]))
    fixed["env_options"]["shift_interval"] = configs["dynamic"]["env_options"]["shift_interval"]
    require(fixed == configs["dynamic"], "Only source shift interval may differ")
    commands, timing = [], {}
    for name in (("dynamic",) if args.timing_check else ("fixed", "dynamic")):
        config = configs[name]
        if args.timing_check:
            config.update(steps=10, checkpoint_interval=10, max_screenshots=3)
        config_path = output / "configs" / f"{name}.json"
        save_json(config_path, config)
        cmd = [sys.executable, str(RUNNER), "--config", str(config_path), "--output-dir", str(output / name)]
        commands.append(shlex.join(cmd))
        print(f"Running {name}: {config['steps']} updates", flush=True)
        started = time.perf_counter()
        with (output / f"{name}.txt").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        meta = json.loads((output / name / "meta.json").read_text())
        timing[name] = {"wall_seconds": time.perf_counter() - started, "simulation_and_capture_seconds": meta["elapsed_seconds"], "updates": config["steps"]}
    save_json(output / "timing.json", timing)
    if not args.timing_check:
        analyze(output, configs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changes = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (output / "working-tree.patch").write_bytes(changes)
    # Include new untracked source/config/test files, which git diff omits.
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if source.is_file():
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[relative] = sha256(source)
    save_json(output / "execution.json", {
        "source_commit": commit, "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "command": command, "child_commands": commands, "python": sys.version,
        "platform": platform.platform(), "packages": {p: version(p) for p in ("torch", "taichi", "numpy", "Pillow", "neat-python")},
        "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TI_CPU_MAX_NUM_THREADS", "TI_OFFLINE_CACHE_FILE_PATH")},
        "changed_source_sha256": source_hashes,
        "reproduction": "Start at source_commit; apply working-tree.patch, then overlay source/ to restore new files. Use a fresh output directory. Same seed is not a guarantee of bitwise repeatability for parallel Taichi kernels.",
    })
    prefix = " ".join(f"{key}={shlex.quote(os.environ[key])}" for key in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TI_CPU_MAX_NUM_THREADS", "TI_OFFLINE_CACHE_FILE_PATH") if key in os.environ)
    reproduction = shlex.join([sys.executable, str(Path(__file__).resolve()), "--output-dir"])
    if args.timing_check:
        reproduction += " \"${1:?Pass a fresh output directory}\" --timing-check"
    else:
        reproduction += " \"${1:?Pass a fresh output directory}\""
    (output / "reproduce.sh").write_text("#!/bin/sh\nset -eu\ncd " + shlex.quote(str(ROOT)) + "\n" + prefix + " " + reproduction + "\n")
    print(json.dumps(timing, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
