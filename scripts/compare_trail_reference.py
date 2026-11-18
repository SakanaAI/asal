"""Original-default trail comparison, with explicit GUI-compatible exports."""

import argparse
import copy
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_survival import measure_state, read_json, save_json, sha256, write_csv

RUNNER = ROOT / "external/coralai/experiments/coral_dev/headless_coral.py"
ORIGINAL = "85e5fc64cecd745b89b8fc2cd8721c77e541d7ae"
SMALL = ROOT / "results/week12/dynamic_energy_demo/seed303_452f3b2_20261111"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_state(run, update):
    path = run / "frame_states" / f"update_{update:07d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            require(int(archive["completed_update"]) == update, "State update label mismatch")
            return torch.from_numpy(archive["state"])
    return torch.load(path.with_suffix(".pt"), weights_only=True, map_location="cpu")


def load_resource(run, update):
    path = run / "frame_states" / f"resource_{update:07d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            result = {k: archive[k].item() if archive[k].ndim == 0 else archive[k] for k in archive.files}
    else:
        result = {k: v.numpy() if torch.is_tensor(v) else v for k, v in torch.load(path.with_suffix(".pt"), weights_only=True).items()}
    require(result["completed_update"] == update, "Resource update label mismatch")
    return result


def gui_values(state):
    """GUI defaults: E/I/rotation, max per channel + 1e-8, x/y -> row/column.

    This exports logical state coordinates without the interactive window/UI.
    It uses the headless completed-update phase, not run.py's pre-radiation phase.
    """
    cells = state.detach().cpu().numpy()[0]
    return np.stack([cells[k] / (cells[k].max() + np.float32(1e-8)) for k in (0, 1, 12)], axis=-1).transpose(1, 0, 2)


def rgb_image(values):
    return Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8))


def headless_image(state):
    cells = state.numpy()[0]
    channels = []
    for index in (0, 1, 13):
        values = cells[index]
        low, high = values.min(), values.max()
        channels.append((values - low) / (high - low) if high > low else np.zeros_like(values))
    return rgb_image(np.stack(channels, axis=-1).transpose(1, 0, 2))


def occupancy_image(state):
    occupied = state[0, 13].numpy().T >= 0
    values = np.zeros((*occupied.shape, 3), dtype=np.uint8)
    values[:] = (20, 29, 40)
    values[occupied] = (69, 211, 230)
    return Image.fromarray(values)


def reserve_image(capacity):
    # Fixed physical scale across treatments AND times; no per-frame rescaling.
    values = np.clip(capacity.T / 60.0, 0, 1)
    return Image.fromarray(np.stack([255 * np.sqrt(values), 220 * values, 55 * values], axis=-1).astype(np.uint8))


def small_rendering_comparison(output):
    directory = output / "small_pilot_gui_views"
    directory.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=13)
    tile, gap, row_height = 144, 15, 180
    canvas = Image.new("RGB", (975, 1170), (17, 22, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "GUI-compatible views of existing 48 x 48 pilot states | same state in each row", fill="white", font=font)
    labels = ["Old RGB: E/I/genome", "GUI: E/I/rotation", "Available E: 0..10", "Infrastructure: 0..10", "Reserve: 0..60", "Occupied (genome>=0)"]
    for column, label in enumerate(labels): draw.text((12 + column * (tile + gap), 36), label, fill="white", font=font)
    hashes = {}
    for index, (name, update) in enumerate((n, u) for n in ("fixed", "dynamic") for u in (0, 100, 200)):
        run = SMALL / name
        state = load_state(run, update)
        resource = load_resource(run, update)
        old_path = run / "screenshots" / f"step_{update:07d}.png"
        with Image.open(old_path) as old:
            require(np.array_equal(np.asarray(old), np.asarray(headless_image(state))), "Historical RGB/state mismatch")
            images = [old.copy(), rgb_image(gui_values(state))]
        for channel in (0, 1):
            gray = np.clip(state[0, channel].numpy().T / 10.0, 0, 1)
            images.append(rgb_image(np.repeat(gray[..., None], 3, axis=-1)))
        images += [reserve_image(resource["capacity"]), occupancy_image(state)]
        y = 62 + index * row_height
        draw.text((12, y), f"{name}, completed update {update}", fill="#c6cbd2", font=font)
        for column, image in enumerate(images):
            canvas.paste(image.resize((tile, tile), Image.Resampling.NEAREST), (12 + column * (tile + gap), y + 21))
        images[1].save(directory / f"{name}_gui_update_{update:07d}.png")
        hashes[str(old_path.relative_to(ROOT))] = sha256(old_path)
    draw.text((12, 1148), "Old RGB: min-max. GUI: independent channel maxima. E/I quantitative views clip at 10; reserve scale is shared.", font=font, fill="#c6cbd2")
    canvas.save(output / "rendering_comparison.png")
    return hashes


def audit_defaults(configs, output):
    # Inspect the actual original source, rather than borrowing a patches config.
    original_sources = {}
    for path in ("environments.py", "run.py", "experiment.py"):
        relative = f"external/coralai/experiments/coral_dev/{path}"
        original_sources[path] = subprocess.check_output(["git", "show", f"{ORIGINAL}:{relative}"], cwd=ROOT, text=True)
    namespace = {"__name__": "reference_trail_defaults"}
    exec(compile(original_sources["environments.py"], "original_environments.py", "exec"), namespace)
    env = namespace["TrailEnvironment"]()
    mask, _ = env._build_trail(400, 400, torch.device("cpu"))
    ys = torch.where(mask.any(dim=0))[0]
    moving = configs["dynamic"]["env_options"]
    shifts = configs["dynamic"]["steps"] // moving["shift_interval"]
    displacement = shifts * moving["shift_cells"]
    require(int(ys.max()) + displacement < 400, "Chosen motion would wrap resources across the edge")
    equalized = copy.deepcopy(configs["fixed"])
    equalized["env_options"]["shift_interval"] = moving["shift_interval"]
    require(equalized == configs["dynamic"], "Only resource-motion treatment may differ")
    for key in ("spawn", "trail_width", "max_capacity", "inject_rate"):
        require(getattr(env, key) == configs["fixed"]["env_options"][key], f"Not an original environment default: {key}")
    save_json(output / "reference.json", {
        "basis": "Executable defaults from 85e5fc6, NOT verified settings of the teammate's exact screenshot/run.",
        "original_commit": ORIGINAL, "original_source_sha256": {k: __import__("hashlib").sha256(v.encode()).hexdigest() for k, v in original_sources.items()},
        "evidence_gap": "No saved original trail launch/config/initial state was found. Tracked older headless metadata and config_test.json are patches; the orphan snapshot has no run configuration.",
        "grid": [400, 400], "seed": 42, "initialization": "spread: original random roughly 20% occupancy, 50 NEAT genomes; no corner-founder substitution",
        "trail_defaults": env.to_dict() | {"spawn": env.spawn},
        "physics": {"infra_decay": 0, "defense_coeff": 0, "radiation_interval": 50, "radiation_spots": 5, "live_radius_cells": 2, "dead_radius_cells": 4, "kill_radius_cells": 5, "kill_interval": 50, "cleanup_interval": 100, "emergency_pop_cap": 1000},
        "boundary": "Simulator neighbor/chunk operations are periodic. Trail drawing is clipped to the grid. Existing translation is torch.roll, but this treatment's support never crosses the edge.",
        "movement": {"shift_cells": moving["shift_cells"], "shift_interval": moving["shift_interval"], "units": "integer y cells, applied before release at completed updates divisible by interval", "reason": "Half the four-cell trail width per movement; eight movements, sixteen cells total, rather than copying small-grid motion.", "initial_y_bounds": [int(ys.min()), int(ys.max())], "maximum_y_bounds": [int(ys.min()) + displacement, int(ys.max()) + displacement], "resource_wraps": 0},
        "source_region_definition": "remaining capacity > 0 after this completed update's translation and release; stored energy and occupancy do not define the source region",
        "rendering": "GUI-compatible default E/I/rotation with per-channel max+1e-8 normalization. Quantitative reserve always 0..60; occupancy is genome >= 0. No UI/manual channel selections are inferred.",
        "capture_phase": "After NN, physics, evolution/killing, resource shift/release, radiation and cleanup; completed_update=timestep. GUI run.py renders before radiation/cleanup and updates its counter later, so exact same-label GUI screenshot parity is not claimed.",
        "duration": "Explicit bounded 1000 updates; original GUI default has no fixed stop. This target is requested, not inferred from a screenshot.",
    })


def run(config, directory):
    directory.mkdir()
    path = directory / "input_config.json"
    save_json(path, config)
    command = [sys.executable, str(RUNNER), "--config", str(path), "--output-dir", str(directory / "run")]
    save_json(directory / "command.json", {"argv": command, "cwd": str(ROOT)})
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "external/coralai"), str(ROOT / "external/coralai/coralai/dependencies/PyTorch-NEAT"), environment.get("PYTHONPATH", "")])
    started = time.perf_counter()
    with (directory / "run.txt").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
    return time.perf_counter() - started


def frame_image(update, states, resources, metrics):
    panel = 320
    canvas = Image.new("RGB", (696, 1180), (17, 22, 31))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    draw.text((12, 7), f"Original executable trail defaults | 400 x 400 | seed 42 | completed update {update:04d}", fill="white", font=font)
    draw.text((12, 29), "GUI-compatible completed-update views; not reproduction of a verified teammate run", fill="#c6cbd2", font=font)
    for column, name in enumerate(("fixed", "dynamic")):
        x = 12 + column * 348
        motion = "fixed reserve" if name == "fixed" else "reserve +2 cells / 125 updates"
        draw.text((x, 54), f"{name}: {motion}", fill="white", font=font)
        draw.text((x, 77), "E / infrastructure / rotation (auto-max)", fill="#c6cbd2", font=font)
        canvas.paste(rgb_image(gui_values(states[name])).resize((panel, panel), Image.Resampling.NEAREST), (x, 98))
        m = metrics[name]
        draw.text((x, 430), f"Occupied {m['occupied_cells']}: source {m['occupied_inside_source']}, outside {m['occupied_outside_source']}", fill="#54d9ed", font=font)
        canvas.paste(occupancy_image(states[name]).resize((panel, panel), Image.Resampling.NEAREST), (x, 453))
        draw.text((x, 785), "Remaining reserve: fixed scale 0..60", fill="#ffe08a", font=font)
        canvas.paste(reserve_image(resources[name]["capacity"]).resize((panel, panel), Image.Resampling.NEAREST), (x, 808))
        draw.text((x, 1140), f"Stored E={m['total_energy']:.1f}; infra={m['total_infra']:.1f}", fill="#c6cbd2", font=font)
        draw.text((x, 1160), f"Cumulative delivered energy: {m['cumulative_delivered']:.2f}", fill="#c6cbd2", font=font)
    return canvas


def analyze(output, configs, frame_renderer=frame_image, title="Original-default trail comparison | 400 x 400 | seed 42"):
    steps = configs["fixed"]["steps"]
    updates = list(range(0, steps + 1, configs["fixed"]["screenshot_interval"]))
    logs = {}
    for name in configs:
        with (output / name / "run/resource_metrics.csv").open() as stream:
            logs[name] = {int(r["completed_update"]): {k: int(v) if k in {"completed_update", "occupied_cells", "occupied_inside_source", "occupied_outside_source", "source_cells"} else float(v) for k, v in r.items()} for r in csv.DictReader(stream)}
        require(list(logs[name]) == list(range(steps + 1)), "Missing completed-update resource/count measurements")
    fixed_initial = torch.load(output / "fixed/run/initial_state.pt", weights_only=True)
    dynamic_initial = torch.load(output / "dynamic/run/initial_state.pt", weights_only=True)
    require(torch.equal(fixed_initial, dynamic_initial), "Starting states differ")
    initial_resource = load_resource(output / "fixed/run", 0)
    require(np.array_equal(initial_resource["capacity"], load_resource(output / "dynamic/run", 0)["capacity"]), "Starting reserves differ")
    # Independent reserve-only accounting: depletion commutes with translation.
    # No organism updates or simulation inference are executed here.
    expected = torch.from_numpy(initial_resource["capacity"].copy())
    requested = 0.0
    reserve_checks = 0
    for update in range(steps + 1):
        if update:
            injection = expected.clamp(max=configs["fixed"]["env_options"]["inject_rate"])
            requested += float(injection.double().sum())
            expected.sub_(injection).clamp_(min=0)
        for name in configs:
            row = logs[name][update]
            require(abs(row["cumulative_requested"] - requested) < 1e-7, "Unexpected reserve creation/release")
            require(abs(row["remaining_capacity"] - float(expected.double().sum())) < 1e-7, "Reserve total/depletion mismatch")
        if update in updates:
            for name, config in configs.items():
                options = config["env_options"]
                displacement = update // options["shift_interval"] * options["shift_cells"] if options["shift_interval"] else 0
                resource = load_resource(output / name / "run", update)
                require(np.array_equal(resource["capacity"], torch.roll(expected, displacement, dims=1).numpy()), "Resource shape/displacement mismatch")
                reserve_checks += 1
    (output / "gui_views").mkdir(exist_ok=True)
    frames, observations, pairs = [], [], []
    previous = {}
    snapshot_checks = 0
    with tempfile.TemporaryDirectory() as temporary:
        for update in updates:
            states, resources, metrics = {}, {}, {}
            for name in configs:
                run_dir = output / name / "run"
                state = states[name] = load_state(run_dir, update)
                resource = resources[name] = load_resource(run_dir, update)
                require(bool(torch.isfinite(state).all()), "Nonfinite state")
                require(bool((state[0, 13] == state[0, 13].floor()).all()), "Noninteger genomes")
                occupied = state[0, 13].numpy() >= 0
                source = resource["capacity"] > 0
                row = metrics[name] = logs[name][update]
                require(row["occupied_cells"] == int(occupied.sum()), "State/count timing mismatch")
                require(row["occupied_inside_source"] == int((occupied & source).sum()), "State/source-count timing mismatch")
                with Image.open(run_dir / "screenshots" / f"step_{update:07d}.png") as image:
                    require(np.array_equal(np.asarray(image), np.asarray(headless_image(state))), "State/RGB mismatch")
                if update in (0, steps):
                    checkpoint = Path(temporary) / "state.pt"
                    torch.save(state, checkpoint)
                    measure_state(checkpoint, [400, 400], image_path=run_dir / "screenshots" / f"step_{update:07d}.png")
                rgb_image(gui_values(state)).save(output / "gui_views" / f"{name}_update_{update:07d}.png")
                prior = previous.get(name)
                observations.append({"configuration": name, **row,
                    "changed_occupied_locations_since_previous_capture": int((occupied != prior).sum()) if prior is not None else 0,
                    "source_displacement_y_cells": update // 125 * 2 if name == "dynamic" else 0,
                    "state_sha256": sha256(run_dir / "frame_states" / f"update_{update:07d}.npz")})
                previous[name] = occupied.copy()
                snapshot_checks += 1
                if update == steps:
                    final = torch.load(run_dir / f"checkpoint_{steps:07d}/substrate.pt", weights_only=True)
                    require(torch.equal(state, final), "Final checkpoint/capture mismatch")
            # Compare both with the fixed initial region to expose mask-only changes.
            initial_source = initial_resource["capacity"] > 0
            pairs.append({"completed_update": update,
                "differing_occupied_locations": int((previous["fixed"] != previous["dynamic"]).sum()),
                "dynamic_occupied_in_initial_source": int((previous["dynamic"] & initial_source).sum()),
                "fixed_occupied_in_moving_source": int((previous["fixed"] & (resources["dynamic"]["capacity"] > 0)).sum()),
                "maximum_abs_state_difference": float((states["fixed"] - states["dynamic"]).abs().max())})
            frame = frame_renderer(update, states, resources, metrics)
            frames.append(frame)
    frames[0].save(output / "comparison.gif", save_all=True, append_images=frames[1:], duration=180, loop=0, disposal=2, optimize=False)
    # Verify every decoded frame's occupied locations against its captured state.
    with Image.open(output / "comparison.gif") as animation:
        require(animation.n_frames == len(updates) and animation.info["loop"] == 0, "Invalid animation count/loop")
        for index, update in enumerate(updates):
            animation.seek(index)
            require(animation.info["duration"] == 180, "Invalid animation timing")
            decoded = np.asarray(animation.convert("RGB"))
            for column, name in enumerate(("fixed", "dynamic")):
                image = occupancy_image(load_state(output / name / "run", update)).resize((320, 320), Image.Resampling.NEAREST)
                expected_mask = np.asarray(image)[..., 1] > 100
                x = 12 + column * 348
                require(np.array_equal(decoded[453:773, x:x + 320, 1] > 100, expected_mask), "GIF occupancy synchronization mismatch")
    frames[-1].save(output / "endpoint.png")
    write_csv(output / "measurements.csv", observations)
    write_csv(output / "paired_measurements.csv", pairs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for name in configs:
        axes[0].plot(list(logs[name]), [r["occupied_cells"] for r in logs[name].values()], label=name)
    axes[0].set_ylabel("Total occupied cells")
    axes[0].set_title(title)
    axes[0].legend()
    axes[1].plot(updates, [logs["dynamic"][u]["occupied_inside_source"] for u in updates], label="Dynamic occupancy in current source")
    axes[1].plot(updates, [r["fixed_occupied_in_moving_source"] for r in pairs], linestyle="--", label="Fixed-run occupancy in the same moving mask")
    axes[1].set_ylabel("Occupied cells in source")
    axes[1].legend(fontsize=8)
    axes[2].plot(updates, [r["differing_occupied_locations"] for r in pairs])
    axes[2].set_ylabel("Locations differing\nbetween treatments")
    axes[2].set_xlabel("Completed update; dotted lines mark source translations")
    for axis in axes:
        axis.grid(alpha=0.2)
        for u in range(125, steps + 1, 125): axis.axvline(u, color="gray", linewidth=0.6, linestyle=":")
    figure.savefig(output / "trajectory.png", dpi=150)
    plt.close(figure)
    results = []
    budgets = {}
    for name in configs:
        first, last = logs[name][0], logs[name][steps]
        budget = read_json(output / name / "run/meta.json")["resource_budget"]
        budget["reserve_depletion_minus_requested"] = budget["initial_capacity"] - budget["remaining_capacity"] - budget["cumulative_requested"]
        budget["requested_minus_delivered"] = budget["cumulative_requested"] - budget["cumulative_delivered"]
        budgets[name] = budget
        results.append({"configuration": name, "initial_occupied": first["occupied_cells"], "final_occupied": last["occupied_cells"],
            "final_occupied_inside_source": last["occupied_inside_source"], "final_occupied_outside_source": last["occupied_outside_source"],
            "actual_energy_delivered": last["cumulative_delivered"], "requested_energy": last["cumulative_requested"],
            "clipped_energy": last["cumulative_clipped"], "remaining_reserve": last["remaining_capacity"],
            "simulation_capture_seconds": read_json(output / name / "run/meta.json")["elapsed_seconds"]})
    write_csv(output / "results.csv", results)
    maximum_difference = max(abs(logs["fixed"][u]["cumulative_delivered"] - logs["dynamic"][u]["cumulative_delivered"]) for u in range(steps + 1))
    save_json(output / "resource_budget.json", {"runs": budgets, "initial_energy": logs["fixed"][0]["total_energy"], "initial_infrastructure": logs["fixed"][0]["total_infra"], "maximum_actual_delivery_difference": maximum_difference,
        "accounting": "Actual after-minus-before addition/clamp at every update; no rescaling. Request and remaining-reserve totals independently checked every update. Float32 residuals retained and reported.",
        "source_region": "remaining capacity > 0 after this completed update's release; mask motion alone can change inside/outside counts"})
    save_json(output / "validation.json", {"starting_states_bitwise_equal": True, "starting_reserves_bitwise_equal": True,
        "initial_state_sha256": {n: sha256(output / n / "run/initial_state.pt") for n in configs},
        "per_update_budget_checks": (steps + 1) * 2, "reserve_location_checks": reserve_checks, "state_rgb_source_measurement_matches": snapshot_checks,
        "decoded_gif_occupied_panel_matches": len(updates) * 2, "final_checkpoint_matches": 2,
        "animation_frames": len(updates), "frame_duration_ms": 180, "resource_wraps": 0,
        "maximum_differing_occupied_locations": max(r["differing_occupied_locations"] for r in pairs),
        "pre_first_movement": {"maximum_abs_state_difference": max(r["maximum_abs_state_difference"] for r in pairs if r["completed_update"] < 125),
            "maximum_differing_occupied_locations": max(r["differing_occupied_locations"] for r in pairs if r["completed_update"] < 125)},
        "first_saved_update_with_different_occupancy": next((r["completed_update"] for r in pairs if r["differing_occupied_locations"]), None)})


def provenance(output, inherited_hashes, commands):
    sources = ["scripts/compare_trail_reference.py", "tests/test_trail_reference.py", "configs/trail_reference_comparison/fixed.json", "configs/trail_reference_comparison/dynamic.json", "scripts/setup_novelty_env.sh", "configs/asal_novelty_pilot/requirements.lock.txt", "external/coralai/experiments/coral_dev/headless_coral.py", "external/coralai/experiments/coral_dev/environments.py", "external/coralai/experiments/coral_dev/experiment.py"]
    source_hashes = {}
    for relative in sources:
        source = ROOT / relative
        if source.exists():
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[relative] = sha256(source)
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    environment = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TI_CPU_MAX_NUM_THREADS", "TI_OFFLINE_CACHE_FILE_PATH", "MPLCONFIGDIR")}
    save_json(output / "execution.json", {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "child_commands": commands, "environment": environment,
        "versions": {p: version(p) for p in ("torch", "taichi", "numpy", "Pillow", "neat-python", "matplotlib")},
        "source_sha256": source_hashes, "protected_existing_sha256": inherited_hashes,
        "setup": "sh scripts/setup_novelty_env.sh; persistent separate .venv-novelty and .runtime-novelty, not temporary Python",
        "restore": "Checkout source_commit, apply working-tree.patch, overlay source/, set up the isolated environment, and reproduce.sh with a fresh directory. Only --historical-rendering requires original local small-pilot raw states.",
        "nondeterminism": "Identical initial tensors do not guarantee bitwise-identical later CPU states; parallel atomic floating-point operations can vary. Serial CPU requested and both treatments use the same seed/settings."})
    (output / "requirements-used.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    lines = ["#!/bin/sh", "set -eu", "cd " + shlex.quote(str(ROOT))]
    lines += ["export " + k + "=" + shlex.quote(v) for k, v in environment.items() if v is not None]
    lines.append(shlex.join([str(ROOT / ".venv-novelty/bin/python"), "scripts/compare_trail_reference.py", "--output-dir"]) + ' "${1:?Pass a fresh output directory}"')
    (output / "reproduce.sh").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true", help="Rebuild derived outputs from this task's completed pair; requires local raw states")
    parser.add_argument("--historical-rendering", action="store_true", help="Also compare old 48x48 views; requires the original local small-pilot raw states")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    configs = {n: read_json(ROOT / f"configs/trail_reference_comparison/{n}.json") for n in ("fixed", "dynamic")}
    started = time.perf_counter()
    if not args.analyze_only: output.mkdir(parents=True, exist_ok=False)
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "scripts/coralai_search.py", ROOT / "configs/coralai_supervised.search.json")}
    if args.historical_rendering:
        require((SMALL / "fixed/frame_states").is_dir(), "Historical rendering requires the original local small-pilot raw states; omit --historical-rendering for a fresh comparison")
        protected.update(small_rendering_comparison(output))
    audit_defaults(configs, output)
    timing_path = output / "timing/run/meta.json"
    if not args.analyze_only:
        calibration = copy.deepcopy(configs["fixed"])
        calibration.update(steps=50, max_screenshots=3, checkpoint_interval=50)
        run(calibration, output / "timing")
    require(timing_path.is_file(), "Analysis requires the local timing/run/meta.json produced by a fresh comparison")
    timing = read_json(timing_path)
    save_json(output / "timing_plan.json", {"timing_updates": 50, "grid": [400, 400], "measured_seconds": timing["elapsed_seconds"], "linear_pair_estimate_seconds": timing["elapsed_seconds"] / 50 * 2000, "decision": "Keep original grid and physics; use lossless NPZ captures. Timing is measured in this fresh output directory."})
    require(timing["elapsed_seconds"] * 40 < 1200, "Timing estimate exceeds 20 minutes; report before running the pair")
    wall, commands = {}, []
    if not args.analyze_only:
        for name, config in configs.items():
            print(f"Running {name}: 400x400, 1000 completed updates", flush=True)
            wall[name] = run(config, output / name)
            commands.append(read_json(output / name / "command.json"))
    else:
        wall = read_json(output / "completed.json")["run_wall_seconds"]
    analyze(output, configs)
    for relative, digest in protected.items(): require(sha256(ROOT / relative) == digest, "Existing artifact/teammate source changed")
    if not args.analyze_only:
        save_json(output / "completed.json", {"run_wall_seconds": wall, "total_seconds_including_rendering_analysis": time.perf_counter() - started})
        provenance(output, protected, commands)
    print(f"Comparison saved: {output}", flush=True)


if __name__ == "__main__":
    main()
