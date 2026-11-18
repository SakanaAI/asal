"""Serial Coralai rollouts for ASAL's existing temporal objective and optimizer.

The scoring image is an explicit occupancy-only rendering adaptation. The
original ASAL metric, including its zero first row/masked entries, is unchanged.
The supervised multi-seed adapter remains independent.
"""

import copy
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from scripts.coralai_search import ROOT, RUNNER, decode, read_json, save_json, sha256
from scripts.clip_scoring import ClipScorer, load_clip_image
from scripts.evaluate_survival import measure_state
from substrates.coralai import render_checkpoint, render_state

sys.path[:0] = [str(ROOT / "external/coralai"), str(ROOT / "external/coralai/coralai/dependencies/PyTorch-NEAT")]
from coralai.provenance import effective_configuration, simulator_fingerprint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_contract(config, settings, updates, model_sha256):
    """Identity of the simulator, full effective config, and scoring/sampling path."""
    return {"schema_version": 1, "simulator": simulator_fingerprint(),
            "effective_config": effective_configuration(config),
            "scoring": {"view": settings["scoring_view"], "model": settings["model"],
                        "revision": settings["revision"], "weights_sha256": model_sha256,
                        "implementation_sha256": {p: sha256(ROOT / p) for p in
                            ("substrates/coralai.py", "scripts/clip_scoring.py", "foundation_models/clip.py",
                             "asal_metrics.py", "scripts/coralai_temporal.py")}},
            "sampled_updates": list(updates), "initial_update_scored": False,
            "capture_phase": "after all operations at completed update"}


def reference_files(updates):
    return [f"{folder}/{prefix}_{update:07d}.{suffix}" for update in [0, *updates]
            for folder, prefix, suffix in (("frame_states", "update", "pt"),
                                           ("frame_states", "resource", "pt"),
                                           ("screenshots", "step", "png"))]


def validate_reference(run, expected):
    """Never accept old/pre-fix references solely on configuration or Git HEAD."""
    try:
        manifest = read_json(run / "reference_compatibility.json")
        require(manifest["contract"] == expected, "implementation/config/scoring/sampling contract differs")
        meta = read_json(run / "meta.json")
        require(meta["simulator_provenance"] == expected["simulator"], "simulator implementation differs")
        require(effective_configuration(meta["effective_config"]) == expected["effective_config"], "effective configuration differs")
        require(meta["completed_steps"] == expected["effective_config"]["steps"], "run is incomplete")
        files = reference_files(expected["sampled_updates"])
        require(set(manifest["capture_sha256"]) == set(files), "capture coverage differs")
        for name in files:
            require(sha256(run / name) == manifest["capture_sha256"][name], f"capture changed: {name}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Cached reference {run} is incompatible or unverifiable: {error}. "
                         "Require fresh evaluation with --coralai_fresh_references.") from error


def reference_run(demo, name, fresh=False):
    path = demo / name
    return None if fresh or not path.is_dir() else path


def temporal_score(embeddings):
    import jax
    import jax.numpy as jnp
    from asal_metrics import calc_open_endedness_score

    z = np.asarray(embeddings)
    require(z.ndim == 2 and len(z) >= 2 and np.isfinite(z).all(), "Need at least two finite image embeddings")
    require(np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4), "CLIP embeddings must be normalized")
    # Call the implemented ASAL objective, rather than a pixel/adjacent-frame proxy.
    loss = float(jax.device_get(calc_open_endedness_score(jnp.asarray(z))))
    closest = np.tril(z @ z.T, k=-1).max(axis=1)
    return {"loss_oe": loss, "novelty": (len(z) - 1) / len(z) - loss,
            "closest_prior_similarity": closest.tolist()}


def apply_parameters(config, vector, bounds):
    require(set(bounds) <= {"shift_interval", "shift_cells"}, "Only reserve translation may be searched")
    result = copy.deepcopy(config)
    result["env_options"].update(decode(vector, bounds))
    return result


def validate_setup(config, settings):
    from jsonschema import Draft202012Validator

    Draft202012Validator(read_json(ROOT / "schemas/coralai_temporal_search.schema.json")).validate(settings)
    require(config["env"] == "trail" and config["backend"] == config["device"] == "cpu" and not config.get("resume_from"), "Use fresh CPU trail runs")
    require(config["seed"] == settings["simulation_seed"], "Simulation seeds differ")
    updates = settings["sampled_updates"]
    require(updates == sorted(updates) and updates[-1] == config["steps"], "Updates must be ordered and include the final update")
    require(all(u % config["screenshot_interval"] == 0 for u in updates), "Scored updates must have saved states/images")
    require(config.get("save_frame_states") is True, "Save matching frame states")
    require(config.get("frame_state_format", "pt") == "pt", "Temporal evaluation requires PT frame captures")
    for name, bounds in settings["parameters"].items():
        require(bounds["min"] < bounds["max"], f"Invalid bounds: {name}")
        require(bounds["min"] <= config["env_options"][name] <= bounds["max"], f"Hand configuration outside bounds: {name}")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TemporalEvaluator:
    def __init__(self, settings, output):
        from huggingface_hub import snapshot_download

        self.settings = settings
        self.output = output
        model = Path(snapshot_download(settings["model"], revision=settings["revision"], local_files_only=True))
        self.scorer = ClipScorer("cpu", model_path=model)
        self.model_sha256 = sha256(model / "flax_model.msgpack")
        save_json(output / "scorer.json", {
            "model": settings["model"], "revision": settings["revision"],
            "weights_sha256": sha256(model / "flax_model.msgpack"),
            "device": str(self.scorer.device), "load_seconds": self.scorer.model_load_seconds,
            "scoring_view": "Adaptation: binary genome >= 0, white occupied/black empty, x/y transpose, nearest 224x224, float32 /255; existing CLIP normalization. No energy, infrastructure, reserve, genome identity, labels, or text prompt.",
            "objective": "Unmodified asal_metrics.calc_open_endedness_score: mean row maximum of tril(z @ z.T, -1); minimize. First row is zero; masked zero entries floor negative similarities at zero.",
            "novelty_display": "(T-1)/T - loss_oe; constant normalized sequence has novelty approximately zero. This is a display transform, not another objective.",
            "sampled_updates": settings["sampled_updates"], "step_zero_scored": False,
        })

    def simulate(self, config, directory):
        directory.mkdir(parents=True, exist_ok=False)
        save_json(directory / "input_config.json", config)
        command = [sys.executable, str(RUNNER), "--config", str(directory / "input_config.json"), "--output-dir", str(directory / "run")]
        save_json(directory / "command.json", {"argv": command, "cwd": str(ROOT)})
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "external/coralai"), str(ROOT / "external/coralai/coralai/dependencies/PyTorch-NEAT"), environment.get("PYTHONPATH", "")])
        started = time.perf_counter()
        with (directory / "simulation.txt").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        return directory / "run", time.perf_counter() - started

    def evaluate(self, config, directory, *, reuse=None, updates=None):
        updates = updates or self.settings["sampled_updates"]
        contract = reference_contract(config, self.settings, updates, self.model_sha256)
        if reuse is not None:
            validate_reference(reuse.resolve(), contract)
        if reuse is None:
            run, simulation_seconds = self.simulate(config, directory)
        else:
            directory.mkdir(parents=True, exist_ok=False)
            run, simulation_seconds = reuse.resolve(), 0.0
        meta = read_json(run / "meta.json")
        require(meta.get("simulator_provenance") == contract["simulator"], "Simulator source changed or is unverifiable; require fresh evaluation")
        require(effective_configuration(meta["effective_config"]) == contract["effective_config"], "Effective run configuration differs")
        require(meta["completed_steps"] == config["steps"] and meta["seed"] == config["seed"], "Incomplete/mismatched run")
        for key in ("shape", "steps", "env", "env_param", "env_options", "infra_decay", "defense_coeff", "radiate_interval", "cleanup_interval", "emergency_pop_cap"):
            require(meta["effective_config"][key] == config[key], f"Reference setting differs: {key}")
        save_json(directory / "effective_config.json", read_json(run / "config.json"))
        (directory / "scoring_images").mkdir()
        measurements, images = [], []
        started = time.perf_counter()
        for update in [0, *updates]:
            state_path = run / "frame_states" / f"update_{update:07d}.pt"
            screenshot = run / "screenshots" / f"step_{update:07d}.png"
            measured, check = measure_state(state_path, [config["shape"]] * 2, image_path=screenshot)
            resource = torch.load(run / "frame_states" / f"resource_{update:07d}.pt", weights_only=True)
            require(resource["completed_update"] == update, "Resource/state capture is not synchronized")
            image = render_checkpoint(state_path, view="occupancy")
            image_path = directory / "scoring_images" / f"update_{update:07d}.png"
            Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(image_path)
            require(np.array_equal(load_clip_image(image_path), image), "Saved scoring pixels differ from model input")
            if update:
                images.append(image)
            measurements.append({"completed_update": update, "occupied_cells": measured["living_cell_count"],
                "occupied_fraction": measured["living_cell_fraction"], "occupied_genome_ids": measured["occupied_genome_ids"],
                "cumulative_delivered": resource["cumulative_delivered"], "cumulative_requested": resource["cumulative_requested"],
                "cumulative_clipped": resource["cumulative_clipped"], "remaining_reserve": resource["remaining_capacity"],
                "state_sha256": sha256(state_path), "rgb_sha256": sha256(screenshot), "scoring_image_sha256": sha256(image_path),
                "rgb_matches_state": check["matches_saved_screenshot_pixels"]})
        embeddings = np.asarray(self.scorer.image_embeddings(images))
        score = temporal_score(embeddings)
        np.save(directory / "embeddings.npy", embeddings)
        for index, row in enumerate(measurements):
            row["closest_prior_similarity"] = score["closest_prior_similarity"][index - 1] if index else None
            row["novelty_contribution"] = (1.0 - row["closest_prior_similarity"]) if index > 1 else None
        write_csv(directory / "measurements.csv", measurements)
        final = torch.load(run / meta["final_checkpoint"] / "substrate.pt", weights_only=True)
        captured_final = torch.load(run / "frame_states" / f"update_{config['steps']:07d}.pt", weights_only=True)
        require(torch.equal(final, captured_final), "Final checkpoint/frame capture mismatch")
        score.update(run=str(run), simulation_seconds=simulation_seconds, scoring_seconds=time.perf_counter() - started,
                     measurements=measurements, resource_budget=meta["resource_budget"], reused=bool(reuse))
        save_json(directory / "scores.json", score)
        if reuse is None:
            save_json(run / "reference_compatibility.json", {"contract": contract,
                "capture_sha256": {p: sha256(run / p) for p in reference_files(updates)}})
        save_json(directory / "reference_contract.json", contract)
        print(f"{directory.name}: loss={score['loss_oe']:.8f}, novelty={score['novelty']:.8f}; sim={simulation_seconds:.2f}s score={score['scoring_seconds']:.2f}s", flush=True)
        return score

    def confound_control(self, run):
        updates = self.settings["sampled_updates"]
        first = updates[0]
        state = torch.load(run / "frame_states" / f"update_{first:07d}.pt", weights_only=True)
        fixed_occupancy = render_state(state, "occupancy")
        with Image.open(run / "screenshots" / f"step_{first:07d}.png") as source:
            rgb = np.asarray(source).copy()
        directory = self.output / "control"
        directory.mkdir()
        sequences = {"static_rgb": [], "moving_energy_rgb": [], "frozen_occupancy": []}
        for update in updates:
            changed = state.clone()
            distance = (update // 25) * 4
            changed[0, 0] = torch.roll(state[0, 0], distance, dims=1)
            occupancy = render_state(changed, "occupancy")
            require(np.array_equal(occupancy, fixed_occupancy), "Resources leaked into occupancy rendering")
            moving = rgb.copy()
            moving[..., 0] = np.roll(rgb[..., 0], distance, axis=0)
            for name, values in (("static_rgb", rgb), ("moving_energy_rgb", moving)):
                image = Image.fromarray(values).resize((224, 224), Image.Resampling.LANCZOS)
                image.save(directory / f"{name}_{update:07d}.png")
                sequences[name].append(np.asarray(image, dtype=np.float32) / 255)
            sequences["frozen_occupancy"].append(occupancy)
            Image.fromarray(np.rint(occupancy * 255).astype(np.uint8)).save(directory / f"frozen_occupancy_{update:07d}.png")
        scores = {}
        for name, images in sequences.items():
            embeddings = np.asarray(self.scorer.image_embeddings(images))
            np.save(directory / f"{name}_embeddings.npy", embeddings)
            scores[name] = temporal_score(embeddings)
        require(abs(scores["frozen_occupancy"]["novelty"]) < 1e-5, "Unchanged scoring image should have zero novelty")
        save_json(directory / "scores.json", {"description": "Synthetic control: freeze infrastructure, genome channel, and occupancy; translate only red energy pixels from the original headless RGB renderer. No simulator/ecosystem change. Binary scoring view stays identical.", "sampled_updates": updates, "scores": scores,
            "rgb_resource_only_novelty_gain": scores["moving_energy_rgb"]["novelty"] - scores["static_rgb"]["novelty"]})


def compare_budgets(scores):
    reference = scores[0]
    initial = torch.load(Path(reference["run"]) / "initial_state.pt", weights_only=True)
    maximum = 0.0
    for score in scores:
        other = torch.load(Path(score["run"]) / "initial_state.pt", weights_only=True)
        require(torch.equal(initial, other), "Initial states differ across configurations")
        for left, right in zip(reference["measurements"], score["measurements"]):
            require(left["completed_update"] == right["completed_update"], "Sample updates differ")
            require(left["cumulative_requested"] == right["cumulative_requested"] and left["remaining_reserve"] == right["remaining_reserve"], "Nominal resource budgets differ")
            require(right["cumulative_clipped"] == 0, "Energy clipping changes actual supply")
            maximum = max(maximum, abs(left["cumulative_delivered"] - right["cumulative_delivered"]))
    require(maximum < 0.005, "Actual delivered budgets differ beyond float32 tolerance")
    return {"initial_states_identical": True, "nominal_budgets_identical_at_all_samples": True, "maximum_actual_supply_difference": maximum, "absolute_supply_tolerance": 0.005, "clipping": 0}


def make_artifacts(output, configs, scores, selected_directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["fixed", "hand", "selected"]
    updates = [row["completed_update"] for row in scores[0]["measurements"]]
    directories = [output / "fixed", output / "hand", selected_directory]
    rows, frames = [], []
    initial = torch.load(Path(scores[0]["run"]) / "initial_state.pt", weights_only=True)[0, 13] >= 0
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for name, config, score in zip(names, configs, scores):
        counts = [r["occupied_cells"] for r in score["measurements"]]
        changed = []
        for update in updates:
            state = torch.load(Path(score["run"]) / "frame_states" / f"update_{update:07d}.pt", weights_only=True)
            changed.append(int(((state[0, 13] >= 0) != initial).sum()))
        rows.append({"configuration": name, "shift_interval": config["env_options"]["shift_interval"], "shift_cells": config["env_options"]["shift_cells"],
                     "loss_oe_lower_is_better": score["loss_oe"], "temporal_novelty_display": score["novelty"],
                     "energy_delivered": score["resource_budget"]["cumulative_delivered"],
                     "initial_occupied": counts[0], "final_occupied": counts[-1], "minimum_sampled_occupied": min(counts),
                     "mean_sampled_occupied_excluding_initial": float(np.mean(counts[1:])), "final_cells_changed_from_initial": changed[-1]})
        style = {"linestyle": "--", "marker": "x"} if name == "selected" else {"linestyle": "-", "marker": "o"}
        axes[0].plot(updates, counts, markersize=4, label=name, **style)
        axes[1].plot(updates[2:], [r["novelty_contribution"] for r in score["measurements"][2:]], markersize=4, label=name, **style)
    axes[0].set_ylabel("Occupied cells (genome >= 0)")
    same_selection = configs[1]["env_options"] == configs[2]["env_options"]
    axes[0].set_title("Finite temporal novelty | seed 303 | occupancy-only CLIP\n" + ("Selected equals hand: no search improvement" if same_selection else "Fixed, hand-configured, and search-selected settings"))
    axes[0].legend(ncol=3)
    axes[1].set_ylabel("1 - closest prior cosine\n(occupancy CLIP embeddings)")
    axes[1].set_xlabel("Completed update")
    for axis in axes: axis.grid(alpha=0.25)
    figure.savefig(output / "trajectory.png", dpi=160)
    plt.close(figure)
    write_csv(output / "comparison.csv", rows)
    font = ImageFont.load_default(size=14)
    maximum = None
    for update in updates:
        canvas = Image.new("RGB", (708, 795), (17, 22, 31))
        draw = ImageDraw.Draw(canvas)
        label = "selected = hand" if same_selection else "seed 303"
        draw.text((12, 8), f"ASAL finite temporal novelty | completed update {update:04d} | {label}", font=font, fill="white")
        for column, (name, config, score, directory) in enumerate(zip(names, configs, scores, directories)):
            x = 12 + column * 236
            opts = config["env_options"]
            draw.text((x, 35), f"{name}: interval {opts['shift_interval']}, shift {opts['shift_cells']}", font=font, fill="white")
            run = Path(score["run"])
            state = torch.load(run / "frame_states" / f"update_{update:07d}.pt", weights_only=True)
            resource = torch.load(run / "frame_states" / f"resource_{update:07d}.pt", weights_only=True)
            with Image.open(run / "screenshots" / f"step_{update:07d}.png") as rgb:
                canvas.paste(rgb.resize((212, 212), Image.Resampling.NEAREST), (x, 85))
            with Image.open(directory / "scoring_images" / f"update_{update:07d}.png") as occupancy:
                canvas.paste(occupancy.resize((212, 212), Image.Resampling.NEAREST), (x, 328))
            if maximum is None: maximum = float(resource["capacity"].max())
            reserve = resource["capacity"].numpy().T / maximum
            colors = np.stack([255 * np.sqrt(reserve), 220 * reserve, 55 * reserve], axis=-1).clip(0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(colors).resize((212, 212), Image.Resampling.NEAREST), (x, 571))
            draw.text((x, 62), "Original RGB (E / I / genome)", font=font, fill="#d4d8de")
            draw.text((x, 305), f"Scored occupancy: {int((state[0,13] >= 0).sum())}", font=font, fill="#d4d8de")
            draw.text((x, 548), "Food reserve (same scale)", font=font, fill="#ffe08a")
        frames.append(canvas)
    frames[0].save(output / "comparison.gif", save_all=True, append_images=frames[1:], duration=400, loop=0, disposal=2, optimize=False)
    with Image.open(output / "comparison.gif") as animation:
        require(animation.n_frames == len(updates), "Animation lost frames")
        for i, update in enumerate(updates):
            animation.seek(i)
            require(animation.info["duration"] == 400, "Animation timing differs")
            decoded = np.asarray(animation.convert("RGB"))
            for column, directory in enumerate(directories):
                with Image.open(directory / "scoring_images" / f"update_{update:07d}.png") as source:
                    expected = np.asarray(source.resize((212, 212), Image.Resampling.NEAREST))[..., 0] > 127
                observed = decoded[328:540, 12 + column * 236:224 + column * 236, :].mean(axis=-1) > 127
                require(np.array_equal(expected, observed), "Decoded GIF occupancy is not synchronized")
    frames[-1].save(output / "endpoint.png")
    return rows


def provenance(args, output, starting_hashes):
    sources = ["main_opt.py", "asal_metrics.py", "scripts/coralai_temporal.py", "scripts/clip_scoring.py", "scripts/coralai_search.py", "scripts/evaluate_survival.py", "scripts/score_entropy.py", "substrates/coralai.py", "foundation_models/clip.py", "scripts/setup_novelty_env.sh", "tests/test_coralai_temporal.py", "schemas/coralai_temporal_search.schema.json", "configs/asal_novelty_pilot/search.json", "configs/asal_novelty_pilot/requirements.lock.txt", "configs/dynamic_energy_demo/fixed.json", "configs/dynamic_energy_demo/dynamic.json", "external/coralai/experiments/coral_dev/environments.py", "external/coralai/experiments/coral_dev/experiment.py", "external/coralai/experiments/coral_dev/headless_coral.py"]
    hashes = {}
    for relative in sources:
        source = ROOT / relative
        if source.is_file():
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative] = sha256(source)
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    environment = {key: os.environ.get(key) for key in ("JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TI_CPU_MAX_NUM_THREADS", "TI_OFFLINE_CACHE_FILE_PATH", "MPLCONFIGDIR", "HF_HUB_OFFLINE")}
    save_json(output / "execution.json", {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "command": shlex.join([sys.executable, *sys.argv]), "environment": environment, "python": sys.version, "versions": {p: version(p) for p in ("torch", "taichi", "neat-python", "numpy", "Pillow", "jax", "jaxlib", "evosax", "flax", "transformers", "matplotlib", "jsonschema")}, "source_sha256": hashes, "protected_starting_files_sha256": starting_hashes,
        "restore": "Checkout source_commit, apply working-tree.patch and overlay source/. Run setup_novelty_env.sh, then reproduce.sh with a fresh output directory. Cached references require a matching content/config/scoring/sampling manifest; --coralai_fresh_references explicitly re-evaluates both baselines. CPU scheduling can cause small floating-point variation."})
    (output / "requirements-used.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    argv = sys.argv[1:]
    argv = argv[:]
    argv[argv.index("--save_dir") + 1] = "__FRESH_OUTPUT__"
    command = shlex.join([str(ROOT / ".venv-novelty/bin/python"), "main_opt.py", *argv]).replace("__FRESH_OUTPUT__", '"${1:?Pass a fresh output directory}"')
    lines = ["#!/bin/sh", "set -eu", "cd " + shlex.quote(str(ROOT))]
    lines += ["export " + key + "=" + shlex.quote(value) for key, value in environment.items() if value is not None]
    lines.append(command)
    (output / "reproduce.sh").write_text("\n".join(lines) + "\n")


def run_search(args):
    import evosax
    import jax
    import jax.numpy as jnp

    require(args.coef_prompt == args.coef_softmax == 0 and args.coef_oe == 1 and args.bs == 1 and args.foundation_model == "clip", "Temporal mode requires coef_prompt=0, coef_softmax=0, coef_oe=1, bs=1, foundation_model=clip")
    require(args.pop_size >= 4 and 1 <= args.n_iters and args.sigma > 0, "Invalid candidate budget")
    config, settings = read_json(args.coralai_config), read_json(args.coralai_search_config)
    validate_setup(config, settings)
    require(args.time_sampling == len(settings["sampled_updates"]), "time_sampling must match sampled_updates")
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "scripts/coralai_search.py", ROOT / "configs/coralai_supervised.search.json")}
    output = Path(args.save_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "search_settings.json", settings)
    save_json(output / "base_config.json", config)
    started = time.perf_counter()
    evaluator = TemporalEvaluator(settings, output)
    # Rendering/capture frequency changes only; physics, seed and food stay fixed.
    sampled_config = copy.deepcopy(config)
    interval = math.gcd(*settings["sampled_updates"])
    sampled_config.update(screenshot_interval=interval, max_screenshots=config["steps"] // interval + 1)
    if args.coralai_timing_check:
        timing_updates = settings["sampled_updates"][:2]
        sampled_config.update(steps=timing_updates[-1], max_screenshots=timing_updates[-1] // interval + 1)
        score = evaluator.evaluate(sampled_config, output / "timing", updates=timing_updates)
        save_json(output / "timing.json", {"total_seconds": time.perf_counter() - started, "simulation_seconds": score["simulation_seconds"], "scoring_seconds_including_warmup": score["scoring_seconds"], "updates": timing_updates[-1], "frames": 2})
        provenance(args, output, protected)
        return
    fixed_config = copy.deepcopy(config)
    fixed_config["env_options"]["shift_interval"] = 0
    demo = ROOT / settings["reference_demo"]
    fixed = evaluator.evaluate(fixed_config, output / "fixed", reuse=reference_run(demo, "fixed", getattr(args, "coralai_fresh_references", False)))
    hand = evaluator.evaluate(config, output / "hand", reuse=reference_run(demo, "dynamic", getattr(args, "coralai_fresh_references", False)))
    evaluator.confound_control(Path(hand["run"]))
    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=len(settings["parameters"]), sigma_init=args.sigma)
    parameters = strategy.default_params.replace(clip_min=0., clip_max=1.)
    rng, init_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    state = strategy.initialize(init_rng, parameters)
    state = state.replace(mean=jnp.array([(config["env_options"][key] - b["min"]) / (b["max"] - b["min"]) for key, b in settings["parameters"].items()]))
    best, best_config, best_id, best_directory = hand, config, "hand", output / "hand"
    candidates, all_scores = [], [fixed, hand]
    budget = {"algorithm": "ASAL evosax.Sep_CMA_ES", "population": args.pop_size, "generations": args.n_iters, "candidate_budget": args.pop_size * args.n_iters, "optimizer_seed": args.seed, "sigma": args.sigma, "simulation_seed": settings["simulation_seed"], "selection": "Lowest original ASAL loss among hand and proposals; improvements <=1e-7 treated as ties. Fixed control excluded from selection.", "seed_scope": "One seed finite pilot; no multi-seed or held-out claim. #26 implementation unchanged."}
    save_json(output / "optimizer.json", budget)
    search_started = time.perf_counter()
    for generation in range(args.n_iters):
        rng, ask_rng = jax.random.split(rng)
        vectors, asked = strategy.ask(ask_rng, state, parameters)
        losses = []
        for index, vector in enumerate(np.asarray(vectors)):
            identity = f"g{generation:02d}_c{index:02d}"
            candidate = apply_parameters(sampled_config, vector, settings["parameters"])
            directory = output / "candidates" / identity
            score = evaluator.evaluate(candidate, directory)
            all_scores.append(score)
            losses.append(score["loss_oe"])
            candidates.append({"id": identity, **decode(vector, settings["parameters"]), "loss_oe": score["loss_oe"], "novelty": score["novelty"], "vector": vector.tolist(), "simulation_seconds": score["simulation_seconds"], "scoring_seconds": score["scoring_seconds"]})
            if score["loss_oe"] < best["loss_oe"] - 1e-7:
                best, best_config, best_id, best_directory = score, candidate, identity, directory
            save_json(output / "candidates.json", candidates)
        state = strategy.tell(vectors, jnp.asarray(losses), asked, parameters)
        np.savez(output / f"optimizer_state_{generation:02d}.npz", **{key: np.asarray(value) for key, value in state.__dict__.items()}, rng=np.asarray(rng))
    search_seconds = time.perf_counter() - search_started
    validation = compare_budgets(all_scores)
    rows = make_artifacts(output, [fixed_config, config, best_config], [fixed, hand, best], best_directory)
    save_json(output / "selected_config.json", best_config)
    save_json(output / "selection.json", {"selected_id": best_id, "selected_artifacts": str(best_directory), "loss_improvement_vs_hand": hand["loss_oe"] - best["loss_oe"], "comparison": rows})
    write_csv(output / "candidates.csv", [{k: v for k, v in row.items() if k != "vector"} for row in candidates])
    for path, digest in protected.items(): require(sha256(ROOT / path) == digest, "Protected #26 file changed")
    validation.update({"scored_state_rgb_matches": sum(len(s["measurements"]) for s in all_scores), "decoded_animation_occupancy_panels": 3 * len(best["measurements"]), "animation_frames": len(best["measurements"]), "frame_duration_ms": 400, "protected_files_unchanged": True})
    save_json(output / "validation.json", validation)
    save_json(output / "completed.json", {"candidate_rollouts": len(candidates), "reference_rollouts_reused": int(fixed["reused"]) + int(hand["reused"]), "scored_frames_per_rollout": len(settings["sampled_updates"]), "search_seconds": search_seconds, "total_seconds": time.perf_counter() - started, "selected_id": best_id})
    provenance(args, output, protected)
    print(f"Selected {best_id}; comparison: {output / 'comparison.csv'}", flush=True)
