"""Host evaluator for ASAL's Sep-CMA-ES and paired held-out Coralai runs.

Search is dispatched by main_opt.py; comparison is a separate CLI below.
Torch/Taichi rollouts run serially in child processes, outside JAX jit/vmap.
Each candidate is scored on every seed in settings["search_seeds"] and selected
on the mean loss across them (issue #26), instead of a single seed, so the
search can't pick a config that only looks good on one lucky seed. The three
comparison_seeds stay held out from the search entirely, for compare() below.
"""

import argparse
import csv
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "external/coralai/experiments/coral_dev/headless_coral.py"


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(vector, bounds):
    result = {}
    for x, (name, spec) in zip(vector, bounds.items()):
        value = spec["min"] + float(np.clip(x, 0, 1)) * (spec["max"] - spec["min"])
        result[name] = int(round(value)) if spec["integer"] else value
    return result


def frame_summary(steps, losses):
    """Minimize mean negative cosine; keep best and final frame distinct."""
    if not steps or len(steps) != len(losses) or not np.isfinite(losses).all():
        raise ValueError("Expected one finite score per sampled frame")
    best = int(np.argmin(losses))
    return {"mean_loss": float(np.mean(losses)),
            "mean_alignment": -float(np.mean(losses)),
            "best_sample_loss": float(losses[best]),
            "best_sample_alignment": -float(losses[best]),
            "best_sample_step": steps[best],
            "final_loss": float(losses[-1]), "final_alignment": -float(losses[-1])}


def paired_summary(rows):
    differences = np.array([row["difference"] for row in rows])
    result = {"n_seeds": len(rows), "positive_means": "selected improves alignment",
              "improved_seeds": int((differences > 0).sum()),
              "paired_difference_mean": float(differences.mean()),
              "paired_difference_sample_std": float(differences.std(ddof=1))}
    for label in ("default", "selected"):
        values = [row[label]["mean_alignment"] for row in rows]
        result[label + "_mean_alignment"] = float(np.mean(values))
        result[label + "_sample_std"] = float(np.std(values, ddof=1))
    return result


def average_across_seeds(scores):
    """Summarize one config's per-seed evaluate() scores as a mean +/- spread.

    Selecting on this mean (instead of a single seed's score) is the fix for
    issue #26: a config that only looks good on one lucky seed will show up
    here with a high alignment_std relative to its peers.
    """
    if len(scores) < 2:
        raise ValueError("Expected at least two seeds to average across")
    losses = np.array([score["mean_loss"] for score in scores])
    alignments = np.array([score["mean_alignment"] for score in scores])
    best_samples = np.array([score["best_sample_alignment"] for score in scores])
    finals = np.array([score["final_alignment"] for score in scores])
    return {"seeds": [score["seed"] for score in scores], "n_seeds": len(scores),
            "mean_loss": float(losses.mean()), "loss_std": float(losses.std(ddof=1)),
            "mean_alignment": float(alignments.mean()), "alignment_std": float(alignments.std(ddof=1)),
            "best_sample_alignment": float(best_samples.mean()), "final_alignment": float(finals.mean()),
            "simulation_seconds": float(sum(score["simulation_seconds"] for score in scores)),
            "scoring_seconds": float(sum(score["scoring_seconds"] for score in scores)),
            "per_seed": scores}


def evaluate_across_seeds(evaluator, config, seeds, directory):
    """Run one config on every search seed and save/return the averaged summary."""
    scores = [evaluator.evaluate(config, seed, directory / f"seed_{seed}") for seed in seeds]
    summary = average_across_seeds(scores)
    save_json(directory / "multi_seed_summary.json", summary)
    return summary


def validate_setup(config, settings):
    if config["backend"] != "cpu" or config["device"] != "cpu" or config.get("resume_from"):
        raise ValueError("This experiment requires fresh CPU simulations")
    steps, interval = config["steps"], config["screenshot_interval"]
    if steps < 1 or interval < 1 or steps % interval:
        raise ValueError("Steps must be positive and divisible by screenshot_interval")
    if config["max_screenshots"] < steps // interval + 1:
        raise ValueError("max_screenshots must include step 0 and every evaluation frame")
    search_seeds, comparison_seeds = settings["search_seeds"], settings["comparison_seeds"]
    if len(search_seeds) < 2 or len(set(search_seeds)) != len(search_seeds):
        raise ValueError("Use at least two distinct search seeds so scores can be averaged")
    if len(comparison_seeds) != 3 or len(set(comparison_seeds) | set(search_seeds)) != 3 + len(search_seeds):
        raise ValueError("Use three distinct comparison seeds separate from the search seeds")
    if not settings["prompt"].strip() or ";" in settings["prompt"]:
        raise ValueError("This adapter supports one nonempty target prompt")
    for name, spec in settings["parameters"].items():
        if name not in ("radiate_interval", "infra_decay", "defense_coeff"):
            raise ValueError(f"Unsupported searched parameter: {name}")
        if not spec["min"] <= config[name] <= spec["max"] or spec["min"] >= spec["max"]:
            raise ValueError(f"Default must fall within valid bounds: {name}")
        if spec["min"] < 0 or (name == "radiate_interval" and spec["min"] < 1):
            raise ValueError(f"Invalid bounds for {name}")


def provenance(output):
    sources = [Path("main_opt.py"), Path("scripts/coralai_search.py"),
               Path("scripts/clip_scoring.py"), Path("foundation_models/clip.py"),
               Path("asal_metrics.py"), RUNNER.relative_to(ROOT),
               Path("external/coralai/experiments/coral_dev/experiment.py"),
               Path("external/coralai/experiments/coral_dev/evolution.py"),
               Path("external/coralai/coralai/evolver.py")]
    record = {"command": shlex.join([sys.executable, *sys.argv]), "cwd": str(Path.cwd()),
              "python": sys.executable, "platform": platform.platform(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "versions": {name: version(name) for name in
                           ("jax", "jaxlib", "evosax", "flax", "transformers", "numpy", "pillow", "torch", "taichi")},
              "source_sha256": {str(path): sha256(ROOT / path) for path in sources}}
    save_json(output / "execution.json", record)
    (output / "requirements-used.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))


class Evaluator:
    def __init__(self, settings, output):
        import jax
        from huggingface_hub import snapshot_download
        from scripts.clip_scoring import ClipScorer

        model_path = snapshot_download(settings["model"], revision=settings["revision"], local_files_only=True)
        started = time.perf_counter()
        self.scorer = ClipScorer("cpu", model_path=model_path)
        self.text = self.scorer.text_embeddings([settings["prompt"]])
        self.settings = settings
        save_json(output / "scorer.json", {
            "model": settings["model"], "revision": settings["revision"],
            "weights_sha256": sha256(Path(model_path) / "flax_model.msgpack"),
            "model_load_seconds": self.scorer.model_load_seconds,
            "load_and_prompt_seconds": time.perf_counter() - started,
            "device": str(self.scorer.device), "preprocessing": "RGB, 224x224 LANCZOS, float32 /255; existing CLIP normalization",
            "objective": "mean ASAL negative cosine over steps interval..steps; step 0 excluded",
            "timing": "simulation includes child startup and disk; scoring includes image loading and synchronization; first evaluation includes JIT warm-up",
            "jax_devices": [str(device) for device in jax.devices()]})

    def evaluate(self, config, seed, directory):
        import jax
        import torch
        from scripts.clip_scoring import load_clip_image

        directory.mkdir(parents=True, exist_ok=False)
        effective = dict(config, seed=seed, output_dir=None, resume_from=None)
        save_json(directory / "input_config.json", effective)
        run = directory / "run"
        command = [sys.executable, str(RUNNER), "--config", str(directory / "input_config.json"),
                   "--seed", str(seed), "--output-dir", str(run)]
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT / "external/coralai"),
            str(ROOT / "external/coralai/coralai/dependencies/PyTorch-NEAT"),
            env.get("PYTHONPATH", "")])
        save_json(directory / "command.json", {"argv": command, "cwd": str(ROOT),
                                               "PYTHONPATH": env["PYTHONPATH"]})
        started = time.perf_counter()
        with (directory / "simulation.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        simulation_seconds = time.perf_counter() - started
        metadata = read_json(run / "meta.json")
        if metadata["completed_steps"] != config["steps"] or metadata["seed"] != seed:
            raise ValueError(f"Incomplete or incorrect run: {run}")
        state = torch.load(run / metadata["final_checkpoint"] / "substrate.pt", map_location="cpu", weights_only=True)
        if not torch.isfinite(state).all():
            raise ValueError(f"Nonfinite final checkpoint: {run}")
        with (run / "step_log.csv").open() as stream:
            if any(not np.isfinite(float(value)) for row in csv.DictReader(stream) for value in row.values()):
                raise ValueError(f"Nonfinite simulation log: {run}")
        steps = list(range(config["screenshot_interval"], config["steps"] + 1, config["screenshot_interval"]))
        frames = [run / "screenshots" / f"step_{step:07d}.png" for step in steps]
        started = time.perf_counter()
        losses = np.asarray(jax.device_get(self.scorer.losses(
            [load_clip_image(path, resize=True) for path in frames], self.text)))[:, 0]
        scoring_seconds = time.perf_counter() - started
        summary = frame_summary(steps, losses)
        summary.update(seed=seed, simulation_seconds=simulation_seconds, scoring_seconds=scoring_seconds,
                       final_checkpoint=str(run / metadata["final_checkpoint"]),
                       frame_scores=[{"step": step, "image": str(path), "sha256": sha256(path),
                                      "loss": float(loss), "alignment": -float(loss)}
                                     for step, path, loss in zip(steps, frames, losses)])
        for label, step in (("best_sample", summary["best_sample_step"]), ("final", config["steps"])):
            shutil.copyfile(run / "screenshots" / f"step_{step:07d}.png", directory / f"{label}.png")
        save_json(directory / "scores.json", summary)
        print(f"{directory.name}: mean={summary['mean_alignment']:.6f}, "
              f"best={summary['best_sample_alignment']:.6f}, final={summary['final_alignment']:.6f}; "
              f"simulation {simulation_seconds:.2f}s, scoring {scoring_seconds:.2f}s", flush=True)
        return summary


def run_search(args):
    import evosax
    import jax
    import jax.numpy as jnp

    if not args.coralai_config or not args.coralai_search_config or not args.save_dir:
        raise ValueError("Coralai requires --coralai_config, --coralai_search_config and --save_dir")
    if args.bs != 1 or args.coef_prompt != 1 or args.coef_softmax != 0 or args.coef_oe != 0:
        raise ValueError("Coralai adapter supports a single rollout batch and the supervised target objective only")
    if args.pop_size < 4 or args.n_iters < 1 or args.sigma <= 0:
        raise ValueError("Use pop_size >= 4, n_iters >= 1 and sigma > 0")
    config, settings = read_json(args.coralai_config), read_json(args.coralai_search_config)
    validate_setup(config, settings)
    output = Path(args.save_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save_json(output / "default_config.json", dict(config, output_dir=None, resume_from=None))
    save_json(output / "search_settings.json", settings)
    seeds = settings["search_seeds"]
    save_json(output / "optimizer.json", {"driver": "main_opt.py", "algorithm": "evosax.Sep_CMA_ES",
              "optimizer_seed": args.seed, "population": args.pop_size, "generations": args.n_iters,
              "sigma": args.sigma, "search_seeds": seeds, "evaluations_per_candidate": len(seeds),
              "candidate_budget": args.pop_size * args.n_iters * len(seeds), "normalized_bounds": [0, 1],
              "selection": "lowest mean loss averaged across search_seeds among default and candidates; default wins exact ties"})
    provenance(output)
    evaluator = Evaluator(settings, output)
    baseline = evaluate_across_seeds(evaluator, config, seeds, output / "default")
    best, best_config, best_id = baseline, config, "default"
    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=len(settings["parameters"]), sigma_init=args.sigma)
    params = strategy.default_params.replace(clip_min=0., clip_max=1.)
    rng, init_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    state = strategy.initialize(init_rng, params)
    center = [(config[name] - spec["min"]) / (spec["max"] - spec["min"])
              for name, spec in settings["parameters"].items()]
    state = state.replace(mean=jnp.array(center))
    candidates, generations = [], []
    for generation in range(args.n_iters):
        rng, ask_rng = jax.random.split(rng)
        vectors, asked_state = strategy.ask(ask_rng, state, params)
        losses = []
        for index, vector in enumerate(np.asarray(vectors)):
            name = f"g{generation:02d}_c{index:02d}"
            candidate_config = dict(config, **decode(vector, settings["parameters"]))
            score = evaluate_across_seeds(evaluator, candidate_config, seeds, output / "candidates" / name)
            losses.append(score["mean_loss"])
            candidates.append({"id": name, "generation": generation, "vector": vector.tolist(),
                               "parameters": decode(vector, settings["parameters"]), "scores": score})
            if score["mean_loss"] < best["mean_loss"]:
                best, best_config, best_id = score, candidate_config, name
            save_json(output / "candidates.json", candidates)
            save_json(output / "selected_config.json", dict(best_config, output_dir=None, resume_from=None))
            save_json(output / "selection.json", {"selected_id": best_id, "selected_scores": best,
                       "default_scores": baseline, "alignment_difference": best["mean_alignment"] - baseline["mean_alignment"]})
        state = strategy.tell(vectors, jnp.asarray(losses), asked_state, params)
        generations.append({"generation": generation, "mean": np.asarray(state.mean).tolist(),
                            "sigma_scale": float(state.sigma_scale), "losses": losses})
        np.savez(output / f"optimizer_state_{generation:02d}.npz", **{
            key: np.asarray(value) for key, value in state.__dict__.items()}, rng=np.asarray(rng))
        save_json(output / "generations.json", generations)
    with (output / "candidates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", *settings["parameters"], "mean_alignment", "alignment_std", "best_sample_alignment",
                         "final_alignment", "simulation_seconds", "scoring_seconds"])
        for candidate in [{"id": "default", "parameters": {k: config[k] for k in settings["parameters"]}, "scores": baseline}, *candidates]:
            score = candidate["scores"]
            writer.writerow([candidate["id"], *candidate["parameters"].values(), *[score[k] for k in
                            ("mean_alignment", "alignment_std", "best_sample_alignment", "final_alignment",
                             "simulation_seconds", "scoring_seconds")]])
    save_json(output / "completed.json", {"evaluations": (len(candidates) + 1) * len(seeds),
              "selected_id": best_id, "total_seconds": time.perf_counter() - started})
    print(f"Selected {best_id}; config: {output / 'selected_config.json'}", flush=True)


def compare(experiment, output):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    default, selected = read_json(experiment / "default_config.json"), read_json(experiment / "selected_config.json")
    settings = read_json(experiment / "search_settings.json")
    validate_setup(default, settings)
    # The comparison can change only the parameters declared in the search.
    if {k: v for k, v in default.items() if k not in settings["parameters"]} != {
            k: v for k, v in selected.items() if k not in settings["parameters"]}:
        raise ValueError("Selected config changed settings outside the search bounds")
    if not (experiment / "completed.json").is_file():
        raise ValueError("Search must finish before held-out comparison")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance(output)
    for name in ("default_config.json", "selected_config.json", "search_settings.json"):
        shutil.copyfile(experiment / name, output / name)
    evaluator = Evaluator(settings, output)
    rows = []
    for index, seed in enumerate(settings["comparison_seeds"]):
        scores = {}
        # Alternate run order to reduce a systematic warm-up/order effect.
        for label in (("default", "selected") if index % 2 == 0 else ("selected", "default")):
            scores[label] = evaluator.evaluate(default if label == "default" else selected, seed, output / f"seed_{seed}" / label)
        rows.append(dict(seed=seed, **scores, difference=scores["selected"]["mean_alignment"] - scores["default"]["mean_alignment"]))
        save_json(output / "per_seed.json", rows)
    result = paired_summary(rows)
    result["total_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", result)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "default_mean_alignment", "selected_mean_alignment", "paired_difference",
                         "default_best_sample_alignment", "selected_best_sample_alignment", "default_final_alignment", "selected_final_alignment"])
        for row in rows:
            writer.writerow([row["seed"], row["default"]["mean_alignment"], row["selected"]["mean_alignment"], row["difference"],
                             *[row[label][metric] for metric in ("best_sample_alignment", "final_alignment") for label in ("default", "selected")]])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--experiment", required=True, type=Path)
    comparison.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args()
    compare(cli.experiment, cli.output_dir)
