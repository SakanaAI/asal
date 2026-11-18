"""Issue #37: the finite temporal-novelty search, made seed-robust.

The pilot (scripts/coralai_temporal.py) scored every candidate world on ONE
simulation seed and reported "selected equals hand-made". This driver asks the
question properly, using the same recipe as issues #26/#31:

  1. score each candidate on several search seeds and select on
     mean loss + spread_penalty * std (steady worlds beat lucky ones);
  2. evaluate the hand-made world the same way, so the comparison is fair;
  3. re-run hand-made vs selected on held-out seeds the search never saw.

scripts/coralai_temporal.py is deliberately NOT modified: its cached reference
runs are keyed to a hash of that file, and the single-seed pilot stays
reproducible. Everything new that is not simulator-bound lives in
scripts/temporal_robustness.py (pure numpy, unit-tested).
"""

import copy
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.coralai_search import ROOT, decode, read_json, save_json, sha256
from scripts.coralai_temporal import (TemporalEvaluator, apply_parameters, compare_budgets, make_artifacts,
                                      provenance, reference_run, require, validate_setup, write_csv)
from scripts.temporal_robustness import (TIE, aggregate_losses, change_summary, paired_heldout,
                                         validate_seed_plan)

MULTISEED_KEYS = ("search_seeds", "spread_penalty", "comparison_seeds", "heldout_after_search")
OBJECTIVE = "temporal_novelty_multiseed"


def split_settings(settings):
    """Return (settings the single-seed pipeline understands, the multi-seed plan)."""
    plan = {key: settings[key] for key in MULTISEED_KEYS if key in settings}
    require("search_seeds" in plan and "comparison_seeds" in plan, "Multi-seed settings need search_seeds and comparison_seeds")
    base = {key: value for key, value in settings.items() if key not in MULTISEED_KEYS}
    plan.setdefault("spread_penalty", 0.0)
    plan.setdefault("heldout_after_search", True)
    validate_seed_plan(settings["simulation_seed"], plan["search_seeds"], plan["comparison_seeds"], plan["spread_penalty"])
    return base, plan


def with_seed(config, seed):
    result = copy.deepcopy(config)
    result["seed"] = seed
    return result


def occupancy_frames(run, updates):
    """Boolean occupied-cell grids at update 0 and every scored update."""
    return [(torch.load(Path(run) / "frame_states" / f"update_{u:07d}.pt", weights_only=True)[0, 13] >= 0).numpy()
            for u in [0, *updates]]


def evaluate_world(evaluator, config, seeds, directory, spread_penalty, *, reuse=None, later_config=None):
    """Run one world on every seed; return the first seed's full score plus the multi-seed summary.

    The first seed is written straight into `directory` (so the existing figure code finds its
    images there); the other seeds go in directory/seed_<n>. `reuse` may point at a verified cached
    run for the first seed only. `later_config` lets the other seeds use the lighter capture settings.
    """
    scores = []
    for index, seed in enumerate(seeds):
        base = config if index == 0 or later_config is None else later_config
        where = directory if index == 0 else directory / f"seed_{seed}"
        scores.append(evaluator.evaluate(with_seed(base, seed), where, reuse=reuse if index == 0 else None))
    summary = aggregate_losses([s["loss_oe"] for s in scores], spread_penalty)
    summary.update(seeds=list(seeds), per_seed_novelty=[s["novelty"] for s in scores],
                   simulation_seconds=float(sum(s["simulation_seconds"] for s in scores)),
                   scoring_seconds=float(sum(s["scoring_seconds"] for s in scores)))
    save_json(directory / "multi_seed_summary.json", summary)
    primary = dict(scores[0])
    primary.update({key: summary[key] for key in ("mean_loss", "loss_std", "selection_loss", "per_seed_loss", "seeds")})
    return primary


def heldout_check(evaluator, hand_config, selected_config, seeds, output, updates):
    """Hand-made vs selected on seeds the search never saw (paired by seed)."""
    rows = []
    for index, seed in enumerate(seeds):
        scores, changes = {}, {}
        for label in (("hand", "selected") if index % 2 == 0 else ("selected", "hand")):  # alternate order
            config = hand_config if label == "hand" else selected_config
            scores[label] = evaluator.evaluate(with_seed(config, seed), output / f"seed_{seed}" / label)
            changes[label] = change_summary(occupancy_frames(scores[label]["run"], updates))
        rows.append({"seed": seed, "hand_loss": scores["hand"]["loss_oe"], "selected_loss": scores["selected"]["loss_oe"],
                     "hand_change": changes["hand"], "selected_change": changes["selected"]})
        save_json(output / "per_seed.json", rows)
    result = paired_heldout([r["hand_loss"] for r in rows], [r["selected_loss"] for r in rows])
    for label in ("hand", "selected"):
        for key in ("changed_from_start", "ongoing_change", "late_change"):
            result[f"{label}_{key}_mean"] = float(np.mean([r[f"{label}_change"][key] for r in rows]))
    result["seeds"] = list(seeds)
    save_json(output / "summary.json", result)
    write_csv(output / "comparison.csv", [{
        "seed": r["seed"], "hand_loss": r["hand_loss"], "selected_loss": r["selected_loss"],
        "paired_difference_hand_minus_selected": r["hand_loss"] - r["selected_loss"],
        "hand_late_change": r["hand_change"]["late_change"], "selected_late_change": r["selected_change"]["late_change"],
        "hand_changed_from_start": r["hand_change"]["changed_from_start"],
        "selected_changed_from_start": r["selected_change"]["changed_from_start"]} for r in rows])
    return result


def run_search(args):
    import evosax
    import jax
    import jax.numpy as jnp

    require(args.coef_prompt == args.coef_softmax == 0 and args.coef_oe == 1 and args.bs == 1 and args.foundation_model == "clip",
            "Temporal mode requires coef_prompt=0, coef_softmax=0, coef_oe=1, bs=1, foundation_model=clip")
    require(args.pop_size >= 4 and 1 <= args.n_iters and args.sigma > 0, "Invalid candidate budget")
    config, full_settings = read_json(args.coralai_config), read_json(args.coralai_search_config)
    settings, plan = split_settings(full_settings)
    validate_setup(config, settings)
    require(args.time_sampling == len(settings["sampled_updates"]), "time_sampling must match sampled_updates")
    seeds, penalty = plan["search_seeds"], plan["spread_penalty"]
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "scripts/coralai_search.py", ROOT / "scripts/coralai_temporal.py",
                                                               ROOT / "configs/coralai_supervised.search.json")}
    output = Path(args.save_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "search_settings.json", full_settings)
    save_json(output / "base_config.json", config)
    started = time.perf_counter()
    evaluator = TemporalEvaluator(settings, output)
    sampled_config = copy.deepcopy(config)
    interval = math.gcd(*settings["sampled_updates"])
    sampled_config.update(screenshot_interval=interval, max_screenshots=config["steps"] // interval + 1)
    fresh = getattr(args, "coralai_fresh_references", False)
    demo = ROOT / settings["reference_demo"]

    # Fixed-food control stays single-seed: it is context for the story, never selected.
    fixed_config = copy.deepcopy(config)
    fixed_config["env_options"]["shift_interval"] = 0
    fixed = evaluator.evaluate(fixed_config, output / "fixed", reuse=reference_run(demo, "fixed", fresh))
    hand = evaluate_world(evaluator, config, seeds, output / "hand", penalty,
                          reuse=reference_run(demo, "dynamic", fresh), later_config=sampled_config)
    evaluator.confound_control(Path(hand["run"]))

    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=len(settings["parameters"]), sigma_init=args.sigma)
    parameters = strategy.default_params.replace(clip_min=0., clip_max=1.)
    rng, init_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    state = strategy.initialize(init_rng, parameters)
    state = state.replace(mean=jnp.array([(config["env_options"][key] - b["min"]) / (b["max"] - b["min"])
                                          for key, b in settings["parameters"].items()]))
    best, best_config, best_id, best_directory = hand, config, "hand", output / "hand"
    candidates, all_scores = [], [fixed, hand]
    save_json(output / "optimizer.json", {
        "algorithm": "ASAL evosax.Sep_CMA_ES", "population": args.pop_size, "generations": args.n_iters,
        "candidate_budget": args.pop_size * args.n_iters, "rollouts_per_candidate": len(seeds),
        "optimizer_seed": args.seed, "sigma": args.sigma, "search_seeds": seeds, "spread_penalty": penalty,
        "comparison_seeds": plan["comparison_seeds"],
        "selection": "Lowest selection_loss = mean ASAL loss over search seeds + spread_penalty * sample std, among hand and "
                     "proposals; improvements <= 1e-7 are ties. Fixed control excluded. ASAL objective itself unchanged.",
        "seed_scope": "Issue #37: multi-seed search plus held-out comparison on seeds never used in the search."})
    search_started = time.perf_counter()
    for generation in range(args.n_iters):
        rng, ask_rng = jax.random.split(rng)
        vectors, asked = strategy.ask(ask_rng, state, parameters)
        values = []
        for index, vector in enumerate(np.asarray(vectors)):
            identity = f"g{generation:02d}_c{index:02d}"
            candidate = apply_parameters(sampled_config, vector, settings["parameters"])
            directory = output / "candidates" / identity
            score = evaluate_world(evaluator, candidate, seeds, directory, penalty)
            all_scores.append(score)
            values.append(score["selection_loss"])
            candidates.append({"id": identity, **decode(vector, settings["parameters"]),
                               "selection_loss": score["selection_loss"], "mean_loss": score["mean_loss"],
                               "loss_std": score["loss_std"], "primary_seed_loss": score["loss_oe"],
                               "per_seed_loss": score["per_seed_loss"], "vector": vector.tolist()})
            if score["selection_loss"] < best["selection_loss"] - TIE:
                best, best_config, best_id, best_directory = score, candidate, identity, directory
            save_json(output / "candidates.json", candidates)
        state = strategy.tell(vectors, jnp.asarray(values), asked, parameters)
        np.savez(output / f"optimizer_state_{generation:02d}.npz", **{k: np.asarray(v) for k, v in state.__dict__.items()}, rng=np.asarray(rng))
    search_seconds = time.perf_counter() - search_started

    # Figures and the budget check use the primary seed's runs, which every world has.
    validation = compare_budgets(all_scores)
    rows = make_artifacts(output, [fixed_config, config, best_config], [fixed, hand, best], best_directory)
    save_json(output / "selected_config.json", best_config)
    save_json(output / "selection.json", {
        "selected_id": best_id, "selected_artifacts": str(best_directory),
        "search_seeds": seeds, "spread_penalty": penalty,
        "hand": {k: hand[k] for k in ("mean_loss", "loss_std", "selection_loss", "per_seed_loss")},
        "selected": {k: best[k] for k in ("mean_loss", "loss_std", "selection_loss", "per_seed_loss")},
        "selection_loss_improvement_vs_hand": hand["selection_loss"] - best["selection_loss"],
        "primary_seed_comparison": rows})
    write_csv(output / "candidates.csv", [{k: v for k, v in row.items() if k != "vector"} for row in candidates])
    for path, digest in protected.items():
        require(sha256(ROOT / path) == digest, f"Protected file changed during the run: {path}")

    heldout = None
    if best_id == "hand":
        save_json(output / "heldout_skipped.json", {"reason": "Search found nothing better than the hand-made world on the "
                  "search seeds, so selected == hand and a held-out comparison would compare a world with itself."})
    elif plan["heldout_after_search"]:
        # sampled_config is the hand-made world with candidate-style capture settings, so only the
        # food-shift parameters differ between the two worlds being compared.
        hand_config = sampled_config
        heldout = heldout_check(evaluator, hand_config, best_config, plan["comparison_seeds"], output / "heldout",
                                settings["sampled_updates"])
    save_json(output / "validation.json", {**validation, "scope": "primary-seed runs only (every world shares seed %d)" % settings["simulation_seed"],
                                           "protected_files_unchanged": True})
    save_json(output / "completed.json", {
        "candidate_worlds": len(candidates), "rollouts_per_world": len(seeds), "search_seconds": search_seconds,
        "total_seconds": time.perf_counter() - started, "selected_id": best_id,
        "heldout_ran": heldout is not None, "heldout_held_up": None if heldout is None else heldout["held_up"]})
    provenance(args, output, protected)
    for name in ("scripts/coralai_temporal_multiseed.py", "scripts/temporal_robustness.py"):
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    print(f"Selected {best_id}; held-out check: {'skipped' if heldout is None else ('held up' if heldout['held_up'] else 'did not hold up')}", flush=True)
