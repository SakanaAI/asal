"""Verify corrected controllers and compare the established trail treatments."""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external/coralai"), str(ROOT / "external/coralai/coralai/dependencies/PyTorch-NEAT")]
from scripts.compare_trail_reference import (analyze, audit_defaults, frame_image, load_state,
                                             read_json, require, run, save_json, sha256, write_csv)


def configurations():
    configs = {n: read_json(ROOT / f"configs/trail_reference_comparison/{n}.json") for n in ("fixed", "dynamic")}
    for c in configs.values(): c.update(log_cell_events=True, save_initial_checkpoint=True)
    equal = copy.deepcopy(configs["fixed"])
    equal["env_options"]["shift_interval"] = configs["dynamic"]["env_options"]["shift_interval"]
    require(equal == configs["dynamic"], "Only source movement may differ")
    return configs


def read_rows(path):
    with path.open() as f: return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]


def controller_equality(output):
    populations = {}
    for n in ("fixed", "dynamic"):
        with (output / n / "run/checkpoint_0000000/population.pkl").open("rb") as f: populations[n] = pickle.load(f)
    a, b = populations.values()
    require(a["genome_keys"] == b["genome_keys"] == list(range(50)), "Initial registries differ")
    for name in ("weights", "biases"):
        require(len(a[name]) == len(b[name]), "Initial controller counts differ")
        require(all(torch.equal(x, y) for x, y in zip(a[name], b[name])), "Initial controller parameters differ")
    require(pickle.dumps(a["genomes"]) == pickle.dumps(b["genomes"]), "Initial genomes differ")
    require(torch.equal(a["rng_state"]["torch"], b["rng_state"]["torch"]), "Initial Torch RNG states differ")
    an, bn = a["rng_state"]["numpy"], b["rng_state"]["numpy"]
    require(an[0] == bn[0] and np.array_equal(an[1], bn[1]) and an[2:] == bn[2:], "Initial NumPy RNG states differ")
    require(a["rng_state"]["python"] == b["rng_state"]["python"], "Initial Python RNG states differ")
    return {"initial_controller_parameters_equal": True, "initial_genomes_equal": True,
            "initial_registries_equal": True, "initial_rng_states_equal": True,
            "registered_controller_count": len(a["genomes"])}


def event_analysis(output, configs):
    rows, totals, cumulative = {}, {}, {}
    for n, c in configs.items():
        rows[n] = read_rows(output / n / "run/cell_events.csv")
        require([r["completed_update"] for r in rows[n]] == list(range(1, c["steps"] + 1)), "Incomplete event log")
        with (output / n / "run/resource_metrics.csv").open() as f:
            counts = {int(r["completed_update"]): int(r["occupied_cells"]) for r in csv.DictReader(f)}
        cumulative[n] = {0: {"local": 0, "forced": 0}}
        for r in rows[n]:
            u = r["completed_update"]
            require(r["occupied_before"] == counts[u-1] and r["occupied_after"] == counts[u], "Event/count phase mismatch")
            gains = sum(r[p + "_gains"] for p in ("colonization", "resource_death", "chunk_death", "radiation", "cleanup", "cull"))
            losses = sum(r[p + "_losses"] for p in ("colonization", "resource_death", "chunk_death", "radiation", "cleanup", "cull"))
            require(gains - losses == counts[u] - counts[u-1], "Event accounting mismatch")
            require(r["resource_death_gains"] == r["chunk_death_gains"] == r["cleanup_gains"] == 0, "Unexpected phase gains")
            if not (u > 1 and (u-1) % c["radiate_interval"] == 0):
                require(r["radiation_gains"] == r["radiation_losses"] == 0, "Radiation timing mismatch")
            last = cumulative[n][u-1]
            cumulative[n][u] = {"local": last["local"] + r["colonization_gains"], "forced": last["forced"] + r["radiation_gains"]}
        totals[n] = {k: sum(r[k] for r in rows[n]) for k in rows[n][0] if k.endswith(("_gains", "_losses", "_replacements"))}
        totals[n].update(initial_occupied=counts[0], final_occupied=counts[c["steps"]],
                         first_local_colonization_update=next((r["completed_update"] for r in rows[n] if r["colonization_gains"]), None))
    save_json(output / "event_summary.json", totals)
    return totals, cumulative


def derived_outputs(output, configs):
    initial = controller_equality(output)
    totals, cumulative = event_analysis(output, configs)
    def render(update, states, resources, metrics):
        original = frame_image(update, states, resources, metrics)
        canvas = Image.new("RGB", (original.width, original.height + 57), (17, 22, 31))
        canvas.paste(original, (0, 0))
        draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=13)
        draw.rectangle((0, 0, canvas.width, 51), fill=(17, 22, 31))
        draw.text((12, 7), f"Corrected controllers | 400 x 400 | seed 42 | completed update {update:04d}", fill="white", font=font)
        draw.text((12, 29), "Reference trail settings preserved; GUI E/I/rotation normalized independently", fill="#c6cbd2", font=font)
        for column, n in enumerate(("fixed", "dynamic")):
            x = 12 + column * 348; c = cumulative[n][update]
            draw.text((x, 1187), f"Cumulative new local occupancy: {c['local']}", fill="#54d9ed", font=font)
            draw.text((x, 1208), f"Cumulative radiation gains: {c['forced']}", fill="#c6cbd2", font=font)
        return canvas
    analyze(output, configs, frame_renderer=render, title="Corrected controllers | 400 x 400 | seed 42")
    validation = read_json(output / "validation.json")
    validation.update(initial)
    validation["event_update_checks"] = sum(c["steps"] for c in configs.values())
    save_json(output / "validation.json", validation)
    with (output / "results.csv").open() as f: results = list(csv.DictReader(f))
    for row in results:
        t = totals[row["configuration"]]
        for name in ("colonization_gains", "colonization_replacements", "resource_death_losses", "chunk_death_losses", "radiation_gains", "radiation_losses", "net_gains", "net_losses"):
            row[name] = t[name]
    write_csv(output / "results.csv", results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for n in configs:
        updates = list(cumulative[n])
        ax.plot(updates, [cumulative[n][u]["local"] for u in updates], label=f"{n}: local colonization")
        ax.plot(updates, [cumulative[n][u]["forced"] for u in updates], linestyle="--", label=f"{n}: radiation gains")
    ax.set(xlabel="Completed update", ylabel="Cumulative empty-to-occupied transitions", title="Observed mechanisms, measured at phase boundaries")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / "mechanisms.png", dpi=140); plt.close(fig)
    print(json.dumps({"results": results, "validation": validation}, indent=2), flush=True)


def provenance(output):
    files = ["scripts/demo_controller_lookup_fix.py", "scripts/verify_controller_lookup.py", "tests/test_controller_lookup.py", "tests/test_controller_integration.py",
             "scripts/compare_trail_reference.py", "scripts/evaluate_survival.py", "scripts/setup_novelty_env.sh",
             "configs/asal_novelty_pilot/requirements.lock.txt", "configs/trail_reference_comparison/fixed.json", "configs/trail_reference_comparison/dynamic.json",
             "external/coralai/coralai/evolver.py", "external/coralai/coralai/experiment.py", "external/coralai/coralai/substrate.py", "external/coralai/coralai/substrate_index.py",
             "external/coralai/experiments/coral_dev/headless_coral.py", "external/coralai/experiments/coral_dev/experiment.py",
             "external/coralai/experiments/coral_dev/cell_events.py", "external/coralai/experiments/coral_dev/environments.py",
             "external/coralai/experiments/coral_dev/physics.py", "external/coralai/experiments/coral_dev/evolution.py", "external/coralai/experiments/coral_dev/neat.config"]
    hashes = {}
    for relative in files:
        target = output / "source" / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target); hashes[relative] = sha256(target)
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    save_json(output / "execution.json", {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "source_sha256": hashes,
        "required_environment": {"OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "TI_CPU_MAX_NUM_THREADS":"1"},
        "child_commands": {n: read_json(output / n / "command.json") for n in ("timing", "fixed", "dynamic")},
        "restore": "Checkout source_commit, apply working-tree.patch, overlay source/, set up .venv-novelty, run reproduce.sh into a fresh directory.",
        "limits": "Original executable trail environment defaults, not verified teammate screenshot settings; headless post-operation capture phase; CPU numerical variation remains possible."})
    (output / "requirements-used.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    lines = ["#!/bin/sh", "set -eu", "cd " + shlex.quote(str(ROOT)), "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TI_CPU_MAX_NUM_THREADS=1",
             "export MPLCONFIGDIR=" + shlex.quote(str(ROOT / ".cache/matplotlib")),
             "export TI_OFFLINE_CACHE_FILE_PATH=" + shlex.quote(str(ROOT / ".cache/taichi-controller")),
             shlex.join([str(ROOT / ".venv-novelty/bin/python"), "-B", "scripts/demo_controller_lookup_fix.py", "--output-dir"]) + ' "${1:?Pass a fresh output directory}"']
    (output / "reproduce.sh").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage", choices=("all", "pair", "analyze"), default="all")
    args = p.parse_args(); output = args.output_dir.resolve()
    if args.stage == "all": output.mkdir(parents=True, exist_ok=False)
    configs = configurations()
    audit_defaults(configs, output)
    if args.stage == "all":
        c = copy.deepcopy(configs["fixed"]); c.update(steps=50, max_screenshots=3, checkpoint_interval=50)
        run(c, output / "timing")
        elapsed = read_json(output / "timing/run/meta.json")["elapsed_seconds"]
        save_json(output / "timing.json", {"updates":50, "shape":400, "simulation_capture_seconds":elapsed, "estimated_1000_update_pair_seconds":elapsed*40})
    if args.stage != "analyze":
        timing = read_json(output / "timing.json")
        require(timing["estimated_1000_update_pair_seconds"] < 1200, "Timing estimate exceeds 20 minutes; report the constraint before proceeding")
        print("Timing estimate for pair:", timing["estimated_1000_update_pair_seconds"], "seconds", flush=True)
        started = time.perf_counter()
        wall = {}
        for n, c in configs.items():
            print("Running", n, c["steps"], "updates", flush=True)
            wall[n] = run(c, output / n)
        save_json(output / "completed.json", {"run_wall_seconds":wall, "pair_wall_seconds":time.perf_counter()-started})
    derived_outputs(output, configs)
    provenance(output)


if __name__ == "__main__": main()
