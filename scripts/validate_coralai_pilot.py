"""Check real pilot artifacts and optionally compare a candidate with its baseline."""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import pickle
import runpy
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_entropy import score_checkpoint
from scripts.validate_coralai_contract import load_json, validate_result


ENTROPY_FIELDS = (
    "alive_cells", "occupied_genome_ids", "genome_shannon_bits",
    "energy_binned_entropy_bits", "infrastructure_binned_entropy_bits", "bin_width",
)
SEARCH_PARAMETERS = (
    "env_param", "infra_decay", "defense_coeff", "radiate_interval",
    "cleanup_interval", "emergency_pop_cap",
)


def artifact_path(run_dir, relative, *, directory=False):
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Artifact resolves outside the run directory: {relative}")
    if not path.exists():
        raise ValueError(f"Missing artifact: {relative}")
    if not (path.is_dir() if directory else path.is_file()):
        kind = "directory" if directory else "file"
        raise ValueError(f"Expected {kind} artifact: {relative}")
    return path


def validate_pilot(manifest_path):
    """Validate saved files and reproduce rendering/entropy, without rerunning a simulation."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json(manifest_path)
    validate_result(manifest)
    if manifest["status"] != "success":
        raise ValueError(f"Runner failed: {manifest['error']['message']}")

    run_dir = manifest_path.parent
    config = manifest["effective_config"]
    simulation = config["simulation"]
    shape = int(simulation["shape"])
    artifacts = manifest["artifacts"]
    for checkpoint in artifacts["checkpoints"]:
        directory = artifact_path(run_dir, checkpoint["directory"], directory=True)
        state_path = artifact_path(directory, "substrate.pt")
        artifact_path(directory, "population.pkl")
        metadata = load_json(artifact_path(directory, "meta.json"))
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON object.")
        if metadata.get("shape") != [shape, shape]:
            raise ValueError(f"Checkpoint grid shape differs from config: {directory.name}")
        if metadata.get("step") != checkpoint["completed_steps"]:
            raise ValueError(f"Checkpoint metadata step differs from manifest: {directory.name}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(state) or tuple(state.shape) != (1, 14, shape, shape):
            raise ValueError(f"Invalid 14-channel cell tensor: {directory.name}")
        if not state.is_floating_point():
            raise ValueError(f"Cell tensor must contain floating point values: {directory.name}")
        if not torch.isfinite(state).all():
            raise ValueError(f"Non-finite cell values: {directory.name}")
        genome = state[0, 13]
        if not torch.equal(genome, torch.floor(genome)):
            raise ValueError(f"Non-integer genome IDs: {directory.name}")

    metadata = load_json(artifact_path(run_dir, artifacts["metadata"]))
    if not isinstance(metadata, dict):
        raise ValueError("Run metadata must be a JSON object.")
    verified_settings = []
    if "effective_config" in metadata:
        if metadata["effective_config"] != config:
            raise ValueError("Run metadata effective_config differs from manifest.")
        verified_settings = sorted(simulation)
    else:
        for key in ("seed", "infra_decay", "defense_coeff", "cleanup_interval", "emergency_pop_cap"):
            if key in metadata:
                if metadata[key] != simulation[key]:
                    raise ValueError(f"Run metadata {key} differs from config.")
                verified_settings.append(key)
        if "shape" in metadata:
            if metadata["shape"] != [shape, shape]:
                raise ValueError("Run metadata shape differs from config.")
            verified_settings.append("shape")
        if "env" in metadata:
            environment = metadata["env"]
            if (
                not isinstance(environment, dict)
                or environment.get("name") != simulation["env"]
            ):
                raise ValueError("Run metadata environment differs from config.")
            verified_settings.append("env")
            if simulation["env"] == "patches":
                if environment.get("n_patches") != simulation["env_param"]:
                    raise ValueError("Run metadata patch count differs from config.")
                verified_settings.append("env_param")
    if metadata.get("start_step", 0) != 0 or metadata.get("resumed_from") is not None:
        raise ValueError("Pilot config requires a fresh run, not a resumed checkpoint.")

    log_path = artifact_path(run_dir, artifacts["log"])
    with log_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            raise ValueError("Step log must have a step column.")
        log_rows = sum(1 for _ in reader)
    if not log_rows:
        raise ValueError("Step log has no data rows.")

    final_directory = artifact_path(run_dir, artifacts["final_checkpoint"], directory=True)
    final_state = artifact_path(final_directory, "substrate.pt")
    # Load the existing renderer directly; importing substrates also imports JAX.
    render = runpy.run_path(str(ROOT / "substrates/coralai.py"))["render_checkpoint"]
    expected_pixels = np.rint(render(final_state) * 255).astype(np.uint8)
    with Image.open(artifact_path(run_dir, artifacts["final_image"])) as image:
        if image.mode != "RGB" or image.size != (224, 224):
            raise ValueError("Final image must be a 224 x 224 RGB image.")
        if not np.array_equal(np.asarray(image), expected_pixels):
            raise ValueError("Final image differs from rendering its saved checkpoint.")

    entropy = score_checkpoint(final_state, config["evaluation"]["bin_width"])
    measured = {key: entropy[key] for key in ENTROPY_FIELDS}
    stored = manifest["evaluation"]["entropy"]
    if stored is not None:
        for key in ENTROPY_FIELDS:
            if not math.isclose(stored[key], measured[key], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Stored entropy differs from recomputation: {key}")
    return {
        "status": "passed",
        "run_id": manifest["run_id"],
        "completed_steps": manifest["completed_steps"],
        "checkpoints_checked": len(artifacts["checkpoints"]),
        "log_rows": log_rows,
        "image_matches_checkpoint": True,
        "entropy": measured,
        "stored_entropy_matches": True if stored is not None else None,
        "clip_score": manifest["evaluation"]["clip"],
        "clip_recomputed": False,
        "settings_verified_from_metadata": verified_settings,
        "simulation_rerun": False,
    }


def compare_pilots(baseline_path, candidate_path, parameter):
    """Require matching evaluation settings before comparing freshly computed entropy."""
    if parameter not in SEARCH_PARAMETERS:
        raise ValueError(f"Unsupported search parameter: {parameter}")
    documents = [load_json(path) for path in (baseline_path, candidate_path)]
    normalized = []
    for document in documents:
        validate_result(document)
        config = copy.deepcopy(document["effective_config"])
        del config["simulation"][parameter]
        del config["output"]["root"]
        normalized.append(config)
    if normalized[0] != normalized[1]:
        raise ValueError("Baseline and candidate settings differ beyond the searched parameter.")
    baseline = validate_pilot(baseline_path)
    candidate = validate_pilot(candidate_path)
    loss_delta = None
    if baseline["clip_score"] is not None and candidate["clip_score"] is not None:
        loss_delta = (
            candidate["clip_score"]["asal_target_loss"]
            - baseline["clip_score"]["asal_target_loss"]
        )
    return {
        "status": "passed",
        "parameter": parameter,
        "baseline_value": documents[0]["effective_config"]["simulation"][parameter],
        "candidate_value": documents[1]["effective_config"]["simulation"][parameter],
        "baseline": baseline,
        "candidate": candidate,
        "entropy_delta_candidate_minus_baseline": {
            key: candidate["entropy"][key] - baseline["entropy"][key]
            for key in ENTROPY_FIELDS if key != "bin_width"
        },
        "stored_target_loss_delta_candidate_minus_baseline": loss_delta,
        "loss_interpretation": "A negative target loss delta favors the candidate.",
        "provenance_records_match": documents[0]["provenance"] == documents[1]["provenance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--parameter", choices=SEARCH_PARAMETERS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.baseline is None) != (args.parameter is None):
        parser.error("--baseline and --parameter must be supplied together")
    run_directories = [args.manifest.resolve().parent]
    if args.baseline:
        run_directories.append(args.baseline.resolve().parent)
    if any(args.output.resolve().is_relative_to(path) for path in run_directories):
        parser.error("--output must be outside the input run directories")
    try:
        if args.baseline:
            report = compare_pilots(args.baseline, args.manifest, args.parameter)
        else:
            report = validate_pilot(args.manifest)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as exc:
        report = {"status": "failed", "message": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    parser.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
