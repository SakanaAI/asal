"""Validate the shared pilot format without importing simulation or model code."""

import argparse
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATHS = {
    "config": ROOT / "schemas/coralai_pilot_config.schema.json",
    "result": ROOT / "schemas/coralai_pilot_result.schema.json",
}


class ContractError(ValueError):
    """A document violates the pilot's structure or semantic rules."""


def _reject_constant(value):
    raise ContractError(f"Non-finite JSON number: {value}")


def _unique_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ContractError(f"Duplicate JSON key: {key}")
        document[key] = value
    return document


def _check_finite(value):
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ContractError("All numbers must be finite and fit floating point range.")
    if isinstance(value, dict):
        for child in value.values():
            _check_finite(child)
    elif isinstance(value, list):
        for child in value:
            _check_finite(child)


def load_json(path):
    """Read strict JSON, rejecting duplicate keys and non-finite numbers."""
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(
            stream, parse_constant=_reject_constant, object_pairs_hook=_unique_keys
        )
    _check_finite(document)
    return document


def _validate_schema(document, kind):
    _check_finite(document)
    schemas = {name: load_json(path) for name, path in SCHEMA_PATHS.items()}
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema))
        for schema in schemas.values()
    )
    validator = Draft202012Validator(schemas[kind], registry=registry)
    try:
        validator.validate(document)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "document"
        raise ContractError(f"{location}: {exc.message}") from exc


def validate_config(config):
    """Validate a complete v1 configuration; never fill defaults or mutate it."""
    _validate_schema(config, "config")


def validate_result(result):
    """Validate a manifest, including relationships JSON Schema cannot express."""
    _validate_schema(result, "result")
    config = result["effective_config"]
    simulation = config["simulation"]
    completed = int(result["completed_steps"])
    if completed > simulation["steps"]:
        raise ContractError("completed_steps exceeds the requested steps.")
    if result["status"] == "success" and completed != simulation["steps"]:
        raise ContractError("A successful run must complete the requested steps.")
    for field in ("backend", "device"):
        if result["provenance"][field] != simulation[field]:
            raise ContractError(f"provenance.{field} differs from effective_config.")

    artifacts = result["artifacts"]
    checkpoints = artifacts["checkpoints"]
    previous = -1
    for checkpoint in checkpoints:
        step = int(checkpoint["completed_steps"])
        if step <= previous or step > completed:
            raise ContractError("Checkpoint steps must increase and not exceed completed_steps.")
        if checkpoint["directory"] != f"checkpoint_{step:07d}":
            raise ContractError("Checkpoint directory must encode its completed update count.")
        previous = step

    if result["status"] == "success":
        interval = int(simulation["checkpoint_interval"])
        expected_count = (
            completed // interval + 1 + int(completed % interval != 0)
            if interval else 1
        )
        if len(checkpoints) != expected_count:
            raise ContractError("Saved checkpoints do not match checkpoint_interval.")
        for index, checkpoint in enumerate(checkpoints):
            expected_step = min(index * interval, completed) if interval else completed
            if checkpoint["completed_steps"] != expected_step:
                raise ContractError("Saved checkpoints do not match checkpoint_interval.")

    final = artifacts["final_checkpoint"]
    if final is not None:
        if not checkpoints or checkpoints[-1]["directory"] != final:
            raise ContractError("final_checkpoint must identify the last listed checkpoint.")
        if checkpoints[-1]["completed_steps"] != completed:
            raise ContractError("final_checkpoint must represent completed_steps.")
    if artifacts["final_image"] is not None and final is None:
        raise ContractError("A final image requires a final checkpoint.")

    clip = result["evaluation"]["clip"]
    if clip is not None and not math.isclose(
        clip["alignment"], -clip["asal_target_loss"], rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ContractError("alignment must equal -asal_target_loss.")
    entropy = result["evaluation"]["entropy"]
    if entropy is not None:
        if entropy["bin_width"] != config["evaluation"]["bin_width"]:
            raise ContractError("Entropy bin_width differs from effective_config.")
        if not (
            entropy["occupied_genome_ids"]
            <= entropy["alive_cells"]
            <= simulation["shape"] ** 2
        ):
            raise ContractError("Entropy counts exceed living cells or grid size.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("config", "result"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        document = load_json(args.path)
        validator = validate_config if args.kind == "config" else validate_result
        validator(document)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Invalid {args.kind}: {exc}\n")
    print(f"Valid pilot {args.kind}: {args.path}")


if __name__ == "__main__":
    main()
