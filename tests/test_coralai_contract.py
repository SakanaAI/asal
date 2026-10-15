"""Tests of the shared format"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator

from scripts.validate_coralai_contract import (
    ContractError, ROOT, SCHEMA_PATHS, load_json, validate_config, validate_result,
)


class PilotContractTests(unittest.TestCase):
    def setUp(self):
        self.config = load_json(ROOT / "configs/coralai_pilot.default.json")
        self.result = load_json(ROOT / "examples/coralai_pilot_result.json")

    def test_schemas_and_checked_in_documents_are_valid(self):
        for path in SCHEMA_PATHS.values():
            Draft202012Validator.check_schema(load_json(path))
        validate_config(self.config)
        validate_result(self.result)
        self.assertEqual(self.config, self.result["effective_config"])

    def test_valid_configurations_and_disabled_features(self):
        for backend, device in [("cpu", "cpu"), ("cuda", "cuda"), ("metal", "mps")]:
            with self.subTest(backend=backend):
                config = copy.deepcopy(self.config)
                config["simulation"].update(
                    backend=backend, device=device, env="flat", env_param=None,
                    checkpoint_interval=0, emergency_pop_cap=0, infra_decay=1.0,
                    seed=4294967295,
                )
                original = copy.deepcopy(config)
                validate_config(config)
                self.assertEqual(config, original)

    def test_missing_unknown_and_unsupported_fields_are_rejected(self):
        for section in [None, "simulation", "evaluation", "output"]:
            for operation in ["missing", "unknown"]:
                with self.subTest(section=section, operation=operation):
                    config = copy.deepcopy(self.config)
                    target = config if section is None else config[section]
                    if operation == "missing":
                        del target[next(iter(target))]
                    else:
                        target["typo"] = 1
                    with self.assertRaises(ContractError):
                        validate_config(config)
        for version in [2, "1", True]:
            with self.subTest(version=version):
                self.config["schema_version"] = version
                with self.assertRaises(ContractError):
                    validate_config(self.config)

    def test_invalid_simulation_values_are_rejected(self):
        cases = [
            ("seed", -1), ("seed", 4294967296), ("seed", True),
            ("shape", 0), ("shape", 10.5), ("steps", 0), ("steps", "1000"),
            ("env", "unknown"), ("env_param", None), ("env_param", 0),
            ("env_param", 2.5), ("backend", "vulkan"), ("device", "cuda"),
            ("infra_decay", -0.1), ("infra_decay", 1.1), ("defense_coeff", -1),
            ("radiate_interval", 0), ("cleanup_interval", 0),
            ("log_interval", 0), ("checkpoint_interval", -1),
            ("emergency_pop_cap", -1),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(self.config)
                config["simulation"][field] = value
                with self.assertRaises(ContractError):
                    validate_config(config)
        self.config["simulation"]["env"] = "flat"
        with self.assertRaises(ContractError):
            validate_config(self.config)

    def test_invalid_evaluation_and_paths_are_rejected(self):
        for field, value in [
            ("prompt", ""), ("prompt", " \n\t"), ("bin_width", 0), ("bin_width", -1),
        ]:
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(self.config)
                config["evaluation"][field] = value
                with self.assertRaises(ContractError):
                    validate_config(config)
        for path in [
            "", "/tmp/pilot", "../pilot", "data/../pilot", "data/./pilot", ".",
            "data//pilot", "data/pilot/", "C:/pilot", "data\\pilot", "data/\npilot",
        ]:
            with self.subTest(path=path):
                self.config["output"]["root"] = path
                with self.assertRaises(ContractError):
                    validate_config(self.config)

    def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for text in [
                '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}',
                '{"x": 1e999}', '{"x": 1, "x": 2}', '{"x": {"y": 1, "y": 2}}',
            ]:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ContractError):
                        load_json(path)
        for value in [float("nan"), float("inf"), float("-inf"), 10 ** 400]:
            with self.subTest(value=value):
                self.config["simulation"]["defense_coeff"] = value
                with self.assertRaises(ContractError):
                    validate_config(self.config)

    def test_success_requires_complete_artifacts_and_no_error(self):
        for field in ["final_checkpoint", "final_image", "log", "metadata"]:
            with self.subTest(field=field):
                result = copy.deepcopy(self.result)
                result["artifacts"][field] = None
                with self.assertRaises(ContractError):
                    validate_result(result)
        self.result["error"] = {"code": "simulation_failed", "message": "Example"}
        with self.assertRaises(ContractError):
            validate_result(self.result)

    def test_failed_run_can_return_partial_or_missing_artifacts(self):
        self.result.update(status="failed", completed_steps=50,
                           error={"code": "simulation_failed", "message": "Example failure"})
        self.result["artifacts"] = {
            "checkpoints": [{"completed_steps": 0, "directory": "checkpoint_0000000"}],
            "final_checkpoint": None, "final_image": None,
            "log": "step_log.csv", "metadata": "meta.json",
        }
        validate_result(self.result)
        self.result["artifacts"].update(checkpoints=[], log=None, metadata=None)
        self.result["completed_steps"] = 0
        validate_result(self.result)
        self.result["error"] = None
        with self.assertRaises(ContractError):
            validate_result(self.result)

    def test_render_failure_after_completed_simulation_is_valid(self):
        self.result.update(
            status="failed", error={"code": "render_failed", "message": "Example failure"},
        )
        self.result["artifacts"]["final_image"] = None
        validate_result(self.result)

    def test_failure_cannot_call_an_earlier_checkpoint_final(self):
        self.result.update(
            status="failed", completed_steps=150,
            error={"code": "simulation_failed", "message": "Example failure"},
        )
        self.result["artifacts"]["checkpoints"] = self.result["artifacts"]["checkpoints"][:2]
        self.result["artifacts"].update(final_checkpoint="checkpoint_0000100", final_image=None)
        with self.assertRaises(ContractError):
            validate_result(self.result)
        self.result["artifacts"].update(final_checkpoint=None, final_image="images/final.png")
        with self.assertRaises(ContractError):
            validate_result(self.result)
        self.result["artifacts"]["final_image"] = None
        validate_result(self.result)

    def test_result_rejects_invalid_structure_and_effective_config(self):
        for field, value in [
            ("schema_version", 2), ("status", "pending"), ("run_id", "../run"),
            ("run_id", "run\n"), ("completed_steps", -1), ("completed_steps", True),
        ]:
            with self.subTest(field=field):
                result = copy.deepcopy(self.result)
                result[field] = value
                with self.assertRaises(ContractError):
                    validate_result(result)
        for section in [None, "artifacts", "provenance", "evaluation"]:
            for operation in ["missing", "unknown"]:
                with self.subTest(section=section, operation=operation):
                    result = copy.deepcopy(self.result)
                    target = result if section is None else result[section]
                    if operation == "missing":
                        del target[next(iter(target))]
                    else:
                        target["typo"] = 1
                    with self.assertRaises(ContractError):
                        validate_result(result)
        self.result["effective_config"]["simulation"]["steps"] = 0
        with self.assertRaises(ContractError):
            validate_result(self.result)

    def test_result_rejects_inconsistent_steps_and_checkpoints(self):
        mutations = [
            lambda r: r.update(completed_steps=999),
            lambda r: r.update(completed_steps=1001),
            lambda r: r["artifacts"]["checkpoints"].reverse(),
            lambda r: r["artifacts"]["checkpoints"].append(r["artifacts"]["checkpoints"][-1]),
            lambda r: r["artifacts"]["checkpoints"][-1].update(completed_steps=1001),
            lambda r: r["artifacts"]["checkpoints"][-1].update(directory="checkpoint_0000999"),
            lambda r: r["artifacts"].update(final_checkpoint="checkpoint_0000900"),
            lambda r: r["artifacts"].update(final_image="../final.png"),
            lambda r: r["provenance"].update(device="cuda"),
            lambda r: r["provenance"].update(git_commit="unknown"),
            lambda r: r["provenance"].update(git_commit="a" * 40 + "\n"),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                result = copy.deepcopy(self.result)
                mutate(result)
                with self.assertRaises(ContractError):
                    validate_result(result)

    def test_final_only_checkpoint_is_valid_when_periodic_saves_are_disabled(self):
        self.result["effective_config"]["simulation"]["checkpoint_interval"] = 0
        self.result["artifacts"]["checkpoints"] = [self.result["artifacts"]["checkpoints"][-1]]
        validate_result(self.result)

    def test_whole_valued_numbers_follow_json_schema_integer_rules(self):
        self.result["completed_steps"] = 1000.0
        simulation = self.result["effective_config"]["simulation"]
        for field, value in simulation.items():
            if isinstance(value, int):
                simulation[field] = float(value)
        for checkpoint in self.result["artifacts"]["checkpoints"]:
            checkpoint["completed_steps"] = float(checkpoint["completed_steps"])
        original = copy.deepcopy(self.result)
        validate_result(self.result)
        self.assertEqual(self.result, original)

    def test_checkpoint_schedule_includes_a_non_aligned_final_state(self):
        self.result["effective_config"]["simulation"]["steps"] = 1050
        self.result["completed_steps"] = 1050
        self.result["artifacts"]["checkpoints"].append(
            {"completed_steps": 1050, "directory": "checkpoint_0001050"}
        )
        self.result["artifacts"]["final_checkpoint"] = "checkpoint_0001050"
        validate_result(self.result)
        del self.result["artifacts"]["checkpoints"][1]
        with self.assertRaises(ContractError):
            validate_result(self.result)

    def test_scores_are_optional_and_use_the_existing_metric_names(self):
        original = copy.deepcopy(self.result)
        validate_result(self.result)
        self.assertEqual(self.result, original)
        self.result["evaluation"]["clip"] = {
            "model": "ASAL CLIP", "asal_target_loss": -0.25, "alignment": 0.25,
        }
        validate_result(self.result)
        self.result["evaluation"]["entropy"] = {
            "alive_cells": 50, "occupied_genome_ids": 2, "genome_shannon_bits": 1.0,
            "energy_binned_entropy_bits": 2.0, "infrastructure_binned_entropy_bits": 1.5,
            "bin_width": 1.0,
        }
        validate_result(self.result)
        scored = copy.deepcopy(self.result)
        mutations = [
            lambda r: r["evaluation"]["clip"].update(alignment=-0.25),
            lambda r: r["evaluation"]["clip"].update(asal_target_loss=float("nan")),
            lambda r: r["evaluation"]["entropy"].update(bin_width=2.0),
            lambda r: r["evaluation"]["entropy"].update(genome_shannon_bits=-1),
            lambda r: r["evaluation"]["entropy"].update(occupied_genome_ids=51),
            lambda r: r["evaluation"]["entropy"].update(alive_cells=10001),
            lambda r: r.update(
                status="failed", error={"code": "simulation_failed", "message": "Example"},
            ),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                result = copy.deepcopy(scored)
                mutate(result)
                with self.assertRaises(ContractError):
                    validate_result(result)

    def test_cli_can_validate_from_another_working_directory(self):
        script = ROOT / "scripts/validate_coralai_contract.py"
        with tempfile.TemporaryDirectory() as folder:
            for kind, path in [
                ("config", ROOT / "configs/coralai_pilot.default.json"),
                ("result", ROOT / "examples/coralai_pilot_result.json"),
            ]:
                with self.subTest(kind=kind):
                    completed = subprocess.run(
                        [sys.executable, str(script), kind, str(path)], cwd=folder,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
            invalid = Path(folder) / "invalid.json"
            invalid.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "config", str(invalid)], cwd=folder,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("Invalid config:", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
