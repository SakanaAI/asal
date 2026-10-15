"""Validate artifact checks with synthetic saved tensors, without running Coralai."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.score_entropy import score_checkpoint
from scripts.validate_coralai_contract import ROOT, load_json
from scripts.validate_coralai_pilot import ENTROPY_FIELDS, compare_pilots, validate_pilot


class PilotArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.manifest = self.make_run("baseline")

    def make_run(self, name, **settings):
        run_dir = self.folder / name
        run_dir.mkdir()
        result = load_json(ROOT / "examples/coralai_pilot_result.json")
        result.update(run_id=name, completed_steps=3)
        simulation = result["effective_config"]["simulation"]
        simulation.update(shape=4, steps=3, checkpoint_interval=2, **settings)
        result["artifacts"]["checkpoints"] = []
        state = torch.zeros((1, 14, 4, 4))
        state[0, 0] = torch.arange(16).reshape(4, 4)
        state[0, 1] = 1.0
        for step in (0, 2, 3):
            directory = f"checkpoint_{step:07d}"
            checkpoint = run_dir / directory
            checkpoint.mkdir()
            torch.save(state, checkpoint / "substrate.pt")
            # The validator checks this file's presence and never unpickles it.
            (checkpoint / "population.pkl").write_bytes(b"synthetic population fixture")
            (checkpoint / "meta.json").write_text(json.dumps({"step": step, "shape": [4, 4]}))
            result["artifacts"]["checkpoints"].append(
                {"completed_steps": step, "directory": directory}
            )
        result["artifacts"]["final_checkpoint"] = "checkpoint_0000003"
        (run_dir / "images").mkdir()
        render = runpy.run_path(str(ROOT / "substrates/coralai.py"))["render_checkpoint"]
        rgb = render(run_dir / "checkpoint_0000003/substrate.pt")
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(run_dir / "images/final.png")
        (run_dir / "step_log.csv").write_text("step,total_energy\n0,120\n3,120\n")
        (run_dir / "meta.json").write_text(
            json.dumps({"shape": [4, 4], "seed": simulation["seed"]})
        )
        path = run_dir / "result.json"
        path.write_text(json.dumps(result))
        return path

    def test_artifacts_are_validated_and_input_is_preserved(self):
        original = self.manifest.read_bytes()
        report = validate_pilot(self.manifest)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checkpoints_checked"], 3)
        self.assertEqual(report["entropy"]["alive_cells"], 16)
        self.assertIsNone(report["stored_entropy_matches"])
        self.assertFalse(report["clip_recomputed"])
        self.assertFalse(report["simulation_rerun"])
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_missing_population_is_rejected(self):
        (self.manifest.parent / "checkpoint_0000002/population.pkl").unlink()
        with self.assertRaisesRegex(ValueError, "Missing artifact"):
            validate_pilot(self.manifest)

    def test_bad_tensor_and_checkpoint_metadata_are_rejected(self):
        state_path = self.manifest.parent / "checkpoint_0000003/substrate.pt"
        original = torch.load(state_path, weights_only=True)
        for state in (
            torch.zeros((1, 13, 4, 4)), original * float("nan"), original.to(torch.int32),
        ):
            with self.subTest(shape=state.shape):
                torch.save(state, state_path)
                with self.assertRaises(ValueError):
                    validate_pilot(self.manifest)
        torch.save(original, state_path)
        metadata = self.manifest.parent / "checkpoint_0000003/meta.json"
        metadata.write_text(json.dumps({"shape": [4, 4], "step": 2}))
        with self.assertRaisesRegex(ValueError, "metadata step"):
            validate_pilot(self.manifest)

    def test_wrong_final_image_is_rejected(self):
        Image.new("RGB", (224, 224), (255, 255, 255)).save(
            self.manifest.parent / "images/final.png"
        )
        with self.assertRaisesRegex(ValueError, "differs from rendering"):
            validate_pilot(self.manifest)

    def test_stored_entropy_is_recomputed(self):
        result = load_json(self.manifest)
        measured = score_checkpoint(self.manifest.parent / "checkpoint_0000003/substrate.pt")
        result["evaluation"]["entropy"] = {key: measured[key] for key in ENTROPY_FIELDS}
        self.manifest.write_text(json.dumps(result))
        self.assertTrue(validate_pilot(self.manifest)["stored_entropy_matches"])
        result["evaluation"]["entropy"]["genome_shannon_bits"] += 1.0
        self.manifest.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "Stored entropy"):
            validate_pilot(self.manifest)

    def test_comparison_requires_matching_settings_and_preserves_unscored_status(self):
        candidate = self.make_run("candidate", infra_decay=0.002)
        report = compare_pilots(self.manifest, candidate, "infra_decay")
        self.assertEqual(report["candidate_value"], 0.002)
        self.assertIsNone(report["stored_target_loss_delta_candidate_minus_baseline"])
        self.assertEqual(report["entropy_delta_candidate_minus_baseline"]["alive_cells"], 0)
        mismatched = self.make_run("other-seed", infra_decay=0.002, seed=43)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            compare_pilots(self.manifest, mismatched, "infra_decay")
        with self.assertRaisesRegex(ValueError, "Unsupported search parameter"):
            compare_pilots(self.manifest, candidate, "seed")

    def test_cli_records_failure_and_does_not_overwrite_manifest(self):
        script = ROOT / "scripts/validate_coralai_pilot.py"
        (self.manifest.parent / "images/final.png").unlink()
        output = self.folder / "failure.json"
        command = [sys.executable, str(script), str(self.manifest), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(load_json(output)["status"], "failed")
        self.assertIn("Missing artifact", load_json(output)["message"])
        for path in (
            self.manifest, self.manifest.parent / "meta.json",
            self.manifest.parent / "checkpoint_0000003/substrate.pt",
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                command[-1] = str(path)
                completed = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
