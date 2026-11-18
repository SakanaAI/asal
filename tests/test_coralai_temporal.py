"""Check the exact ASAL objective, resource-free input, and safe parameter map."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from scripts.coralai_temporal import apply_parameters, compare_budgets, temporal_score, validate_setup
from substrates.coralai import render_state, render_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/dynamic_energy_demo/dynamic.json").read_text())
        self.settings = json.loads((ROOT / "configs/asal_novelty_pilot/search.json").read_text())

    def test_objective_compares_all_earlier_frames_and_preserves_zero_mask(self):
        # The third image repeats the first, even though adjacent images differ.
        score = temporal_score(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32))
        self.assertAlmostEqual(score["loss_oe"], 1 / 3, places=6)
        self.assertEqual(score["closest_prior_similarity"], [0, 0, 1])
        unchanged = temporal_score(np.tile([[1, 0]], (4, 1)).astype(np.float32))
        self.assertAlmostEqual(unchanged["loss_oe"], 0.75)
        self.assertAlmostEqual(unchanged["novelty"], 0)
        opposite = temporal_score(np.array([[1, 0], [-1, 0]], dtype=np.float32))
        self.assertEqual(opposite["loss_oe"], 0)  # ASAL's masked zeros floor negatives.
        with self.assertRaises(ValueError):
            temporal_score(np.array([[2, 0], [2, 0]], dtype=np.float32))

    def test_occupancy_view_ignores_resources_and_genome_identity(self):
        state = torch.zeros(1, 14, 8, 8)
        state[0, 13].fill_(-1)
        state[0, 13, 2:4, 5:7] = 0
        before = render_state(state, "occupancy")
        changed = state.clone()
        changed[0, :13] = torch.arange(13 * 64).reshape(13, 8, 8)
        changed[0, 13, 2:4, 5:7] = 99
        self.assertTrue(np.array_equal(before, render_state(changed, "occupancy")))
        self.assertFalse(np.array_equal(render_state(state), render_state(changed)))
        changed[0, 13, 0, 0] = 1
        self.assertFalse(np.array_equal(before, render_state(changed, "occupancy")))

    def test_original_rgb_renderer_default_is_unchanged(self):
        baseline = subprocess.check_output(["git", "show", "452f3b2:substrates/coralai.py"], cwd=ROOT, text=True)
        namespace = {"__name__": "baseline_renderer"}
        exec(compile(baseline, "original_renderer", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            state = torch.arange(14 * 64, dtype=torch.float32).reshape(1, 14, 8, 8) / 100
            torch.save(state, path)
            self.assertTrue(np.array_equal(namespace["render_checkpoint"](path), render_checkpoint(path)))

    def test_only_translation_is_mapped_and_source_config_is_preserved(self):
        original = copy.deepcopy(self.config)
        mapped = apply_parameters(self.config, [-1, 2], self.settings["parameters"])
        self.assertEqual(mapped["env_options"]["shift_interval"], 10)
        self.assertEqual(mapped["env_options"]["shift_cells"], 8)
        mapped["env_options"].update({key: original["env_options"][key] for key in self.settings["parameters"]})
        self.assertEqual(mapped, original)
        self.assertEqual(self.config, original)
        with self.assertRaises(ValueError):
            apply_parameters(self.config, [1], {"inject_rate": {"min": 1, "max": 2, "integer": True}})

    def test_schema_rejects_energy_search_and_invalid_sampling(self):
        from jsonschema import ValidationError
        validate_setup(self.config, self.settings)
        bad = copy.deepcopy(self.settings)
        bad["parameters"]["inject_rate"] = {"min": 1, "max": 2, "integer": True}
        with self.assertRaises(ValidationError):
            validate_setup(self.config, bad)
        bad = copy.deepcopy(self.settings)
        bad["sampled_updates"][0] = 19
        with self.assertRaises(ValueError):
            validate_setup(self.config, bad)

    def test_budget_check_detects_actual_supply_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(torch.zeros(1, 14, 2, 2), path / "initial_state.pt")
            score = {"run": str(path), "measurements": [{"completed_update": 20, "cumulative_requested": 1,
                     "remaining_reserve": 9, "cumulative_delivered": 1, "cumulative_clipped": 0}]}
            other = copy.deepcopy(score)
            self.assertTrue(compare_budgets([score, other])["initial_states_identical"])
            other["measurements"][0]["cumulative_delivered"] = 0.9
            with self.assertRaises(ValueError):
                compare_budgets([score, other])


if __name__ == "__main__":
    unittest.main()
