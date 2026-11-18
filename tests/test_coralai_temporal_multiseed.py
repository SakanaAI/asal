"""Tests for the issue #37 multi-seed driver that need no simulator, jax or CLIP."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.coralai_temporal import validate_setup
from scripts.coralai_temporal_multiseed import evaluate_world, split_settings, with_seed

ROOT = Path(__file__).resolve().parents[1]


class FakeEvaluator:
    """Records how it is called and returns a fixed loss per seed."""

    def __init__(self, loss_by_seed):
        self.loss_by_seed, self.calls = loss_by_seed, []

    def evaluate(self, config, directory, *, reuse=None, updates=None):
        directory.mkdir(parents=True, exist_ok=False)
        self.calls.append({"seed": config["seed"], "directory": directory, "reuse": reuse,
                           "capture": config["screenshot_interval"]})
        return {"loss_oe": self.loss_by_seed[config["seed"]], "novelty": 0.9 - self.loss_by_seed[config["seed"]],
                "simulation_seconds": 1.0, "scoring_seconds": 0.5, "run": str(directory), "measurements": []}


class MultiSeedSettingsTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/dynamic_energy_demo/dynamic.json").read_text())
        self.settings = json.loads((ROOT / "configs/asal_novelty_pilot/search_multiseed.json").read_text())

    def test_extra_keys_are_split_off_and_the_rest_passes_the_pilot_validation(self):
        base, plan = split_settings(self.settings)
        self.assertNotIn("search_seeds", base)
        self.assertEqual(plan["search_seeds"][0], self.settings["simulation_seed"])
        validate_setup(self.config, base)

    def test_pilot_settings_without_a_seed_plan_are_rejected(self):
        pilot = json.loads((ROOT / "configs/asal_novelty_pilot/search.json").read_text())
        with self.assertRaises(ValueError):
            split_settings(pilot)

    def test_overlapping_held_out_seeds_are_rejected(self):
        bad = copy.deepcopy(self.settings)
        bad["comparison_seeds"][0] = bad["search_seeds"][1]
        with self.assertRaises(ValueError):
            split_settings(bad)

    def test_with_seed_does_not_change_the_original(self):
        original = copy.deepcopy(self.config)
        self.assertEqual(with_seed(self.config, 7)["seed"], 7)
        self.assertEqual(self.config, original)


class EvaluateWorldTests(unittest.TestCase):
    def test_first_seed_lives_in_the_world_folder_and_only_it_may_reuse_a_cached_run(self):
        evaluator = FakeEvaluator({303: 0.9, 7: 0.8, 13: 0.7})
        config, later = {"seed": 303, "screenshot_interval": 5}, {"seed": 303, "screenshot_interval": 20}
        with tempfile.TemporaryDirectory() as tmp:
            world = Path(tmp) / "hand"
            result = evaluate_world(evaluator, config, [303, 7, 13], world, 0.0, reuse=Path("cached"), later_config=later)
            self.assertTrue((world / "multi_seed_summary.json").is_file())
        first, second, third = evaluator.calls
        self.assertEqual((first["directory"], first["reuse"], first["capture"]), (world, Path("cached"), 5))
        self.assertEqual((second["directory"], second["reuse"], second["capture"]), (world / "seed_7", None, 20))
        self.assertEqual(third["directory"], world / "seed_13")
        self.assertAlmostEqual(result["mean_loss"], 0.8)
        self.assertEqual(result["seeds"], [303, 7, 13])
        self.assertEqual(result["loss_oe"], 0.9)  # the first seed's own score is kept for the figures

    def test_selection_loss_charges_for_disagreement_between_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            steady = evaluate_world(FakeEvaluator({1: 0.52, 2: 0.51, 3: 0.53}), {"seed": 1, "screenshot_interval": 5},
                                    [1, 2, 3], Path(tmp) / "steady", 1.0)
            erratic = evaluate_world(FakeEvaluator({1: 0.50, 2: 0.90, 3: 0.10}), {"seed": 1, "screenshot_interval": 5},
                                     [1, 2, 3], Path(tmp) / "erratic", 1.0)
        self.assertLess(erratic["mean_loss"], steady["mean_loss"])
        self.assertGreater(erratic["selection_loss"], steady["selection_loss"])


if __name__ == "__main__":
    unittest.main()
