"""Check temporal scoring, bounded configuration decoding, and paired reporting."""

import copy
import json
from pathlib import Path
import unittest

from scripts.coralai_search import average_across_seeds, decode, frame_summary, paired_summary, validate_setup

ROOT = Path(__file__).resolve().parents[1]


class CoralaiSearchTests(unittest.TestCase):
    def test_objective_uses_all_frames_and_keeps_best_and_final_distinct(self):
        score = frame_summary([50, 100, 150], [-0.3, -0.5, -0.1])
        self.assertAlmostEqual(score["mean_alignment"], 0.3)
        self.assertEqual(score["best_sample_step"], 100)
        self.assertEqual(score["best_sample_alignment"], 0.5)
        self.assertEqual(score["final_alignment"], 0.1)
        with self.assertRaises(ValueError):
            frame_summary([50], [float("nan")])

    def test_decoded_candidates_are_bounded_and_integer_intervals(self):
        settings = json.loads((ROOT / "configs/coralai_supervised.search.json").read_text())
        self.assertEqual(decode([-1, 2], settings["parameters"]),
                         {"radiate_interval": 10, "infra_decay": 0.01})
        self.assertIsInstance(decode([0.123, 0.321], settings["parameters"])["radiate_interval"], int)

    def test_comparison_sign_and_sample_variation(self):
        rows = [{"default": {"mean_alignment": 0.3}, "selected": {"mean_alignment": value},
                 "difference": value - 0.3} for value in (0.2, 0.4, 0.6)]
        summary = paired_summary(rows)
        self.assertEqual(summary["improved_seeds"], 2)
        self.assertAlmostEqual(summary["paired_difference_mean"], 0.1)
        self.assertAlmostEqual(summary["paired_difference_sample_std"], 0.2)

    def test_holdout_seeds_and_full_sampling_are_required(self):
        config = json.loads((ROOT / "configs/coralai_supervised.default.json").read_text())
        settings = json.loads((ROOT / "configs/coralai_supervised.search.json").read_text())
        validate_setup(config, settings)
        modified = copy.deepcopy(settings)
        modified["comparison_seeds"][0] = settings["search_seeds"][0]
        with self.assertRaises(ValueError):
            validate_setup(config, modified)
        single_seed = copy.deepcopy(settings)
        single_seed["search_seeds"] = settings["search_seeds"][:1]
        with self.assertRaises(ValueError):
            validate_setup(config, single_seed)
        config["max_screenshots"] = 20
        with self.assertRaises(ValueError):
            validate_setup(config, settings)

    def test_average_across_seeds_rewards_high_mean_and_flags_high_spread(self):
        consistent = [{"seed": seed, "mean_loss": -0.5, "mean_alignment": 0.5,
                        "best_sample_alignment": 0.6, "final_alignment": 0.5,
                        "simulation_seconds": 1.0, "scoring_seconds": 0.5} for seed in (1, 2, 3)]
        summary = average_across_seeds(consistent)
        self.assertAlmostEqual(summary["mean_alignment"], 0.5)
        self.assertAlmostEqual(summary["alignment_std"], 0.0)
        self.assertEqual(summary["seeds"], [1, 2, 3])
        self.assertEqual(summary["n_seeds"], 3)

        lucky_once = [{"seed": seed, "mean_loss": loss, "mean_alignment": -loss,
                        "best_sample_alignment": -loss, "final_alignment": -loss,
                        "simulation_seconds": 1.0, "scoring_seconds": 0.5}
                      for seed, loss in zip((1, 2, 3), (-0.9, 0.1, 0.1))]
        lucky_summary = average_across_seeds(lucky_once)
        # Same-ish mean alignment can hide behind a much higher spread across seeds.
        self.assertGreater(lucky_summary["alignment_std"], summary["alignment_std"])

        with self.assertRaises(ValueError):
            average_across_seeds(consistent[:1])


if __name__ == "__main__":
    unittest.main()
