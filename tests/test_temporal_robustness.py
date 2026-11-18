"""Tests for the multi-seed / held-out helpers behind issue #37 (pure numpy)."""

import unittest

import numpy as np

from scripts.temporal_robustness import (aggregate_losses, change_summary, changed_fraction,
                                         paired_heldout, validate_seed_plan)


class SeedPlanTests(unittest.TestCase):
    def test_accepts_a_sound_plan(self):
        validate_seed_plan(303, [303, 7, 13], [101, 202, 404], 1.0)

    def test_rejects_unsound_plans(self):
        bad_plans = [
            (303, [303], [101, 202, 404], 1.0),             # one seed cannot be averaged
            (303, [303, 303, 7], [101, 202, 404], 1.0),     # duplicate search seed
            (303, [7, 303], [101, 202, 404], 1.0),          # primary seed must come first
            (303, [303, 7], [101, 202], 1.0),               # too few held-out seeds
            (303, [303, 7], [101, 7, 404], 1.0),            # held-out overlaps search
            (303, [303, 7], [101, 202, 404], -0.1),         # negative penalty
            (303, [303, 7], [101, 202, 404], float("nan")),
            (303, [303, True], [101, 202, 404], 1.0),       # bool is not a seed
            (303, [303, 2 ** 32], [101, 202, 404], 1.0),    # out of range
        ]
        for plan in bad_plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_seed_plan(*plan)


class AggregateTests(unittest.TestCase):
    def test_plain_mean_when_penalty_is_zero(self):
        summary = aggregate_losses([0.9, 0.8, 0.7])
        self.assertAlmostEqual(summary["mean_loss"], 0.8)
        self.assertAlmostEqual(summary["selection_loss"], 0.8)
        self.assertEqual(summary["n_seeds"], 3)

    def test_steady_world_beats_erratic_world_only_with_a_penalty(self):
        erratic = [0.50, 0.90, 0.10]   # mean 0.50, lucky on one seed
        steady = [0.52, 0.51, 0.53]    # slightly worse mean, very consistent
        self.assertLess(aggregate_losses(erratic)["selection_loss"], aggregate_losses(steady)["selection_loss"])
        self.assertGreater(aggregate_losses(erratic, 1.0)["selection_loss"], aggregate_losses(steady, 1.0)["selection_loss"])

    def test_uses_sample_std_and_rejects_bad_input(self):
        self.assertAlmostEqual(aggregate_losses([1.0, 3.0])["loss_std"], np.sqrt(2.0))
        for bad in ([0.5], [0.5, float("nan")], [[0.1, 0.2], [0.3, 0.4]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aggregate_losses(bad)
        with self.assertRaises(ValueError):
            aggregate_losses([0.1, 0.2], -1)


class HeldoutTests(unittest.TestCase):
    def test_positive_difference_means_selected_is_more_novel(self):
        result = paired_heldout([0.90, 0.90, 0.90], [0.80, 0.85, 0.89])
        self.assertEqual(result["improved_seeds"], 3)
        self.assertGreater(result["paired_difference_mean"], 0)
        self.assertTrue(result["held_up"])

    def test_tiny_gaps_count_as_ties_and_do_not_hold_up(self):
        result = paired_heldout([0.9, 0.9, 0.9], [0.9 - 1e-9] * 3)
        self.assertEqual(result["improved_seeds"], 0)
        self.assertFalse(result["held_up"])

    def test_winning_one_seed_by_a_lot_is_not_enough(self):
        # Selected wins big on one seed but loses on two: the mean is positive, the majority is not.
        result = paired_heldout([0.9, 0.9, 0.9], [0.5, 0.91, 0.92])
        self.assertEqual(result["improved_seeds"], 1)
        self.assertGreater(result["paired_difference_mean"], 0)
        self.assertFalse(result["held_up"])

    def test_rejects_mismatched_or_nonfinite_input(self):
        for hand, selected in (([0.9, 0.9], [0.9]), ([0.9], [0.9]), ([0.9, float("inf")], [0.9, 0.9])):
            with self.subTest(hand=hand), self.assertRaises(ValueError):
                paired_heldout(hand, selected)


class ChangeTests(unittest.TestCase):
    def grid(self, *cells):
        frame = np.zeros((4, 4), dtype=bool)
        for row, col in cells:
            frame[row, col] = True
        return frame

    def test_changed_fraction(self):
        self.assertEqual(changed_fraction(self.grid(), self.grid()), 0.0)
        self.assertEqual(changed_fraction(self.grid((0, 0)), self.grid((1, 1))), 2 / 16)
        with self.assertRaises(ValueError):
            changed_fraction(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_a_world_that_freezes_scores_low_late_change(self):
        moving = [self.grid((0, i % 4)) for i in range(6)]
        frozen = [self.grid((0, 0)), self.grid((0, 1)), self.grid((0, 2))] + [self.grid((0, 3))] * 5
        self.assertGreater(change_summary(moving)["late_change"], 0)
        self.assertEqual(change_summary(frozen)["late_change"], 0.0)
        self.assertGreater(change_summary(frozen)["ongoing_change"], 0)  # it did move early on

    def test_needs_enough_frames(self):
        with self.assertRaises(ValueError):
            change_summary([self.grid(), self.grid()])


if __name__ == "__main__":
    unittest.main()
