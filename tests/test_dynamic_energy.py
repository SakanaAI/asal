"""Check finite source translation, real supply accounting, and saved configs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "external/coralai/experiments/coral_dev"
sys.path.insert(0, str(EXPERIMENT))
from environments import TrailEnvironment, make_env


def substrate():
    mem = torch.zeros(1, 14, 16, 16)
    mem[0, :2] = 1
    mem[0, 13] = torch.arange(256).reshape(16, 16) % 7 - 1
    return SimpleNamespace(mem=mem, torch_device=torch.device("cpu"),
                           ti_indices={None: SimpleNamespace(energy=0, infra=1, genome=13)})


class DynamicEnergyTests(unittest.TestCase):
    def test_translation_preserves_finite_budget_and_does_not_move_cells(self):
        envs = [TrailEnvironment(max_capacity=0.15, inject_rate=0.04, shift_interval=i, shift_cells=3) for i in (0, 2)]
        subs = [substrate(), substrate()]
        for env, sub in zip(envs, subs):
            env.seed(sub)
        self.assertTrue(torch.equal(subs[0].mem, subs[1].mem))
        initial_genomes = subs[0].mem[0, 13].clone()
        initial_infra = subs[0].mem[0, 1].clone()
        initial_capacity = envs[0]._capacity.clone()
        for _ in range(15):
            for env, sub in zip(envs, subs):
                env.step(sub)
                self.assertTrue(torch.equal(sub.mem[0, 13], initial_genomes))
                self.assertTrue(torch.equal(sub.mem[0, 1], initial_infra))
            self.assertEqual(envs[0].resource_state()["remaining_capacity"], envs[1].resource_state()["remaining_capacity"])
        for env in envs:
            budget = env.resource_state()
            self.assertEqual(budget["completed_update"], 15)
            self.assertEqual(budget["remaining_capacity"], 0)
            # The reserve is float32; repeated subtraction has a small rounding residual.
            self.assertAlmostEqual(budget["cumulative_requested"], float(initial_capacity.double().sum()), delta=1e-5)
            self.assertAlmostEqual(budget["cumulative_delivered"], budget["cumulative_requested"], places=4)
            self.assertEqual(budget["cumulative_clipped"], 0)

    def test_shift_occurs_before_injection_at_exact_completed_update(self):
        env = TrailEnvironment(shift_interval=2, shift_cells=3)
        sub = substrate()
        env.seed(sub)
        original = env._capacity.clone()
        env.step(sub)
        self.assertTrue(torch.equal(env._capacity, (original - env.inject_rate).clamp(min=0)))
        remaining = env._capacity.clone()
        env.step(sub)
        expected = torch.roll(remaining, 3, dims=1)
        self.assertTrue(torch.equal(env.resource_state()["last_injection"], expected.clamp(max=env.inject_rate)))
        self.assertTrue(torch.equal(env._capacity, (expected - env.inject_rate).clamp(min=0)))

    def test_saturated_substrate_records_delivered_not_just_requested_energy(self):
        env, sub = TrailEnvironment(), substrate()
        env.seed(sub)
        sub.mem[0, 0].fill_(100)
        env.step(sub)
        budget = env.resource_state()
        self.assertEqual(budget["cumulative_delivered"], 0)
        self.assertGreater(budget["cumulative_requested"], 0)
        self.assertAlmostEqual(budget["cumulative_clipped"], budget["cumulative_requested"], places=8)

    def test_saved_pair_differs_only_in_shift_interval_and_defaults_are_stationary(self):
        configs = [json.loads((ROOT / f"configs/dynamic_energy_demo/{n}.json").read_text()) for n in ("fixed", "dynamic")]
        self.assertEqual(TrailEnvironment().shift_interval, 0)
        self.assertEqual(TrailEnvironment().inject_rate, 0.02)
        self.assertEqual(TrailEnvironment().trail_width, 4)
        self.assertEqual(TrailEnvironment().spawn, "spread")
        configs[0]["env_options"]["shift_interval"] = 25
        self.assertEqual(*configs)
        for config in configs:
            env = make_env(config["env"], config["env_param"], config["env_options"])
            self.assertEqual(env.to_dict()["spawn"], "spread")
            self.assertEqual(env.to_dict()["shift_interval"], 25)
        for options in ({"inject_rate": -1}, {"max_capacity": float("nan")},
                        {"shift_interval": 1.5}, {"shift_cells": True},
                        {"spawn": "typo"}, {"speed": 1}):
            with self.assertRaises(ValueError):
                make_env("trail", options=options)
        with self.assertRaises(ValueError):
            make_env("flat", options={"shift_interval": 2})

    def test_headless_trail_initialization_and_frame_timing(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = json.loads((ROOT / "configs/dynamic_energy_demo/dynamic.json").read_text())
            config.update(steps=3, shape=16, screenshot_interval=1, max_screenshots=4)
            config["env_options"].update(spawn="corner", shift_interval=2)
            path = folder / "config.json"
            path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(EXPERIMENT / "headless_coral.py"),
                                     "--config", str(path), "--output-dir", str(folder / "run")],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            initial = torch.load(folder / "run/initial_state.pt", weights_only=True)
            self.assertLess(int((initial[0, 13] >= 0).sum()), 40)
            self.assertLess(float(initial[0, 0].max()), 0.401)
            for update in range(4):
                frame = torch.load(folder / f"run/frame_states/update_{update:07d}.pt", weights_only=True)
                resource = torch.load(folder / f"run/frame_states/resource_{update:07d}.pt", weights_only=True)
                self.assertEqual(resource["completed_update"], update)
                self.assertTrue(torch.isfinite(frame).all())
                with Image.open(folder / f"run/screenshots/step_{update:07d}.png") as image:
                    self.assertEqual(image.size, (16, 16))
            final = torch.load(folder / "run/checkpoint_0000003/substrate.pt", weights_only=True)
            self.assertTrue(torch.equal(frame, final))


if __name__ == "__main__":
    unittest.main()
