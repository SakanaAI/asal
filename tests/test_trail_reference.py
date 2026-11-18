"""Check GUI compatibility, lossless capture, and the bounded reference treatment."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.compare_trail_reference import ROOT, RUNNER, audit_defaults, gui_values, headless_image, load_state, load_resource


class TrailReferenceTests(unittest.TestCase):
    def test_gui_export_matches_actual_gui_taichi_kernel_without_a_window(self):
        import taichi as ti
        from coralai.visualization import Visualization

        ti.init(arch=ti.cpu, cpu_max_num_threads=1)
        state = torch.zeros(1, 14, 8, 8)
        state[0, 0, 2:6] = 0.4
        state[0, 1] = 1
        state[0, 1, 2:6] = 1.4
        state[0, 12] = torch.arange(64).reshape(8, 8) % 8
        # Exercise the existing rendering kernel directly; never create a GUI.
        vis = Visualization.__new__(Visualization)
        vis.w = vis.h = 8
        vis.scale = 1
        vis.image = ti.Vector.field(3, dtype=ti.f32, shape=(8, 8))
        vis.write_to_renderer(state, torch.tensor([0.4, 1.4, 7]), torch.tensor([0, 1, 12]),
                              torch.ones(3, dtype=torch.int32), 0, 8)
        actual = vis.image.to_numpy().transpose(1, 0, 2)
        np.testing.assert_allclose(gui_values(state), actual, rtol=1e-6, atol=1e-7)

    def test_gui_and_headless_views_show_different_infrastructure_background(self):
        state = torch.zeros(1, 14, 8, 8)
        state[0, 1] = 1
        state[0, 1, 2:6] = 1.4
        state[0, 13] = -1
        self.assertAlmostEqual(float(gui_values(state)[0, 0, 1]), 1 / 1.4, places=6)
        self.assertEqual(int(np.asarray(headless_image(state))[0, 0, 1]), 0)

    def test_original_default_pair_only_changes_motion_and_has_no_resource_wrap(self):
        configs = {n: json.loads((ROOT / f"configs/trail_reference_comparison/{n}.json").read_text()) for n in ("fixed", "dynamic")}
        modified = copy.deepcopy(configs["fixed"])
        modified["env_options"]["shift_interval"] = 125
        self.assertEqual(modified, configs["dynamic"])
        with tempfile.TemporaryDirectory() as tmp:
            audit_defaults(configs, Path(tmp))
            reference = json.loads((Path(tmp) / "reference.json").read_text())
            self.assertEqual(reference["movement"]["initial_y_bounds"], [18, 382])
            self.assertEqual(reference["movement"]["maximum_y_bounds"], [34, 398])
            self.assertEqual(reference["movement"]["resource_wraps"], 0)

    def test_resource_step_only_changes_energy_and_reserve(self):
        sys.path.insert(0, str(RUNNER.parent))
        from environments import TrailEnvironment

        state = torch.arange(14 * 16 * 16, dtype=torch.float32).reshape(1, 14, 16, 16) / 1000
        sub = SimpleNamespace(mem=state, torch_device=torch.device("cpu"),
                              ti_indices={None: SimpleNamespace(energy=0, infra=1, genome=13)})
        env = TrailEnvironment(shift_interval=1, shift_cells=2)
        env.seed(sub)
        before = state.clone()
        expected_capacity = torch.roll(env._capacity, 2, dims=1)
        injection = expected_capacity.clamp(max=env.inject_rate)
        env.step(sub)
        self.assertTrue(torch.equal(state[:, 1:], before[:, 1:]))
        self.assertTrue(torch.equal(state[0, 0], before[0, 0] + injection))
        self.assertTrue(torch.equal(env._capacity, expected_capacity - injection))

    def test_lossless_npz_capture_matches_checkpoint_rgb_and_completed_update_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = json.loads((ROOT / "configs/trail_reference_comparison/dynamic.json").read_text())
            config.update(shape=16, steps=3, screenshot_interval=1, max_screenshots=4)
            config["env_options"]["shift_interval"] = 2
            path = folder / "config.json"
            path.write_text(json.dumps(config))
            completed = subprocess.run([sys.executable, str(RUNNER), "--config", str(path), "--output-dir", str(folder / "run")],
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            from PIL import Image
            run = folder / "run"
            with (run / "resource_metrics.csv").open() as stream: rows = list(csv.DictReader(stream))
            self.assertEqual([int(r["completed_update"]) for r in rows], [0, 1, 2, 3])
            for update, row in enumerate(rows):
                state, resource = load_state(run, update), load_resource(run, update)
                occupied = state[0, 13].numpy() >= 0
                inside = int((occupied & (resource["capacity"] > 0)).sum())
                self.assertEqual(int(row["occupied_inside_source"]), inside)
                self.assertEqual(int(row["occupied_cells"]), int(occupied.sum()))
                self.assertEqual(int(row["occupied_outside_source"]), int(occupied.sum()) - inside)
                with Image.open(run / "screenshots" / f"step_{update:07d}.png") as image:
                    np.testing.assert_array_equal(np.asarray(image), np.asarray(headless_image(state)))
            final = torch.load(run / "checkpoint_0000003/substrate.pt", weights_only=True)
            self.assertTrue(torch.equal(load_state(run, 3), final))


if __name__ == "__main__":
    unittest.main()
