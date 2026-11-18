"""Event instrumentation and demo comparison checks, separate from controller regressions."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import taichi as ti
import torch

from scripts.verify_controller_lookup import ROOT, fixture

sys.path.insert(0, str(ROOT / "external/coralai/experiments/coral_dev"))
from coralai import evolver as module
from cell_events import CellEvents
from experiment import EXPERIMENT
from physics import activate_outputs, invest_liquidate, explore_physics


class ControllerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, debug=True)

    def test_controlled_colonization_and_bid_accounting(self):
        report = controlled_colonization()
        self.assertEqual(report["target_genome"], 1)
        self.assertAlmostEqual(report["parent_energy_after_claim"], 0)
        self.assertAlmostEqual(report["target_infrastructure_after_claim"], 0.9, places=6)
        self.assertAlmostEqual(report["energy_plus_infra_loss"], 0.1, places=6)
        self.assertEqual(report["local_colonization_gains"], 1)

    def test_event_observation_does_not_modify_state_or_rng(self):
        ev = fixture(module)
        state, rng = ev.substrate.mem.clone(), torch.get_rng_state().clone()
        events = CellEvents(ev.substrate); events.begin(1)
        for phase in events.phases: events.observe(phase)
        self.assertEqual(events.finish()["net_gains"], 0)
        self.assertTrue(torch.equal(state, ev.substrate.mem))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_pair_initial_state_comparison_uses_rng_values(self):
        from scripts.demo_controller_lookup_fix import controller_equality
        ev = fixture(module, n=50)
        ev.genomes = [SimpleNamespace(key=k) for k in range(50)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("fixed", "dynamic"):
                ev.save_checkpoint(root / name / "run", 0)
            self.assertTrue(controller_equality(root)["initial_rng_states_equal"])



def controlled_colonization():
    """Known controller, radiation disabled; a mechanism demonstration only."""
    ev = fixture(module, n=2)
    sub = EXPERIMENT.make_substrate((8, 8), torch.device("cpu"))
    ev.substrate = sub
    ev.sense_chinds = sub.windex[EXPERIMENT.sense_chs]
    ev.act_chinds = sub.windex[EXPERIMENT.act_chs]
    ev.kernel = torch.tensor(EXPERIMENT.kernel)
    ev.dir_kernel = ev.kernel[1:]
    ev.dir_order = torch.tensor(EXPERIMENT.dir_order)
    ev.combined_weights = [torch.zeros(1, 10, 24) for _ in range(2)]
    ev.combined_biases = [torch.zeros(1, 10, 1) for _ in range(2)]
    ev.combined_biases[1][0, 3, 0] = 10  # output row 3 -> forward exploration
    ev._invalidate_weight_cache()
    sub.mem.zero_(); sub.mem[0, 13] = -1
    parent, target = (3, 3), (4, 3)
    sub.mem[0, 13, *parent] = 1
    sub.mem[0, 12, *parent] = 1  # Actual existing explore kernel: direction (1, 0)
    sub.mem[0, 0, *parent] = sub.mem[0, 1, *parent] = 1
    before = sub.mem.clone()
    output = torch.zeros(10, 8, 8)
    module.apply_weights_and_biases(sub.mem, output, ev.sense_chinds, *ev.get_combined_weights(),
                                  ev.dir_kernel, ev.dir_order, sub.ti_indices, ev._get_key_to_local())
    sub.mem[0, ev.act_chinds] = output
    activate_outputs(sub); invest_liquidate(sub)
    events = CellEvents(sub); events.begin(1)
    explore_physics(sub, ev.kernel, ev.dir_order, defense_coeff=0)
    events.observe("colonization")
    after = sub.mem.clone()
    result = {"label": "Controlled mechanism demonstration, not evolved/spontaneous behavior",
              "radiation_or_reseeding_calls": 0, "parent": list(parent), "target": list(target),
              "parent_genome": int(after[0, 13, *parent]), "target_genome": int(after[0, 13, *target]),
              "parent_energy_after_claim": float(after[0, 0, *parent]),
              "target_infrastructure_after_claim": float(after[0, 1, *target]),
              "energy_plus_infra_loss": float(before[:, :2].sum() - after[:, :2].sum()),
              "local_colonization_gains": events.finish()["colonization_gains"]}
    return result



if __name__ == "__main__": unittest.main()
