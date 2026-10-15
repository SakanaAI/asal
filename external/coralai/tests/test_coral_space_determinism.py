"""
Deterministic correctness tests for coral_runner_space.

NOTE: Taichi's CPU backend has a known issue where creating multiple
evolvers in the same process causes NaN from stale kernel template
caching. All step-simulation tests use a SINGLE evolver instance.
This does not affect Metal/CUDA backends or real usage (one evolver
per run).
"""

import os
import random
import numpy as np
import pytest
import torch
import taichi as ti
import neat

ti.init(ti.cpu)

from coralai.substrate import Substrate
from coralai.evolver import SpaceEvolver
from experiments.coral.physics import (
    activate_outputs, invest_liquidate, explore_physics, energy_physics,
    apply_weights_and_biases,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_evolver(shape=(20, 20), seed=42):
    seed_everything(seed)
    channels = {
        "energy": ti.f32,
        "infra": ti.f32,
        "acts": ti.types.struct(
            invest=ti.f32,
            liquidate=ti.f32,
            explore=ti.types.vector(n=4, dtype=ti.f32),
        ),
        "com": ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
        "rot": ti.f32,
        "genome": ti.f32,
    }
    device = torch.device("cpu")
    substrate = Substrate(shape, torch.float32, device, channels)
    substrate.malloc()

    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(local_dir,
                               "experiments/coral/neat.config")
    kernel = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
              [-1, 0], [-1, -1], [0, -1], [1, -1]]
    dir_order = [0, -1, 1]

    evolver = SpaceEvolver(config_path, substrate, kernel, dir_order,
                           ["energy", "infra", "com"], ["acts", "com"])
    return evolver, substrate


def get_state_fingerprint(substrate):
    inds = substrate.ti_indices[None]
    return {
        "mem_sum": float(substrate.mem.sum()),
        "energy_sum": float(substrate.mem[0, inds.energy].sum()),
        "infra_sum": float(substrate.mem[0, inds.infra].sum()),
        "genome_sum": float(substrate.mem[0, inds.genome].sum()),
    }


# ── Module-scoped evolver for step tests ────────────────────────────
# Only ONE evolver created to avoid Taichi CPU template caching issue.

_shared_ev = None
_shared_sub = None


def get_shared_evolver():
    global _shared_ev, _shared_sub
    if _shared_ev is None:
        _shared_ev, _shared_sub = make_evolver(seed=42)
    return _shared_ev, _shared_sub


class TestStepSimulation:
    """Tests using a single shared evolver instance."""

    def test_single_step_no_nan(self):
        ev, sub = get_shared_evolver()
        seed_everything(42)
        cw = torch.stack(ev.combined_weights, dim=0)
        cb = torch.stack(ev.combined_biases, dim=0)
        ev.step_sim(cw, cb)
        assert not torch.isnan(sub.mem).any(), \
            f"NaN after step: {torch.isnan(sub.mem).sum()} values"

    def test_multi_step_no_nan(self):
        ev, sub = get_shared_evolver()
        for step in range(20):
            seed_everything(100 + step)
            cw = torch.stack(ev.combined_weights, dim=0)
            cb = torch.stack(ev.combined_biases, dim=0)
            ev.step_sim(cw, cb)
            ev.timestep += 1
        nan_count = torch.isnan(sub.mem).sum().item()
        assert nan_count == 0, \
            f"NaN after 20 steps: {nan_count} values"

    def test_fingerprint_finite(self):
        ev, sub = get_shared_evolver()
        fp = get_state_fingerprint(sub)
        for k, v in fp.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


class TestPhysics:

    def test_invest_liquidate_conservation(self):
        ev, sub = get_shared_evolver()
        inds = sub.ti_indices[None]
        sub.mem[0, inds.energy] = torch.rand_like(sub.mem[0, inds.energy]) + 0.1
        sub.mem[0, inds.infra] = torch.rand_like(sub.mem[0, inds.infra]) + 0.1
        sub.mem[0, inds.acts_invest] = 0.3
        sub.mem[0, inds.acts_liquidate] = 0.2
        total_before = (sub.mem[0, inds.energy].sum()
                        + sub.mem[0, inds.infra].sum()).item()
        invest_liquidate(sub)
        total_after = (sub.mem[0, inds.energy].sum()
                       + sub.mem[0, inds.infra].sum()).item()
        assert total_before == pytest.approx(total_after, rel=1e-4)

    def test_activate_outputs_no_nan(self):
        ev, sub = get_shared_evolver()
        inds = sub.ti_indices[None]
        sub.mem[0, inds.genome] = 0
        sub.mem[0, inds.com] = torch.randn_like(sub.mem[0, inds.com])
        sub.mem[0, inds.acts] = torch.randn_like(sub.mem[0, inds.acts])
        activate_outputs(sub)
        assert not torch.isnan(sub.mem[0, inds.acts]).any()
        assert not torch.isnan(sub.mem[0, inds.com]).any()

    def test_dead_cells_zeroed(self):
        ev, sub = get_shared_evolver()
        inds = sub.ti_indices[None]
        sub.mem[0, inds.genome] = -1
        sub.mem[0, inds.acts] = torch.ones_like(sub.mem[0, inds.acts])
        activate_outputs(sub)
        assert (sub.mem[0, inds.acts] == 0).all()


class TestWeightCaching:

    def test_cache_invalidation(self):
        ev, sub = get_shared_evolver()
        cw1, cb1 = ev.get_combined_weights()
        assert not ev._weights_dirty
        new_genome = neat.DefaultGenome("test")
        new_genome.configure_new(ev.neat_config.genome_config)
        ev.add_organism_get_key(new_genome)
        assert ev._weights_dirty
        cw2, cb2 = ev.get_combined_weights()
        assert cw2.shape[0] == cw1.shape[0] + 1
