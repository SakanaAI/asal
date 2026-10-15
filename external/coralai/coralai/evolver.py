import copy
from datetime import datetime
import os
import random
import numpy as np
from neat.reporting import ReporterSet
from neat.reporting import BaseReporter

import torch
import neat
from neat.six_util import iteritems, itervalues

import taichi as ti
import torch.nn as nn


# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import identity_activation
from pytorch_neat.linear_net import LinearNet
from .nn_lib import ch_norm


@ti.kernel
def apply_weights_and_biases(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                             sense_chinds: ti.types.ndarray(),
                             combined_weights: ti.types.ndarray(), combined_biases: ti.types.ndarray(),
                             dir_kernel: ti.types.ndarray(), dir_order: ti.types.ndarray(),
                             ti_inds: ti.template(),
                             key_to_local: ti.types.ndarray()):
    """Forward pass: each cell looks up its genome key → local weight row."""
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        key = int(mem[0, inds.genome, i, j])
        if key < 0 or key >= key_to_local.shape[0]:
            for act_k in range(out_mem.shape[0]):
                out_mem[act_k, i, j] = 0.0
        else:
            local_idx = int(key_to_local[key])
            if local_idx < 0 or local_idx >= combined_weights.shape[0]:
                for act_k in range(out_mem.shape[0]):
                    out_mem[act_k, i, j] = 0.0
            else:
                rot = int(mem[0, inds.rot, i, j])
                n_dirs = dir_kernel.shape[0]
                for act_k in range(out_mem.shape[0]):
                    val = 0.0
                    for sense_ch_n in ti.ndrange(sense_chinds.shape[0]):
                        start_weight_ind = sense_ch_n * (n_dirs + 1)
                        val += (mem[0, sense_chinds[sense_ch_n], i, j] *
                                combined_weights[local_idx, 0, act_k, start_weight_ind])
                        for offset_m in ti.ndrange(n_dirs):
                            ind = (rot + int(dir_order[offset_m]) + n_dirs) % n_dirs
                            neigh_x = (i + dir_kernel[ind, 0]) % mem.shape[2]
                            neigh_y = (j + dir_kernel[ind, 1]) % mem.shape[3]
                            weight_ind = start_weight_ind + offset_m
                            val += (mem[0, sense_chinds[sense_ch_n], neigh_x, neigh_y] *
                                    combined_weights[local_idx, 0, act_k, weight_ind])
                    out_mem[act_k, i, j] = val + combined_biases[local_idx, 0, act_k, 0]


@ti.data_oriented
class SpaceEvolver():
    def __init__(self, config_path, substrate, kernel, dir_order, sense_chs, act_chs):
        torch_device = substrate.torch_device
        self.torch_device = torch_device
        self.substrate = substrate
        self.reporters = ReporterSet()
        self.substrate = substrate
        self.kernel = torch.tensor(kernel, device=torch_device)
        self.dir_kernel = self.kernel[1:] # for directional operations
        # Represents the order in which the kernel should be applied
        # in this case, in alternating closest vectors from a direction
        self.dir_order = torch.tensor(dir_order, device=torch_device)
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)


        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        self.timestep = 0
        self.out_mem = None
        self.energy_offset = 0.0

        self.genomes = []
        self.genome_keys = []          # stable integer key per genome (never remapped)
        self._next_key = 0             # monotonically increasing key counter
        self.ages = []
        self.combined_weights = []
        self.combined_biases = []
        self._cached_cw = None
        self._cached_cb = None
        self._weights_dirty = True
        self._key_to_local_cache = None
        self._key_map_dirty = True
        self._scratch = {}
        self.init_population()
        self.init_substrate(self.genomes)
        self.time_last_cull = 0
    

    def get_combined_weights(self):
        if self._weights_dirty or self._cached_cw is None:
            self._cached_cw = torch.stack(self.combined_weights, dim=0)
            self._cached_cb = torch.stack(self.combined_biases, dim=0)
            self._weights_dirty = False
        return self._cached_cw, self._cached_cb

    def _invalidate_weight_cache(self):
        self._weights_dirty = True

    def _get_scratch(self, key, like_tensor):
        if key not in self._scratch or self._scratch[key].shape != like_tensor.shape:
            self._scratch[key] = torch.zeros_like(like_tensor)
        else:
            self._scratch[key].zero_()
        return self._scratch[key]

    def _get_key_to_local(self) -> torch.Tensor:
        """Return int32 tensor (max_key+1,): stable_key → local_idx. -1 = dead."""
        if not self._key_map_dirty and self._key_to_local_cache is not None:
            return self._key_to_local_cache
        if not self.genome_keys:
            self._key_to_local_cache = torch.zeros(1, dtype=torch.int32,
                                                    device=self.torch_device)
            self._key_map_dirty = False
            return self._key_to_local_cache
        max_key = max(self.genome_keys)
        kt = torch.full((max_key + 1,), -1, dtype=torch.int32, device=self.torch_device)
        for local_idx, key in enumerate(self.genome_keys):
            kt[key] = local_idx
        self._key_to_local_cache = kt
        self._key_map_dirty = False
        return kt

    def local_idx_for_key(self, key: int) -> int:
        """Return local index for a stable genome key, -1 if not present."""
        for i, k in enumerate(self.genome_keys):
            if k == key:
                return i
        return -1

    def run(self, n_timesteps, vis, n_rad_spots, radiate_interval, cull_max_pop, cull_interval=100):
        timestep = 0
        while timestep < n_timesteps and vis.window.running:
            combined_weights, combined_biases = self.get_combined_weights()
            self.step_sim(combined_weights, combined_biases)
            vis.update()
            if timestep % radiate_interval == 0:
                self.apply_radiation_mutation(n_rad_spots)
                print("RADIATING")
            if vis.next_generation:
                vis.next_generation = False
                break
            if len(self.genomes) > cull_max_pop and (self.timestep - self.time_last_cull) > cull_interval:
                self.reduce_population_to_threshold(cull_max_pop)
            timestep += 1
            self.timestep = timestep

    
    def step_sim(self, combined_weights, combined_biases):
        inds = self.substrate.ti_indices[None]
        self.forward(combined_weights, combined_biases)
        self.energy_offset = self.get_energy_offset(self.timestep)
        self.ages = [age + 1 for age in self.ages]
        offset = self.energy_offset
        energy_ch = inds.energy
        infra_ch = inds.infra
        self.substrate.mem[0, energy_ch].add_(
            torch.randn_like(self.substrate.mem[0, energy_ch]).add_(offset).mul_(0.1))
        self.substrate.mem[0, infra_ch].add_(
            torch.randn_like(self.substrate.mem[0, infra_ch]).add_(offset).mul_(0.1))
        self.substrate.mem[0, energy_ch].clamp_(0.01, 100)
        self.substrate.mem[0, infra_ch].clamp_(0.01, 100)
        if self.timestep % 50 == 0:
            self.kill_random_chunk(5)
    

    def forward(self, weights, biases):
        inds = self.substrate.ti_indices[None]
        out_mem = self._get_scratch("out_mem",
                                    self.substrate.mem[0, self.act_chinds])
        apply_weights_and_biases(
            self.substrate.mem, out_mem,
            self.sense_chinds,
            weights, biases,
            self.dir_kernel, self.dir_order,
            self.substrate.ti_indices, self._get_key_to_local())
        self.substrate.mem[0, self.act_chinds] = out_mem
        self.apply_physics()
    


    def apply_physics(self):
        # Physics is experiment-specific. When using evolver.run(), override this
        # method in a subclass or patch it. The recommended approach is to manage
        # the step loop in your runner (see experiments/coral/run.py).
        raise NotImplementedError(
            "Override apply_physics() or manage the step loop in your runner.")
        inds = self.substrate.ti_indices[None]
        activate_outputs(self.substrate)
        invest_liquidate(self.substrate)
        explore_physics(self.substrate, self.kernel, self.dir_order)
        energy_physics(self.substrate, self.kernel, max_infra=10, max_energy=1.5)

        alive = (self.substrate.mem[0, inds.infra]
                 + self.substrate.mem[0, inds.energy]) > 0.05
        self.substrate.mem[0, inds.genome].masked_fill_(~alive, -1)


    def produce_alternating_order(self, len):
        order = []
        ind = 0
        i = 0
        while i < len:
            order.append(ind)
            if ind > 0:
                ind = -ind
            else:
                ind = (-ind + 1)
            i += 1
        return torch.tensor(order, device = self.torch_device)
    

    @ti.kernel
    def replace_genomes(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                        genome_transitions: ti.types.ndarray(), ti_indices: ti.template()):
        inds = ti_indices[None]
        for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
            if mem[0, inds.genome, i, j] < 0:
                out_mem[i, j] = mem[0, inds.genome, i, j]
            else:
                out_mem[i, j] = genome_transitions[int(mem[0, inds.genome, i, j])]

    def remove_extinct_genomes(self):
        """Remap and drop genomes whose cell count has reached zero.

        Never kills a genome that still holds any cells.  Returns the number
        of genomes removed.  Call this periodically instead of
        reduce_population_to_threshold to let natural selection regulate
        population size without destroying live organisms.
        """
        inds = self.substrate.ti_indices[None]
        genome_ch = self.substrate.mem[0, inds.genome]

        alive_cells = genome_ch[genome_ch >= 0]
        if alive_cells.numel() == 0:
            return 0

        alive_ids = set(alive_cells.long().unique().tolist())
        n_total = len(self.genomes)
        n_extinct = sum(1 for i in range(n_total) if i not in alive_ids)
        if n_extinct == 0:
            return 0

        new_genomes, new_ages, new_cw, new_cb = [], [], [], []
        genome_transitions = [-1] * n_total

        for old_idx in range(n_total):
            if old_idx in alive_ids:
                new_idx = len(new_genomes)
                new_genomes.append(self.genomes[old_idx])
                new_ages.append(self.ages[old_idx])
                new_cw.append(self.combined_weights[old_idx])
                new_cb.append(self.combined_biases[old_idx])
                genome_transitions[old_idx] = new_idx
            # extinct entries stay -1

        transitions_t = torch.tensor(genome_transitions, dtype=torch.float32,
                                     device=self.torch_device)
        out_mem = torch.zeros_like(genome_ch)
        self.replace_genomes(self.substrate.mem, out_mem, transitions_t,
                             self.substrate.ti_indices)
        self.substrate.mem[0, inds.genome] = out_mem

        self.genomes          = new_genomes
        self.ages             = new_ages
        self.combined_weights = new_cw
        self.combined_biases  = new_cb
        self._invalidate_weight_cache()
        self.time_last_cull = self.timestep
        print(f"  [cleanup] {n_extinct} extinct genomes removed -> {len(self.genomes)} remain")
        return n_extinct

    def reduce_population_to_threshold(self, max_population):
        print(f"REDUCING pop to max of {max_population} from current size: {len(self.genomes)}")
        if len(self.genomes) <= max_population:
            print("Population within threshold. No reduction needed.")
            return

        inds = self.substrate.ti_indices[None]
        genome_cell_counts = [
            (i, self.substrate.mem[0, inds.genome].eq(i).sum().item())
            for i in range(len(self.genomes))
        ]
        sorted_by_cells = sorted(genome_cell_counts, key=lambda x: x[1], reverse=True)

        new_genomes = []
        new_ages = []
        new_combined_weights = []
        new_combined_biases = []
        genome_transitions = [-1] * len(self.genomes)  # default: kill

        for rank, (old_idx, cell_count) in enumerate(sorted_by_cells):
            if rank >= max_population:
                # genome_transitions[old_idx] stays -1 → cells marked dead
                print(f"KILLING genome {old_idx} ({cell_count} cells)")
            else:
                # Assign 0-based new index BEFORE appending
                new_idx = len(new_genomes)
                new_genomes.append(self.genomes[old_idx])
                new_ages.append(self.ages[old_idx])
                new_combined_weights.append(self.combined_weights[old_idx])
                new_combined_biases.append(self.combined_biases[old_idx])
                genome_transitions[old_idx] = new_idx

        # float32 matches the genome channel dtype
        transitions_t = torch.tensor(genome_transitions, dtype=torch.float32,
                                     device=self.torch_device)
        out_mem = torch.zeros_like(self.substrate.mem[0, inds.genome])
        self.replace_genomes(self.substrate.mem, out_mem, transitions_t,
                             self.substrate.ti_indices)
        self.substrate.mem[0, inds.genome] = out_mem

        self.genomes = new_genomes
        self.ages = new_ages
        self.combined_weights = new_combined_weights
        self.combined_biases = new_combined_biases
        self._invalidate_weight_cache()
        self.time_last_cull = self.timestep
        print(f"\tPop size after reduction: {len(self.genomes)}")
        if len(self.genomes) == 0:
            print("NO GENOMES LEFT. REINITIALIZING")
            self.init_population()
            self.init_substrate(self.genomes)


    def save_checkpoint(self, run_dir, step):
        """Save substrate + population to run_dir/checkpoint_NNNNNNN/.

        Files written:
          substrate.pt     — substrate.mem tensor (CPU, device-agnostic)
          population.pkl   — genomes, ages, weights, biases, timestep,
                             energy_offset, time_last_cull, rng_state
          meta.json        — step, timestamp, git hash (advisory), grid shape
        """
        import json
        import pickle
        import subprocess

        ckpt_dir = os.path.join(run_dir, f"checkpoint_{step:07d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(self.substrate.mem.cpu(),
                   os.path.join(ckpt_dir, "substrate.pt"))

        # Capture all three RNG states so the run is reproducible from here
        rng_state = {
            "torch":  torch.get_rng_state(),
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        }
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") \
                and torch.backends.mps.is_available():
            rng_state["mps"] = torch.mps.get_rng_state()
        elif torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()

        pop_data = {
            "genomes":        self.genomes,
            "ages":           self.ages,
            "weights":        [w.cpu() for w in self.combined_weights],
            "biases":         [b.cpu() for b in self.combined_biases],
            "timestep":       self.timestep,
            "energy_offset":  self.energy_offset,
            "time_last_cull": self.time_last_cull,
            "rng_state":      rng_state,
        }
        with open(os.path.join(ckpt_dir, "population.pkl"), "wb") as f:
            pickle.dump(pop_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = "unknown"

        meta = {
            "step":      step,
            "timestamp": datetime.now().isoformat(),
            "git_hash":  git_hash,
            "shape":     [self.substrate.w, self.substrate.h],
            "n_genomes": len(self.genomes),
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [checkpoint] step {step:,} → {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir):
        """Restore substrate + evolver state from a checkpoint directory.

        After calling this the step loop can resume with the same RNG state
        as when the checkpoint was saved, producing a reproducible trajectory.
        """
        import pickle as _pickle

        # 1. Restore substrate memory
        saved_mem = torch.load(
            os.path.join(ckpt_dir, "substrate.pt"), map_location="cpu"
        )
        self.substrate.mem.copy_(saved_mem.to(self.substrate.torch_device))

        # 2. Restore population and scalar evolver state
        with open(os.path.join(ckpt_dir, "population.pkl"), "rb") as f:
            pop = _pickle.load(f)

        self.genomes         = pop["genomes"]
        self.ages            = pop["ages"]
        self.combined_weights = [w.to(self.substrate.torch_device) for w in pop["weights"]]
        self.combined_biases  = [b.to(self.substrate.torch_device) for b in pop["biases"]]
        self.timestep        = pop["timestep"]
        self.energy_offset   = pop.get("energy_offset", 0.0)
        self.time_last_cull  = pop.get("time_last_cull", 0)

        # Invalidate weight cache so next get_combined_weights() rebuilds it
        self._weights_dirty = True
        self._cached_cw = None
        self._cached_cb = None

        # 3. Restore RNG state (graceful no-op if checkpoint predates this feature)
        rng = pop.get("rng_state", {})
        if "torch"  in rng:
            torch.set_rng_state(rng["torch"])
        if "numpy"  in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        if "mps" in rng and hasattr(torch, "mps"):
            torch.mps.set_rng_state(rng["mps"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])

        print(f"  [loaded] step {self.timestep:,} ← {ckpt_dir}")
    
    def report_if_necessary(self, fitness_function, n=None):
        for i in range(len(self.genomes)):
            # org['genome'].fitness += self.substrate.mem[0, inds.genome].eq(i).sum().item()
            self.genomes[i].fitness = fitness_function(self.genomes[i], i)
        # self.reporters.start_generation(self.generation)

        # # Evaluate all genomes using the user-provided function.
        # fitness_function(list(iteritems(self.population)), self.config)

        # # Gather and report statistics.
        # best = None
        # for g in itervalues(self.population):
        #     if best is None or g.fitness > best.fitness:
        #         best = g

        # self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # # # Create the next generation from the current generation.
        # # self.population = self.reproduction.reproduce(self.config, self.species,
        # #                                               self.config.pop_size, self.generation)

        # # Check for complete extinction.
        # if not self.species.species:
        #     self.reporters.complete_extinction()
        #     # TODO: re-seed env

        # # Divide the new population into species.
        # self.species.speciate(self.config, self.population, self.generation)

        # self.reporters.end_generation(self.config, self.population, self.species)

        # self.generation += 1

        # if self.config.no_fitness_termination:
        #     self.reporters.found_solution(self.config, self.generation, self.best_genome)

        # return self.best_genome
    
    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)    

    def init_population(self):
        genomes = []
        for i in range(self.neat_config.pop_size):
            genome = neat.DefaultGenome(str(i))
            genome.configure_new(self.neat_config.genome_config)
            self.add_organism_get_key(genome)

        self.add_reporter(neat.StdOutReporter(True))
        self.add_reporter(neat.StatisticsReporter())

        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        self.checkpoint_dir = f'history/NEAT_{current_datetime}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_prefix_full = os.path.join(self.checkpoint_dir, f"checkpoint")
        self.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix_full))

        return genomes


    def init_substrate(self, genomes):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome] = torch.where(
            torch.rand_like(self.substrate.mem[0, inds.genome]) > 0.8,
            torch.randint_like(self.substrate.mem[0, inds.genome], 0, len(genomes)),
            -1
        )
        self.substrate.mem[0, inds.energy, ...] = 1.0
        self.substrate.mem[0, inds.infra, ...] = 1.0
        self.substrate.mem[0, inds.rot] = torch.randint_like(self.substrate.mem[0, inds.rot], 0, self.dir_kernel.shape[0])


    def add_organism_get_key(self, genome):
        self.genomes.append(genome)
        net = self.create_torch_net(genome)
        self.combined_weights.append(net.weights)
        self.combined_biases.append(net.biases)
        self.ages.append(0)
        self._invalidate_weight_cache()
        return len(self.combined_biases) - 1
    

    def set_chunk(self, genome_key, x, y, radius):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome, x%self.substrate.w, y%self.substrate.h] = genome_key
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                self.substrate.mem[0, inds.genome, i%self.substrate.w, j%self.substrate.h] = genome_key


    def create_torch_net(self, genome):
        input_coords = []
        # TODO: adjust for direcitonal kernel
        for ch in range(self.n_senses):
             input_coords.append([0, 0, self.sense_chinds[ch]])
             for offset_i in range(self.dir_order.shape[0]):
                offset_x = self.dir_kernel[self.dir_order[offset_i], 0]
                offset_y = self.dir_kernel[self.dir_order[offset_i], 1]
                input_coords.append([offset_x, offset_y, self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])

        net = LinearNet.create(
            genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.0,
            weight_max=3.0,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.torch_device,
        )
        return net
    

    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum

    # def cull_genomes(self, n_cells_thresh, age_thresh):
    #     print(f"CULLING pop of size: {len(self.genomes)}")
    #     inds = self.substrate.ti_indices[None]
    #     new_genomes = []
    #     new_ages = []
    #     new_combined_weights = []
    #     new_combined_biases = []
    #     for i in range(len(self.genomes)):
    #         where_i = self.substrate.mem[0, inds.genome].eq(i)
    #         n_cells = where_i.sum().item()
    #         print(f"\tGenome {i} is {self.ages[i]} steps old and has {n_cells} cells")
    #         if n_cells == 0 or (self.ages[i] > age_thresh and n_cells < n_cells_thresh):
    #             print(f"\t\tCulling genome {i}")
    #             self.substrate.mem[0, inds.genome] = torch.where(where_i, -1, self.substrate.mem[0, inds.genome])
    #         else:
    #             self.substrate.mem[0, inds.genome]= torch.where(where_i, len(new_genomes), self.substrate.mem[0, inds.genome])
    #             new_genomes.append(self.genomes[i])
    #             new_ages.append(self.ages[i])
    #             new_combined_weights.append(self.combined_weights[i])
    #             new_combined_biases.append(self.combined_biases[i])
    #     self.genomes = new_genomes
    #     self.ages = new_ages
    #     self.combined_weights = new_combined_weights
    #     self.combined_biases = new_combined_biases
    #     print(f"\tPop size after culling: {len(self.genomes)}")
    #     if len(self.genomes) == 0:
    #         print("Population extinct")
    #         self.init_population()
    #         self.init_substrate(self.genomes)
    #         self.timestep = 0