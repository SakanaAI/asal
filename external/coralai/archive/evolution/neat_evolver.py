from datetime import datetime
import os
import numpy as np
import torch
import neat
import taichi as ti
import torch.nn as nn

from ..substrate.nn_lib import ch_norm

from coralai.instances.coral.coral_physics import invest_liquidate, explore_physics, energy_physics, activate_outputs

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import relu_activation, sigmoid_activation, tanh_activation, identity_activation
from pytorch_neat.linear_net import LinearNet
from .neat_organism import NeatOrganism
from ..substrate.nn_lib import ch_norm

@ti.data_oriented
class NEATEvolver():
    def __init__(self, config_path, substrate, kernel, ind_of_middle, sense_chs, act_chs):
        self.substrate = substrate
        torch_device = substrate.torch_device
        self.substrate = substrate
        self.kernel = kernel
        self.torch_device = torch_device
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)

        self.ind_of_middle = ind_of_middle

        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        self.timestep = 0
        self.out_mem = None
        self.energy_offset = 0.0
        self.organisms = None


    def gen_population(self):
        self.population = neat.Population(self.neat_config)

        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())

        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        self.checkpoint_dir = f'history/NEAT_{current_datetime}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_prefix_full = os.path.join(self.checkpoint_dir, f"checkpoint")
        self.population.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix_full))


    def eval_genomes(self, genomes, n_timesteps, vis=None):
        inds = self.substrate.ti_indices[None]
        
        self.substrate.mem[0, inds.energy,...] = 1.0
        self.substrate.mem[0, inds.infra,...] = 1.0
        organisms = []
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            organisms.append({"net": self.create_torch_net(genome), "genome": genome})
        self.organisms = organisms
        self.substrate.mem[0, inds.genome] = torch.where(
            torch.rand_like(self.substrate.mem[0, inds.genome]) > 0.8,
            torch.randint_like(self.substrate.mem[0, inds.genome], 0, len(organisms)),
            -1
        )
        for i in range(len(organisms)):
            org = organisms[i]
            org['genome'].fitness = -(self.get_genome_infra_sum(i)).item()
            # org['genome'].fitness = -self.substrate.mem[0, inds.genome].eq(i).sum().item()

        combined_weights = torch.zeros(
            (len(organisms), 1, self.n_acts, self.n_senses * self.kernel.shape[0]), device=self.torch_device)
        combined_biases = torch.zeros(
            (len(organisms), 1, self.n_acts, 1), device=self.torch_device)
        
        for i in range(len(organisms)):
            combined_weights[i, 0] = organisms[i]["net"].weights
            combined_biases[i, 0] = organisms[i]["net"].biases

        rand_time_offset = torch.randint(0, 100, (1,)).item()

        for timestep in range(n_timesteps):
            self.step_sim(combined_weights, combined_biases)
            self.timestep = timestep
            if vis is not None:
                vis.update()
                if vis.next_generation:
                    vis.next_generation = False
                    break

        for i in range(len(organisms)):
            org = organisms[i]
            # org['genome'].fitness += self.substrate.mem[0, inds.genome].eq(i).sum().item()
            org['genome'].fitness += (self.get_genome_infra_sum(i)).item()

    
    def step_sim(self, combined_weights, combined_biases):
        inds = self.substrate.ti_indices[None]
        self.forward(combined_weights, combined_biases)
        self.energy_offset = self.get_energy_offset(self.timestep)
        self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + self.energy_offset) * 0.1
        self.substrate.mem[0, inds.infra] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + self.energy_offset) * 0.1
        self.substrate.mem[0, inds.energy] = torch.clamp(self.substrate.mem[0, inds.energy], 0.01, 100)
        self.substrate.mem[0, inds.infra] = torch.clamp(self.substrate.mem[0, inds.infra], 0.01, 100)
        if self.timestep % 20 == 0:
            self.kill_random_chunk(5)
        self.apply_physics()
    

    def apply_physics(self):
        inds = self.substrate.ti_indices[None]
        # self.substrate.mem[0, inds.energy, self.substrate.w//2, self.substrate.h//2] += 10
        activate_outputs(self.substrate, self.ind_of_middle)
        invest_liquidate(self.substrate)
        explore_physics(self.substrate, self.kernel)
        energy_physics(self.substrate, self.kernel, max_infra=10, max_energy=1.5)

        self.substrate.mem[0, inds.genome] = torch.where(
            (self.substrate.mem[0, inds.infra] + self.substrate.mem[0, inds.energy]) > 0.05,
            self.substrate.mem[0, inds.genome],
            -1
        )


    def kill_random_chunk(self, width):
        inds = self.substrate.ti_indices[None]
        x = np.random.randint(0, self.substrate.w)
        y = np.random.randint(0, self.substrate.h)
        for i in range(x-width, x+width):
            for j in range(y-width, y+width):
                self.substrate.mem[0, inds.genome, i%self.substrate.w, j%self.substrate.h] = -1


    def get_energy_offset(self, timestep, repeat_steps=50, amplitude=1, positive_scale=1, negative_scale=1):
        frequency = (2 * np.pi) / repeat_steps
        value = amplitude * np.sin(frequency * timestep)
        if value > 0:
            return value * positive_scale
        else:
            return value * negative_scale
    
    def forward(self, weights, biases):
        inds = self.substrate.ti_indices[None]
        out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
        self.apply_weights_and_biases(
            self.substrate.mem, out_mem,
            self.kernel, self.sense_chinds,
            weights, biases,
            inds.genome)
        self.substrate.mem[0, self.act_chinds] = out_mem
        
    @ti.kernel
    def apply_weights_and_biases(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                                      kernel: ti.types.ndarray(), sense_chinds: ti.types.ndarray(),
                                      combined_weights: ti.types.ndarray(), combined_biases: ti.types.ndarray(),
                                      genome_ind: ti.i32):
        for i, j, act_k in ti.ndrange(mem.shape[2], mem.shape[3], out_mem.shape[0]):
            val = 0.0
            genome_key = int(mem[0, genome_ind, i, j])
            for sensor_n, neigh_m in ti.ndrange(sense_chinds.shape[0], kernel.shape[0]):
                neigh_x = (i + kernel[neigh_m, 0]) % mem.shape[2]
                neigh_y = (j + kernel[neigh_m, 1]) % mem.shape[3]
                val += (mem[0, sense_chinds[sensor_n], neigh_x, neigh_y] *
                        combined_weights[genome_key, 0, act_k, sensor_n])
            out_mem[act_k, i, j] = val + combined_biases[genome_key, 0, act_k, 0]


    def create_torch_net(self, genome):
        input_coords = []
        for offset in self.kernel:
            for ch in range(self.n_senses):
                input_coords.append([offset[0], offset[1], self.sense_chinds[ch]])

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
