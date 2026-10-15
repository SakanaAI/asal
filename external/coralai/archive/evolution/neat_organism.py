import os
import copy
import torch
import neat
import uuid
from datetime import datetime

import configparser
import torch.nn as nn
import taichi as ti

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.recurrent_net import RecurrentNet
from ..substrate.nn_lib import ch_norm
from .organism import Organism

@ti.data_oriented
class NeatOrganism(Organism):
    def __init__(self, config_path, substrate, kernel, sense_chs, act_chs, torch_device):
        super().__init__(substrate, kernel, sense_chs, act_chs, torch_device)
        self.config_path = config_path
        self.name = "evolvable_organism"
        self.neat_config = self.load_neat_config()

        self.genome = None
        self.genome_key = None
        self.net = None
        self.is_evolvable = True


    def load_neat_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        n_in = self.n_senses * len(self.kernel)
        n_out = self.n_acts
        genome_section = 'DefaultGenome'
        config.set(genome_section, 'num_inputs', f'{n_in}')
        config.set(genome_section, 'num_hidden', '0')
        config.set(genome_section, 'num_outputs', f'{n_out}')

        # Save the modified configuration in 'configs' folder with a specific name format
        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        config_dir = f'history/{self.name}'
        os.makedirs(config_dir, exist_ok=True)
        temp_config_path = os.path.join(config_dir, f'config_{current_datetime}.ini')
        with open(temp_config_path, 'w') as config_file:
            config.write(config_file)

        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           temp_config_path)


    def set_genome(self, genome_key, genome=None):
        self.genome_key = genome_key
        if genome is None:
            self.genome = self.gen_random_genome(genome_key)
        else:
            self.genome = genome


    def gen_random_genome(self, genome_key):
        genome = neat.DefaultGenome(str(genome_key))
        # Initialize the new genome with the configuration settings
        genome.configure_new(self.neat_config.genome_config)
        
        return genome


    def create_torch_net(self, batch_size=None):
        if batch_size is None:
            batch_size = self.substrate.w*self.substrate.h
        self.net = RecurrentNet.create(self.genome, self.neat_config,
                                       batch_size=batch_size, dtype=torch.float32,
                                       device=self.torch_device)


    def get_cell_coords(self, genome_map):
        matches = genome_map.eq(self.genome_key)
        coords = torch.where(matches)
        if coords[0].shape[0] != 0:
            pass
        combined_coords = torch.stack((coords[0], coords[1]), dim=1).contiguous()

        return combined_coords
    

    @ti.kernel
    def sense_to(self, mem: ti.types.ndarray(), sensor_mem: ti.types.ndarray(),
                 cell_coords: ti.types.ndarray(),
                 sense_chinds: ti.types.ndarray(), kernel: ti.types.ndarray()):
        for cell_n in ti.ndrange(cell_coords.shape[0]):
            x = cell_coords[cell_n, 0]
            y = cell_coords[cell_n, 1]
            sensor_num = 0
            for neigh_n in ti.ndrange(kernel.shape[0]):
                off_x = kernel[neigh_n, 0]
                off_y = kernel[neigh_n, 1]
                for s_ind in sense_chinds:
                    sensor_mem[cell_n, sensor_num] = mem[0, s_ind, (x + off_x) % mem.shape[2], (y+off_y) % mem.shape[3]]
                    sensor_num += 1


    @ti.kernel
    def store_actions(self, actions: ti.types.ndarray(), mem: ti.types.ndarray(), 
                      act_chinds: ti.types.ndarray(), cell_coords: ti.types.ndarray()):
        for cell_n, act_n in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
            mem[0, act_chinds[act_n], cell_coords[cell_n, 0], cell_coords[cell_n, 1]] = actions[cell_n, act_n]


    def activate(self, sensor_mem):
        return self.net.activate(sensor_mem)


    def forward(self, mem, genome_map):
      
        # mem += torch.randn_like(mem) * 0.1

        cell_coords = self.get_cell_coords(genome_map)
        n_cells = cell_coords.shape[0]
        sensor_mem = torch.zeros(n_cells,
                                 self.kernel.shape[0] * self.n_senses,
                                 dtype=torch.float32, device=self.torch_device)
        self.sense_to(mem, sensor_mem, cell_coords, self.sense_chinds, self.kernel)

        actions = self.activate(sensor_mem).contiguous()

        self.store_actions(actions, mem, self.act_chinds, cell_coords)
        # mem[:, self.act_chinds] = nn.ReLU()(mem[:, self.act_chinds])
        mem[:, self.act_chinds] = ch_norm(mem[:, self.act_chinds])
        mem[:, self.act_chinds] = torch.sigmoid(mem[:, self.act_chinds])
        
        return mem


    def mutate(self):
        cloned_genome = copy.deepcopy(self.genome)
        cloned_genome.mutate(self.neat_config.genome_config)
        return cloned_genome