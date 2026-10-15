import os
import torch
import neat
import taichi as ti
import configparser
from datetime import datetime

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import relu_activation, sigmoid_activation, tanh_activation
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.adaptive_net import AdaptiveNet
from ...evolution.neat_organism import NeatOrganism

@ti.data_oriented
class MinimalOrganismHyper(NeatOrganism):
    def __init__(self, neat_config_path, substrate, kernel, sense_chs, act_chs, torch_device):
        super().__init__(neat_config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        self.name = "Minimal_HyperNEAT"
        self.net = None
        self.neat_config = self.load_neat_config()


    def load_neat_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        # Adaptive Linear Net:
        # ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
        # ["delta_w"],

        # Adaptive Net
        # ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],
        # ['w_ih', 'b_h', 'w_hh', 'b_o', 'w_ho', 'delta_w'])
        n_in = 7
        n_out = 1
        genome_section = 'DefaultGenome'
        config.set(genome_section, 'num_inputs', f'{n_in}')
        config.set(genome_section, 'num_hidden', '7')
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


    def create_torch_net(self, batch_size = None):
        if batch_size is None:
            batch_size=self.substrate.w*self.substrate.h

        input_coords = self.kernel
        output_coords = [[0.0, 0.0]]

        self.net = AdaptiveLinearNet.create(
            self.genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.5,
            weight_max=4.0,
            batch_size=batch_size,
            activation=tanh_activation,
            output_activation=relu_activation,
            cppn_activation=relu_activation,
            device=self.torch_device,
        )
        return self.net
    

    def activate(self, sensor_mem):
        return self.net.activate(sensor_mem)
