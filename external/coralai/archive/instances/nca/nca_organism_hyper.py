import os
import torch
import neat
import taichi as ti
import configparser
from datetime import datetime
import torch.nn as nn

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import relu_activation, sigmoid_activation, tanh_activation, identity_activation
from pytorch_neat.linear_net import LinearNet
from ...evolution.neat_organism import NeatOrganism
from ...substrate.nn_lib import ch_norm


@ti.data_oriented
class CoralHyperOrganism(NeatOrganism):
    def __init__(self, neat_config_path, substrate, kernel, sense_chs, act_chs, torch_device):
        super().__init__(neat_config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        self.name = "Coral_Hyper_Organism"
        self.net = None
        self.neat_config = self.load_neat_config()
        self.w = self.substrate.w
        self.h = self.substrate.h


    def load_neat_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        # Adaptive Linear Net:
        # ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
        # ["delta_w"],

        # Adaptive Net
        # ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],
        # ['w_ih', 'b_h', 'w_hh', 'b_o', 'w_ho', 'delta_w'])

        # Linear Net
        # ["x_in", "y_in", "z_in", "x_out", "y_out", "z_out"],
        # ["w, b"],
        n_in = 6
        n_out = 2
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

        input_coords = []
        for offset in self.kernel:
            for ch in range(self.n_senses):
                input_coords.append([offset[0], offset[1], self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])
        

        self.net = LinearNet.create(
            self.genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.0,
            weight_max=3.0,
            batch_size=batch_size,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.torch_device,
        )
        return self.net


    @ti.kernel
    def apply_weights_and_biases(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                                 cell_coords: ti.types.ndarray(),
                                 kernel: ti.types.ndarray(), sense_chinds: ti.types.ndarray(), act_chinds: ti.types.ndarray(),
                                 weights: ti.types.ndarray(), biases: ti.types.ndarray()):
        for cell_i, act_j in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
            val = 0.0
            center_x = cell_coords[cell_i, 0]
            center_y = cell_coords[cell_i, 1]
            for sensor_n, off_m in ti.ndrange(sense_chinds.shape[0], kernel.shape[0]):
                neigh_x = (center_x + kernel[off_m, 0]) % mem.shape[2]
                neigh_y = (center_y + kernel[off_m, 1]) % mem.shape[3]
                val += mem[0, sense_chinds[sensor_n], neigh_x, neigh_y] * weights[0, act_j, sensor_n]
            # This is okay because actuators are independent of sensors
            out_mem[0, act_chinds[act_j], center_x, center_y] = val + biases[0, act_j, 0]


    def forward(self):
        """
        input: (w,h) where each value corresponds to a genome key

        Updates the actuators in substrate memory where this organism's genome is present on the map
        """
        with torch.no_grad():
            inds = self.substrate.ti_indices[None]
            cell_coords = self.get_cell_coords(self.substrate.mem[0, inds.genome])
            if cell_coords.shape[0] == 0:
                return
            out_mem = torch.zeros_like(self.substrate.mem)
            self.substrate.mem[0,inds.com] += torch.randn_like(self.substrate.mem[0,inds.com]) * 0.01
            self.substrate.mem[0,inds.com] = torch.clamp(self.substrate.mem[0,inds.com], 0, 1)
            self.apply_weights_and_biases(self.substrate.mem, out_mem, cell_coords,
                      self.kernel, self.sense_chinds, self.act_chinds,
                      self.net.weights, self.net.biases)
            self.mem = out_mem
