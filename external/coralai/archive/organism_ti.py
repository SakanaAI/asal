import taichi as ti
import torch
import torch.nn as nn
from .nn_lib import ch_norm
from .Organism import Organism

LATENT_SIZE = 4

@ti.data_oriented
class Organism_ti(Organism):
    # NOT CURRENTLY INTEGRATED INTO THE FLOW, MORE OF A REPOSITORY. LOOK AT CORAL ORGANISM TI
    def __init__(self, world):
        self.world = world
        self.shape = world.shape
        self.w = world.w
        self.h = world.h
        self.sensors = [('com', 'r'), ('com', 'g'), ('com', 'b'), ('com', 'a')]
        self.actuators = [('com', 'r'), ('com', 'g'), ('com', 'b'), ('com', 'a')]
        self.define_weights()


    def define_weights(self):
        self.sensor_inds = self.world.windex_obj[self.sensors]
        self.actuator_inds = self.world.windex_obj[self.actuators]
        self.n_sensors = self.sensor_inds.shape[0]
        self.n_actuators = self.actuator_inds.shape[0]
        # self.sense_weights = torch.randn(self.n_sensors, LATENT_SIZE, 3, 3)
        # self.convout = torch.zeros(self.shape[0], self.shape[1], self.n_sensors)
        self.sense_weights = torch.randn(self.n_sensors, LATENT_SIZE, 3, 3, device=self.world.torch_device)
        self.latent_layer = torch.zeros(self.w, self.h, LATENT_SIZE,  device=self.world.torch_device)
        self.act_weights = torch.randn(LATENT_SIZE, self.n_actuators, device=self.world.torch_device)
        self.latent_bias = torch.zeros(1, device=self.world.torch_device)#torch.randn(1)
        self.act_bias = torch.zeros(1, device=self.world.torch_device)#torch.randn(1)
        

    @ti.kernel
    def sense_act(self,
                  mem: ti.types.ndarray(),
                  sensor_inds: ti.types.ndarray(),
                  weights: ti.types.ndarray(),
                  bias: ti.types.ndarray(),
                  actuator_inds: ti.types.ndarray()):
        """
        This is a zero layer simple convolution
        """
        for row_idx, col_idx, actuator_idx in ti.ndrange(self.w, self.h, self.n_actuators):
            actuator_sum = 0.0
            for offset_row, offset_col, sensor_idx in ti.ndrange((-1, 2), (-1, 2), self.n_sensors):
                offset_row_idx = (row_idx + offset_row) % self.w
                offset_col_idx = (col_idx + offset_col) % self.h
                actuator_sum += (weights[sensor_idx, actuator_idx, offset_row, offset_col] *
                                   mem[offset_row_idx, offset_col_idx, sensor_inds[sensor_idx]])
            actuator_sum += bias
            mem[row_idx, col_idx, actuator_inds[actuator_idx]] = actuator_sum


    @ti.kernel
    def sense(self,
              mem: ti.types.ndarray(),
              sensor_inds: ti.types.ndarray(),
              sense_weights: ti.types.ndarray(),
              latent_bias: ti.types.ndarray(),
              latent_layer: ti.types.ndarray()):
        for row_idx, col_idx, latent_idx in ti.ndrange(self.w, self.h, LATENT_SIZE):
            lat_node_sum = 0.0
            for sensor_idx, offset_row, offset_col in ti.ndrange(self.n_sensors, (-1, 2), (-1, 2)):
                offset_row_idx = (row_idx + offset_row) % self.w
                offset_col_idx = (col_idx + offset_col) % self.h
                lat_node_sum += (sense_weights[sensor_idx, latent_idx, offset_row, offset_col] *
                                   mem[offset_row_idx, offset_col_idx, sensor_inds[sensor_idx]])
            latent_layer[row_idx, col_idx, latent_idx] = lat_node_sum


    @ti.kernel
    def act(self,
            latent_layer: ti.types.ndarray(),
            actuator_weights: ti.types.ndarray(),
            actuator_biases: ti.types.ndarray(),
            actuator_inds: ti.types.ndarray(),
            mem: ti.types.ndarray()):
        for actuator_idx, row_idx, col_idx in ti.ndrange(self.n_actuators, self.w, self.h):
            actuator_sum = 0.0
            for latent_idx in ti.static(range(LATENT_SIZE)):
                actuator_sum = actuator_weights[latent_idx, actuator_idx] * latent_layer[latent_idx, row_idx, col_idx]

            # TODO: APPLY FINAL ACTIVATION BASED ON LIMITS
            mem[row_idx, col_idx, actuator_inds[actuator_idx]] = actuator_sum

    
    def apply_weights(self):
        self.sense(
            self.world.mem,
            self.sensor_inds,
            self.sense_weights,
            self.latent_bias,
            self.world[self.actuators])
            # self.latent_layer)
        x = self.world[self.actuators]
        x = nn.ReLU()(x)
        ch_norm(x)
        x = torch.sigmoid(x)
        self.world[self.actuators] = x

        # x = self.latent_layer
        # self.world[self.actuators] = x.permute(1, 2, 0)
        # x = torch.sigmoid(x)
        # x = nn.ReLU()(x)
        # self.ch_norm_(self.latent_layer)
        # x = torch.sigmoid(x)
        # self.latent_layer = x

        # self.act(
        #     self.latent_layer,
        #     self.act_weights,
        #     self.act_bias,
        #     self.actuator_inds,
        #     self.world.mem)
        
        # x = self.world[self.actuators]
        # x = torch.sigmoid(x)
        # x = nn.ReLU()(x)
        # self.ch_norm_(x)
        # x = torch.sigmoid(x)

    def perturb_weights(self, perturb_strength=0.1):
        self.sense_weights += torch.randn_like(self.sense_weights) * perturb_strength
        self.act_weights += torch.randn_like(self.act_weights) * perturb_strength

    def perturb_biases(self, perturb_strength=0.1):
        self.latent_bias += torch.randn_like(self.latent_bias) * perturb_strength
        self.act_bias += torch.randn_like(self.act_bias) * perturb_strength
