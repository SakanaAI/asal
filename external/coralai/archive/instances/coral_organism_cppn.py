import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm
from ...evolution.organism import Organism

@ti.data_oriented
class CoralOrganism(Organism):
    def __init__(self, n_sensors, n_actuators, torch_device, latent_size = None):
        super().__init__(n_sensors, n_actuators)

        if latent_size is None:
            latent_size = (self.n_sensors + self.n_actuators) // 2
        self.latent_size = latent_size

        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_sensors,
            self.latent_size,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=torch_device,
            bias=False
        )

        self.latent_conv = nn.Conv2d(
            self.latent_size,
            self.latent_size,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=torch_device,
            bias=False
        )

        self.latent_conv_2 = nn.Conv2d(
            self.latent_size,
            self.n_actuators,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=torch_device,
            bias=False
        )
    
    def forward(self, x):
        with torch.no_grad():
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = ch_norm(x)
            x = torch.sigmoid(x)

            x = self.latent_conv(x)
            x = nn.ReLU()(x)
            x = ch_norm(x)
            x = torch.sigmoid(x)

            return self.latent_conv_2(x)


    def perturb_weights(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data)
        self.latent_conv.weight.data += perturbation_strength * torch.randn_like(self.latent_conv.weight.data)
        self.latent_conv_2.weight.data += perturbation_strength * torch.randn_like(self.latent_conv_2.weight.data)