import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm
from ...evolution.organism import Organism

@ti.data_oriented
class NCAOrganismCNN(Organism):
    def __init__(self, substrate, kernel, sense_chs, act_chs, torch_device, latent_size = None):
        super().__init__(substrate, kernel, sense_chs, act_chs, torch_device)

        if latent_size is None:
            latent_size = (self.n_senses + self.n_acts) // 2
        self.latent_size = latent_size

        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_senses,
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
            self.n_acts,
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

            x = self.latent_conv_2(x)
            x = nn.ReLU()(x)
            x = ch_norm(x)
            x = torch.sigmoid(x)

            return x


    def mutate(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data)
        self.latent_conv.weight.data += perturbation_strength * torch.randn_like(self.latent_conv.weight.data)
        self.latent_conv_2.weight.data += perturbation_strength * torch.randn_like(self.latent_conv_2.weight.data)