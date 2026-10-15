import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm
from ...evolution.organism import Organism

@ti.data_oriented
class MinimalOrganismCNN(Organism):
    def __init__(self, substrate, kernel, sense_chs, act_chs, torch_device):
        super().__init__(substrate, kernel, sense_chs, act_chs, torch_device)
        self.name = "MinimalOrganismCNN"
        
        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_senses,
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
            x = ch_norm(x)
            x = nn.ReLU()(x)
            x = torch.sigmoid(x)

            self.substrate.mem[self.act_chinds] = x
            # return x


    def mutate(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data)
    