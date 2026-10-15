import torch
import taichi as ti

from pytorch_neat.cppn import create_cppn
from .neat_organism import NeatOrganism

@ti.data_oriented
class CPPNOrganism(NeatOrganism):
    def __init__(self, config_path, substrate, kernel, sense_chs, act_chs, torch_device):
        super().__init__(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        self.name = "OrganismCPPN"

        (self.leaf_names, self.node_names) = self.gen_leaf_node_names()
        self.net = None


    def gen_leaf_node_names(self):
        leaf_names = []
        node_names = []
        for kernel_offset in self.kernel:
            for s_chind in self.sense_chinds:
                leaf_names.append(f'sense_chind_{s_chind}_in_{kernel_offset}')
            for a_chind in self.act_chinds:
                node_names.append(f'act_chind_{a_chind}_out_{kernel_offset}')
        return leaf_names, node_names

    
    def create_torch_net(self, batch_size=None):
        self.net = create_cppn(genome = self.genome,
                               config = self.neat_config,
                               leaf_names = self.leaf_names,
                               node_names = self.node_names,
                               device = self.torch_device)


    def activate(self, sensor_mem):
        inputs = {}
        for i, name in enumerate(self.leaf_names):
            inputs[name] = sensor_mem[:, i]
        
        batch_size = sensor_mem.shape[0]
        n_actions = len(self.net)
        actions = torch.zeros((batch_size, n_actions), dtype=torch.float32, device=self.torch_device)
        
        for i, output_node in enumerate(self.net):
            result = output_node(**inputs)
            actions[:, i] = result
        
        return actions
