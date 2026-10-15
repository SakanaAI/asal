import os
import torch

import taichi as ti

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.evolution.cppn_organism import CPPNOrganism
from coralai.instances.minimal.minimal_organism_cnn import MinimalOrganismCNN
from coralai.instances.minimal.minimal_organism_hyper import MinimalOrganismHyper

from coralai.evolution.neat_organism import NeatOrganism


def define_substrate(shape, torch_device):
    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "bw": ti.f32,
        },
    )
    substrate.malloc()
    return substrate


def main():
    shape = (256, 256)
    torch_device = torch.device("mps")
    ti.init(ti.metal)

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = "coralai/instances/minimal/minimal_neat.config"
    config_path = os.path.join(local_dir, config_filename)

    substrate = define_substrate(shape, torch_device)
    kernel = [[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]]
    
    sense_chs = ['bw']
    act_chs = ['bw']
    
    genome_key = 0
    genome_map = torch.zeros(shape[0], shape[1], dtype=torch.int32, device=torch_device)
    vis = Visualization(substrate, ["bw"])

    organism_cnn = MinimalOrganismCNN(substrate, kernel, sense_chs, act_chs, torch_device)
    organism_rnn = NeatOrganism(config_path,substrate, kernel, sense_chs, act_chs, torch_device)
    organism_cppn = CPPNOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_hyper = MinimalOrganismHyper(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism = organism_hyper
    
    if organism.is_evolvable:
        organism.set_genome(genome_key, organism.gen_random_genome(0))
        organism.create_torch_net()
    
    while vis.window.running:
        if organism.is_evolvable:
            organism.forward(substrate.mem, genome_map)
        else:
            organism.forward(substrate.mem)
        vis.update()
        if vis.mutating:
            new_genome = organism.mutate()
            if organism.is_evolvable:
                organism.set_genome(organism.genome_key, new_genome) # mutates all cells at once
                organism.create_torch_net()
            

if __name__ == "__main__":
    main()