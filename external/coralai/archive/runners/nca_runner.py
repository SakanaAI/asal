import os
import torch
import neat
import taichi as ti
import torch.nn as nn

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.evolution.neat_evolver import NEATEvolver


def nca_activation(mem):
    # mem = nn.ReLU()(mem)
    # Calculate the mean across batch and channel dimensions
    mean = mem.mean(dim=(0, 2, 3), keepdim=True)
    # Calculate the variance across batch and channel dimensions
    var = mem.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    # Normalize the input tensor
    mem.sub_(mean).div_(torch.sqrt(var + 1e-5))
    mem = torch.sigmoid(mem)
    return mem


def main(config_filename, channels, shape, kernel, sense_chs, act_chs, torch_device):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    kernel = torch.tensor(kernel, device = torch_device)
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()
    inds = substrate.ti_indices[None]

    num_cells_to_activate = 100
    for _ in range(num_cells_to_activate):
        x = torch.randint(0, shape[0], (1,))
        y = torch.randint(0, shape[1], (1,))
        substrate.mem[0,inds.genome, x, y] = -1

    neat_evolver = NEATEvolver(config_path, substrate, kernel, 2, sense_chs, act_chs,)
    genome = neat.DefaultGenome(str(0))
    genome.configure_new(neat_evolver.neat_config.genome_config)
    net = neat_evolver.create_torch_net(genome)
    weights = net.weights.unsqueeze(0)
    biases = net.biases.unsqueeze(0)

    vis = Visualization(substrate, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])
    while vis.window.running:
        substrate.mem[:, neat_evolver.act_chinds] += torch.rand_like(substrate.mem[:, neat_evolver.act_chinds]) * 0.1
        neat_evolver.forward(weights, biases)
        substrate.mem[:, neat_evolver.act_chinds] = nca_activation(substrate.mem[:, neat_evolver.act_chinds])
        vis.update()
        if vis.mutating:
            genome.mutate(neat_evolver.neat_config.genome_config)
            net = neat_evolver.create_torch_net(genome)
            weights = net.weights.unsqueeze(0)
            biases = net.biases.unsqueeze(0)
            vis.mutating=False


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename="coralai/instances/nca/nca_neat.config",
        channels={
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=10, dtype=ti.f32),
            "genome": ti.f32,
        },
        shape=(80, 80),
        kernel=[        [0,-1],
                [-1, 0],[0, 0],[1, 0],
                        [0, 1]],
        sense_chs=['rgb', 'hidden'],
        act_chs=['rgb', 'hidden'],
        torch_device=torch_device
    )
