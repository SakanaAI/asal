"""
coral_dev evolution strategies — experiment-specific, snapshotted with each run.

These standalone functions implement the spatial evolution dynamics for the
coral_dev experiment. They are imported by run.py and replay.py, and snapshotted
into each run directory so replays always use the exact strategy that generated them.
"""

import copy
import random

import neat
import numpy as np
import torch


def kill_random_chunk(evolver, radius=5):
    """Kill all cells in a randomly placed circle."""
    x = np.random.randint(0, evolver.substrate.w)
    y = np.random.randint(0, evolver.substrate.h)
    evolver.set_chunk(-1, x, y, radius)


def apply_radiation_mutation(evolver, n_spots=5, spot_live_radius=2, spot_dead_radius=4):
    """Spatial NEAT radiation: kill a circle, then mutate or crossover at that spot."""
    inds = evolver.substrate.ti_indices[None]
    xs = torch.randint(0, evolver.substrate.w, (n_spots,))
    ys = torch.randint(0, evolver.substrate.h, (n_spots,))

    for i in range(n_spots):
        genome_key = int(evolver.substrate.mem[0, inds.genome, xs[i], ys[i]].item())
        rand_genome_key = torch.randint(0, len(evolver.genomes), (1,)).item()
        evolver.set_chunk(-1, xs[i], ys[i], spot_dead_radius)

        if genome_key < 0:
            # Empty spot — seed with a random existing genome
            evolver.set_chunk(rand_genome_key, xs[i], ys[i], spot_live_radius)
        else:
            if random.random() < 0.5:
                # Mutate
                new_genome = copy.deepcopy(evolver.genomes[genome_key])
                new_genome.mutate(evolver.neat_config.genome_config)
                new_key = evolver.add_organism_get_key(new_genome)
                evolver.set_chunk(new_key, xs[i], ys[i], spot_live_radius)
            else:
                # Crossover
                new_genome = neat.DefaultGenome(str(len(evolver.genomes)))
                evolver.genomes[genome_key].fitness = 0.0
                evolver.genomes[rand_genome_key].fitness = 0.0
                new_genome.configure_crossover(
                    evolver.genomes[genome_key],
                    evolver.genomes[rand_genome_key],
                    evolver.neat_config,
                )
                new_key = evolver.add_organism_get_key(new_genome)
                evolver.set_chunk(new_key, xs[i], ys[i], spot_live_radius)
