import torch
import neat

class Ecosystem():
    def __init__(self, substrate, create_organism, apply_physics, min_size=5, max_size=30):
        self.substrate = substrate
        self.create_organism = create_organism
        self.apply_physics = apply_physics
        self.min_size = min_size
        self.max_size = max_size

        inds = self.substrate.ti_indices[None]

        if min_size < 1:
            raise ValueError("ecosystem: initial_size must be greater than 0")
        
        self.act_chinds = None
        self.sense_chinds = None
        self.neat_config = None

        self.population = {}
        self.next_free_genome_key = 0
        self.gen_random_pop(min_size)

        self.time_step = 0
        self.out_mem = None
        self.total_energy_added = 0.0

    
    def gen_random_pop(self, num_organisms):
        for _ in range(num_organisms):
            org = self.create_organism(genome_key=self.next_free_genome_key)
            if self.act_chinds is None:
                self.act_chinds = org.act_chinds
                self.sense_chinds = org.sense_chinds
                self.neat_config = org.neat_config
            self.population[self.next_free_genome_key] = {"org": org, "infra": 0.1, "age": 0}
            self.next_free_genome_key += 1
    

    def sexual_reproduction(self, merging_cell_coords, incoming_genome_matrix):
        inds = self.substrate.ti_indices[None]
        if len(merging_cell_coords[0]) == 0:
            return
        for i in range(merging_cell_coords[0].shape[0]):
            x, y = merging_cell_coords[0][i].item(), merging_cell_coords[1][i].item()
            old_genome_key = self.substrate.mem[0, inds.genome, x, y].item()
            incoming_genome_key = incoming_genome_matrix[x, y].item()
            incoming_org = self.population[incoming_genome_key]['org']
            if (old_genome_key == -1 or
                old_genome_key == incoming_genome_matrix[x, y]
                or old_genome_key not in self.population):
                self.substrate.mem[0, inds.genome, x, y] = incoming_genome_key
                continue
            old_org = self.population[old_genome_key]['org']
            child_genome = neat.DefaultGenome(str(self.next_free_genome_key))
            child_genome.configure_crossover(old_org.genome, incoming_org.genome, self.neat_config)
            child_organism = self.create_organism(genome_key=self.next_free_genome_key, genome=child_genome)
            self.population[self.next_free_genome_key] = {"org": child_organism, "infra": 0.1, "age": 0}
            self.substrate.mem[0, inds.genome, x, y] = self.next_free_genome_key
            self.next_free_genome_key += 1


    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum


    def update_population_infra_sum(self):
        for genome_key in self.population.keys():
            infra_sum = self.get_genome_infra_sum(genome_key)
            self.population[genome_key]["infra"] = infra_sum
            self.population[genome_key]["org"].fitness = infra_sum


    def get_random_coords_of_genome(self, genome_key, n_coords=1):
        inds = self.substrate.ti_indices[None]
        genome_coords = torch.where(self.substrate.mem[0, inds.genome] == genome_key)
        if genome_coords[0].shape[0] == 0:  # Check if there are no matching coordinates
            return []  # Return an empty list or handle this case as needed
        random_indices = torch.randint(0, genome_coords[0].shape[0], (n_coords,))
        x_coords = genome_coords[0][random_indices]
        y_coords = genome_coords[1][random_indices]
        coords = torch.stack((x_coords, y_coords), dim=1)
        return coords.tolist()


    def get_random_genome_keys(self, n_genomes):
        inds = self.substrate.ti_indices[None]
        infras = torch.tensor([org_info['infra'] for org_info in self.population.values()], dtype=torch.float32)
        infra_sum = infras.sum()
        if infra_sum != 0:
            infra_probs = infras / infra_sum
        else:
            # Create a uniform distribution if infra_sum is 0
            infra_probs = torch.ones_like(infras) / len(infras)
        selected_index = torch.multinomial(infra_probs, n_genomes, replacement=True)
        selected_genomes = [list(self.population.keys())[i] for i in selected_index]
        return selected_genomes


    def sew_seeds(self, n_seeds):
        inds = self.substrate.ti_indices[None]
        selected_genome_keys = self.get_random_genome_keys(n_seeds)
        random_x_coords = torch.randint(0, self.substrate.w, (n_seeds,))
        random_y_coords = torch.randint(0, self.substrate.h, (n_seeds,))
        for i in range(n_seeds):
            self.substrate.mem[0, inds.genome, random_x_coords[i], random_y_coords[i]] = selected_genome_keys[i]
            # self.substrate.mem[0, inds.infra, random_x_coords[i], random_y_coords[i]] += 2


    def mutate(self, genome_key, report=False):
        new_genome = self.population[genome_key]['org'].mutate()
        new_organism = self.create_organism(genome_key=self.next_free_genome_key, genome=new_genome)
        self.population[self.next_free_genome_key] = {"org": new_organism, "infra": 0.1, "age": 0}
        self.next_free_genome_key += 1
        if report:
            print(f"Mutated genome {genome_key} to {self.next_free_genome_key-1}")
            print(f"New Genome: {new_genome}")
            print(f"Weights: {new_organism.net.weights}")
            print("--------")
        return self.next_free_genome_key-1


    def apply_radiation(self, n_radiation_spots):
        inds = self.substrate.ti_indices[None]
        selected_genome_keys = self.get_random_genome_keys(n_radiation_spots)
        random_x_coords = torch.randint(0, self.substrate.w, (n_radiation_spots,))
        random_y_coords = torch.randint(0, self.substrate.h, (n_radiation_spots,))
        for i in range(n_radiation_spots):
            new_genome_key = self.mutate(selected_genome_keys[i])
            coords = self.get_random_coords_of_genome(selected_genome_keys[i])
            if coords:  # Check if coords is not empty
                self.substrate.mem[0, inds.genome, coords[0][0], coords[0][1]] = new_genome_key
    
    
    def save_population(self):
        pass


    def update(self, seed_interval=100, seed_volume=10, radiation_interval=500, radiation_volume=10):
        self.update_population_infra_sum()
        if self.time_step % seed_interval == 0:
            self.sew_seeds(seed_volume)
        if self.time_step % radiation_interval == 0:
            self.apply_radiation(radiation_volume)
        genomes_to_remove = []
        if self.out_mem is None:
            self.out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
        else:
            self.out_mem[:] = 0.0
        for genome_key, org_info in self.population.items():
            self.out_mem = org_info['org'].forward(self.out_mem)
            org_info['age'] += 1
            if org_info['age'] > 500 and org_info['infra'] < 1:
                genomes_to_remove.append(genome_key)

        for genome_key in genomes_to_remove:
            self.population.pop(genome_key)

        if len(self.population.keys()) > self.max_size:
            # Calculate how many genomes to remove
            num_to_remove = len(self.population) - self.max_size
            # Sort genomes by infra value (ascending order) and select the ones to remove
            genomes_to_remove = sorted(self.population.items(), key=lambda x: x[1]['infra'])[:num_to_remove]
            # Remove the selected genomes
            for genome_key, _ in genomes_to_remove:
                self.population.pop(genome_key)

        if len(self.population) < self.min_size:
            self.gen_random_pop(self.min_size - len(self.population))
        
        self.substrate.mem[0, self.act_chinds] = self.out_mem
        self.apply_physics()
        self.time_step += 1