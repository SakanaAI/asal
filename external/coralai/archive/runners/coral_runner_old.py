import os
import torch
import taichi as ti
from coralai.instances.coral.coral_physics import apply_physics
from coralai.substrate.substrate import Substrate
from coralai.evolution.ecosystem import Ecosystem
from coralai.evolution.hyper_organism import HyperOrganism
from coralai.substrate.visualization import Visualization

class CoralVis(Visualization):
    def __init__(self, substrate, ecosystem, vis_chs):
        super().__init__(substrate, vis_chs)
        self.ecosystem = ecosystem

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(380 / self.img_w, self.img_w)
        opt_h = min(640 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.opt_window(sub_w)
            current_pos = self.window.get_cursor_pos()
            pos_x = int(current_pos[0] * self.w) % self.w
            pos_y = int(current_pos[1] * self.h) % self.h
            sub_w.text(f"Stats at ({pos_x}, {pos_y}):")
            sub_w.text(
                f"Energy: {self.substrate.mem[0, inds.energy, pos_x, pos_y]:.2f}," +
                f"Infra: {self.substrate.mem[0, inds.infra, pos_x, pos_y]:.2f}," +
                f"Genome: {self.substrate.mem[0, inds.genome, pos_x, pos_y]:.2f}, " 
                # f"Acts: {self.substrate.mem[0, inds.acts, pos_x, pos_y]}"
            )
            sub_w.text(f"Total Energy added: {self.ecosystem.total_energy_added}")
            sub_w.text(f"Total Energy: {torch.sum(self.substrate.mem[0, inds.energy])}")
            sub_w.text(f"Population:")
            for genome_key in self.ecosystem.population.keys():
                sub_w.text(f"{genome_key}: {self.ecosystem.population[genome_key]['infra']}")
            # for channel_name in ['energy', 'infra']:
            #     chindex = self.world.windex[channel_name]
            #     max_val = self.world.mem[0, chindex].max()
            #     min_val = self.world.mem[0, chindex].min()
            #     avg_val = self.world.mem[0, chindex].mean()
            #     sum_val = self.world.mem[0, chindex].sum()
            #     sub_w.text(f"{channel_name}: Max: {max_val:.2f}, Min: {min_val:.2f}, Avg: {avg_val:.2f}, Sum: {sum_val:.2f}")


def main(config_filename, channels, shape, kernel, sense_chs, act_chs, torch_device):
    kernel = torch.tensor(kernel, device=torch_device)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.genome,...] = -1

    def _create_organism(genome_key, genome=None):
        org = HyperOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        if genome is None:
            genome = org.gen_random_genome(genome_key)
        org.set_genome(genome_key, genome=genome)
        org.genome.fitness = 0.1
        org.create_torch_net()
        return org


    def _apply_physics():
        apply_physics(substrate, ecosystem, kernel)

    ecosystem = Ecosystem(substrate, _create_organism, _apply_physics, min_size = 5)
    vis = CoralVis(substrate, ecosystem, ['energy', "infra", "genome"])

    while vis.window.running:
        # substrate.mem[0, inds.com] += torch.randn_like(substrate.mem[0, inds.com]) * 0.1
        if ecosystem.time_step % 10 == 0:
            vis.update()
        ecosystem.update()
        # ecosystem.update_population_infra_sum()

if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename = "coralai/instances/coral/coral_neat.config",
        channels = {
            "genome": ti.f32,
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=7, dtype=ti.f32) # must equal length of kernel
            ),
            "com": ti.types.struct(
                a=ti.f32,
                b=ti.f32,
                c=ti.f32,
                d=ti.f32
            ),
        },
        shape = (100,100),
        kernel = [        [0,-1],
                  [-1, 0],[0, 0],[1, 0],
                          [0, 1],        ],
        sense_chs = ['com'],
        act_chs = ['com'],
        torch_device = torch_device
    )
