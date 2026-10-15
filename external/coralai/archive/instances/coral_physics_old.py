import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm

@ti.kernel
def distribute_energy(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                      kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        infra_sum = 0.0
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            if mem[0, inds.infra, neigh_x, neigh_y] < 0.1:
                mem[0, inds.infra, neigh_x, neigh_y] = 0.1
            infra_sum += mem[0, inds.infra, neigh_x, neigh_y]
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            neigh_infra = mem[0, inds.infra, neigh_x, neigh_y]
            out_energy_mem[neigh_x, neigh_y] += (
                central_energy * (neigh_infra/infra_sum))


def energy_physics(substrate, kernel):
    inds = substrate.ti_indices[None]
    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    distribute_energy(substrate.mem, energy_out_mem, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = energy_out_mem


def invest_liquidate(substrate, live_mask):
    inds = substrate.ti_indices[None]
    substrate.mem[0, [inds.acts_invest, inds.acts_liquidate]] = torch.softmax(substrate.mem[0, [inds.acts_invest, inds.acts_liquidate]], dim=0) * live_mask
    investments = substrate.mem[0, inds.acts_invest] * substrate.mem[0, inds.energy]
    liquidations = substrate.mem[0, inds.acts_liquidate] * substrate.mem[0, inds.infra]
    substrate.mem[0, inds.energy] += liquidations - investments
    substrate.mem[0, inds.infra] += investments - liquidations


@ti.kernel
def explore(mem: ti.types.ndarray(), max_act_i: ti.types.ndarray(), live_mask: ti.types.ndarray(),
            infra_delta: ti.types.ndarray(), winning_genomes: ti.types.ndarray(),
            kernel: ti.types.ndarray(), explore_inds: ti.types.ndarray(), ti_inds: ti.template()):
    # incoming investments
    inds = ti_inds[None]
    for center_x, center_y in ti.ndrange(mem.shape[2], mem.shape[3]):
        winning_genome = -1.0
        max_bid = 0.0
        for offset_n in ti.ndrange(kernel.shape[0]):
            offset_x = kernel[offset_n, 0]
            offset_y = kernel[offset_n, 1]
            neigh_x = (center_x + offset_x) % mem.shape[2]
            neigh_y = (center_y + offset_y) % mem.shape[3]

            if live_mask[neigh_x, neigh_y] == 0:
                continue
            neigh_max_act_i = max_act_i[neigh_x, neigh_y]
            bid = 0.0
            # if neigh's explore dir points towards this center
            if (((kernel[neigh_max_act_i, 0] + offset_x) == 0 and (kernel[neigh_max_act_i, 1] + offset_y) == 0) or
                (offset_x == 0 and offset_y == 0)): # central cell always gets a bid 
                if offset_x == 0 and offset_y == 0:
                    bid = mem[0, inds.infra, center_x, center_y] # max possible bid
                else:
                    bid = (
                        mem[0, inds.infra, neigh_x, neigh_y] *
                        mem[0, explore_inds[neigh_max_act_i], neigh_x, neigh_y])
                    infra_delta[neigh_x, neigh_y] -= bid # bids are always taken as investment
                    infra_delta[center_x, center_y] += bid
                if bid > max_bid:
                    max_bid = bid
                    winning_genome = mem[0, inds.genome, neigh_x, neigh_y]
        winning_genomes[center_x, center_y] = winning_genome


# def handle_investment(substrate, investments):
#     inds = substrate.ti_indices[None]
#     # remove energy where investments are negative, convert investments to infra where positive
#     substrate.mem[0, inds.energy] = torch.where(investments < 0, substrate.mem[0, inds.energy] - investments, substrate.mem[0, inds.energy])
#     substrate.mem[0, inds.infra] += torch.where(investments > 0, substrate.mem[0, inds.infra] + torch.log(investments+1), substrate.mem[0, inds.infra])


def explore_physics(substrate, live_mask, kernel):
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.acts_explore] = torch.softmax(substrate.mem[0, inds.acts_explore], dim=1) * live_mask
    max_act_i = torch.argmax(substrate.mem[0, inds.acts_explore], dim=0) # be warned, this is the index of the actuator not the index in memory, so 0-6 not
    infra_delta = torch.zeros_like(substrate.mem[0, inds.infra])
    winning_genome = substrate.mem[0, inds.genome] * 1
    # This is intermediate storage for each cell in the kernel to use:
    explore_inds = torch.tensor(inds.acts_explore)
    explore(substrate.mem, max_act_i, live_mask,
            infra_delta, winning_genome,
            kernel, explore_inds, substrate.ti_indices)
    # handle_investment(substrate, infra_delta)
    substrate.mem[0, inds.infra] += infra_delta


@ti.kernel
def get_live_cell_mask(mem: ti.types.ndarray(), live_mask: ti.types.ndarray(),
                            live_genomes: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        is_alive = 0
        curr_genome = mem[0, inds.genome, i, j]
        for genome_i in ti.ndrange(live_genomes.shape[0]):
            if curr_genome == live_genomes[genome_i]:
                is_alive = 1
        live_mask[i, j] = is_alive
        if is_alive == 0:
            mem[0, inds.genome, i, j] = -1


def apply_physics(substrate, ecosystem, kernel):
    inds = substrate.ti_indices[None]

    substrate.mem[0, inds.com] = torch.sigmoid(nn.ReLU()(ch_norm(substrate.mem[:, inds.com])))

    live_genomes = torch.tensor(list(ecosystem.population.keys()), device = substrate.torch_device)
    live_mask = torch.zeros_like(substrate.mem[0, inds.genome])
    get_live_cell_mask(substrate.mem, live_mask, live_genomes, substrate.ti_indices)
    substrate.mem[0, inds.acts] *= live_mask

    substrate.mem[0, inds.energy] += torch.randn_like(substrate.mem[0, inds.energy]) * 0.1

    invest_liquidate(substrate, live_mask)
    # to_convert = substrate.mem[0, inds.infra] * (torch.randn_like(substrate.mem[0, inds.energy])+1) * 0.1
    # substrate.mem[0, inds.infra] -= to_convert
    # substrate.mem[0, inds.energy] += to_convert
    explore_physics(substrate, live_mask, kernel)
    energy_physics(substrate, kernel)
