import torch
import taichi as ti
import torch.nn as nn

from coralai.nn_lib import ch_norm


def activate_outputs(substrate):
    inds = substrate.ti_indices[None]
    mem = substrate.mem

    com_indices = list(inds.com)
    com_start, com_end = com_indices[0], com_indices[-1] + 1
    com_slice = mem[:, com_start:com_end]
    com_mean = com_slice.mean(dim=(0, 2, 3), keepdim=True)
    com_var = com_slice.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    mem[:, com_start:com_end] = torch.sigmoid(
        (com_slice - com_mean) / torch.sqrt(com_var + 1e-5))

    inv_liq = mem[0, inds.acts_invest:inds.acts_liquidate + 1]
    mem[0, inds.acts_invest:inds.acts_liquidate + 1] = torch.softmax(inv_liq, dim=0)

    explore_indices = list(inds.acts_explore)
    exp_start, exp_end = explore_indices[0], explore_indices[-1] + 1
    explore_vals = torch.relu(mem[0, exp_start:exp_end])
    explore_vals[0] = explore_vals.mean(dim=0)
    mem[0, exp_start:exp_end] = torch.softmax(explore_vals, dim=0)

    act_indices = list(inds.acts)
    act_start, act_end = act_indices[0], act_indices[-1] + 1
    dead = mem[0, inds.genome:inds.genome + 1] < 0
    mem[0, act_start:act_end] *= (~dead).float()


@ti.kernel
def explore(mem: ti.types.ndarray(), max_act_i: ti.types.ndarray(),
            infra_delta: ti.types.ndarray(), energy_delta: ti.types.ndarray(),
            winning_genomes: ti.types.ndarray(), winning_rots: ti.types.ndarray(),
            dir_kernel: ti.types.ndarray(), dir_order: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        winning_genome = mem[0, inds.genome, i, j]
        max_bid = mem[0, inds.energy, i, j]
        winning_rot = mem[0, inds.rot, i, j]

        for offset_n in ti.ndrange(dir_kernel.shape[0]): # this order doesn't matter
            neigh_x = (i + dir_kernel[offset_n, 0]) % mem.shape[2]
            neigh_y = (j + dir_kernel[offset_n, 1]) % mem.shape[3]
            if mem[0, inds.genome, neigh_x, neigh_y] < 0:
                continue
            neigh_max_act_i = max_act_i[neigh_x, neigh_y] # Could be [0,0], so could overflow dir_kernel
            if neigh_max_act_i == 0:
                continue
            neigh_max_act_i -= 1 # aligns with dir_kernel now
            neigh_rot = int(mem[0, inds.rot, neigh_x, neigh_y])
            neigh_dir_ind = (neigh_rot + int(dir_order[neigh_max_act_i]) + dir_kernel.shape[0]) % dir_kernel.shape[0]
            neigh_dir_x = dir_kernel[neigh_dir_ind, 0]
            neigh_dir_y = dir_kernel[neigh_dir_ind, 1]
            bid = 0.0
            # If neigh's explore dir points towards this center
            if ((neigh_dir_x + dir_kernel[offset_n, 0]) == 0 and (neigh_dir_y + dir_kernel[offset_n, 1]) == 0):
                bid = mem[0, inds.energy, neigh_x, neigh_y]
                energy_delta[neigh_x, neigh_y] -= bid # bids are always taken as investment
                bid = 0.9 # cost of dooing business
                infra_delta[i, j] += bid
                if bid > max_bid:
                    max_bid = bid
                    winning_genome = mem[0, inds.genome, neigh_x, neigh_y]
                    winning_rot = (neigh_rot + int(dir_order[neigh_max_act_i]) + dir_kernel.shape[0]) % dir_kernel.shape[0]
        winning_genomes[i, j] = winning_genome
        winning_rots[i, j] = winning_rot


_explore_scratch = {}

def explore_physics(substrate, dir_kernel, dir_order):
    inds = substrate.ti_indices[None]
    grid_shape = substrate.mem[0, inds.infra].shape
    device = substrate.torch_device

    max_act_i = torch.argmax(substrate.mem[0, inds.acts_explore], dim=0)

    if "infra_delta" not in _explore_scratch or _explore_scratch["infra_delta"].shape != grid_shape:
        _explore_scratch["infra_delta"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=device)
        _explore_scratch["energy_delta"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=device)
        _explore_scratch["winning_genome"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=device)
        _explore_scratch["winning_rots"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=device)
    else:
        _explore_scratch["infra_delta"].zero_()
        _explore_scratch["energy_delta"].zero_()
        _explore_scratch["winning_genome"].zero_()
        _explore_scratch["winning_rots"].zero_()

    explore(substrate.mem, max_act_i,
            _explore_scratch["infra_delta"], _explore_scratch["energy_delta"],
            _explore_scratch["winning_genome"], _explore_scratch["winning_rots"],
            dir_kernel, dir_order, substrate.ti_indices)
    substrate.mem[0, inds.infra] += _explore_scratch["infra_delta"]
    substrate.mem[0, inds.energy] += _explore_scratch["energy_delta"]
    substrate.mem[0, inds.genome] = _explore_scratch["winning_genome"]
    substrate.mem[0, inds.rot] = _explore_scratch["winning_rots"]


@ti.kernel
def flow_energy_down(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                     max_energy: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        if central_energy > max_energy:
            energy_sum_inverse = 0.0
            # Calculate the sum of the inverse of energy levels for neighboring cells
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                energy_level = mem[0, inds.energy, neigh_x, neigh_y]
                # Avoid division by zero by ensuring a minimum energy level
                energy_level = max(energy_level, 0.0001)  # Assuming 0.0001 as a minimum to avoid division by zero
                energy_sum_inverse += 1.0 / energy_level
            # Distribute energy based on the inverse proportion of energy levels
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                neigh_energy = mem[0, inds.energy, neigh_x, neigh_y]
                neigh_energy = max(neigh_energy, 0.0001)  # Again, ensuring a minimum energy level
                # Calculate the share of energy based on the inverse of energy level
                energy_share = central_energy * ((1.0 / neigh_energy) / energy_sum_inverse)
                out_energy_mem[neigh_x, neigh_y] += energy_share
        else:
            out_energy_mem[i, j] += mem[0, inds.energy, i, j]

@ti.kernel
def distribute_energy(mem: ti.types.ndarray(), out_energy: ti.types.ndarray(), max_energy: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        if mem[0, inds.energy, i, j] > max_energy:
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                out_energy[neigh_x, neigh_y] += (mem[0, inds.energy, i, j] / kernel.shape[0])
        else:
            out_energy[i, j] += mem[0, inds.energy, i, j]

@ti.kernel
def flow_energy_up(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                      kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        infra_sum = 0.0
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            infra_sum += mem[0, inds.infra, neigh_x, neigh_y]
        if infra_sum > 1e-8:
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                neigh_infra = mem[0, inds.infra, neigh_x, neigh_y]
                out_energy_mem[neigh_x, neigh_y] += (
                    central_energy * (neigh_infra / infra_sum))
        else:
            out_energy_mem[i, j] += central_energy
            
@ti.kernel
def distribute_infra(mem: ti.types.ndarray(), out_infra: ti.types.ndarray(), max_infra: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        if mem[0, inds.infra, i, j] > max_infra:
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                out_infra[neigh_x, neigh_y] += (mem[0, inds.infra, i, j] / kernel.shape[0])
        else:
            out_infra[i, j] += mem[0, inds.infra, i, j]
    

_energy_scratch = {}

def energy_physics(substrate, kernel, max_infra, max_energy):
    inds = substrate.ti_indices[None]
    grid_shape = substrate.mem[0, inds.energy].shape
    device = substrate.torch_device

    if "energy_out" not in _energy_scratch or _energy_scratch["energy_out"].shape != grid_shape:
        _energy_scratch["energy_out"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=device)
        _energy_scratch["infra_out"] = torch.zeros(grid_shape, dtype=substrate.mem.dtype, device=device)
    
    _energy_scratch["energy_out"].zero_()
    flow_energy_up(substrate.mem, _energy_scratch["energy_out"], kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = _energy_scratch["energy_out"]

    _energy_scratch["energy_out"].zero_()
    distribute_energy(substrate.mem, _energy_scratch["energy_out"], max_energy, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = _energy_scratch["energy_out"]

    _energy_scratch["infra_out"].zero_()
    distribute_infra(substrate.mem, _energy_scratch["infra_out"], max_infra, kernel, substrate.ti_indices)
    substrate.mem[0, inds.infra] = _energy_scratch["infra_out"]


def invest_liquidate(substrate):
    inds = substrate.ti_indices[None]
    energy = substrate.mem[0, inds.energy]
    infra = substrate.mem[0, inds.infra]
    invest = substrate.mem[0, inds.acts_invest]
    liquidate = substrate.mem[0, inds.acts_liquidate]
    delta = invest * energy - liquidate * infra
    substrate.mem[0, inds.energy] -= delta
    substrate.mem[0, inds.infra] += delta