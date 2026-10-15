import torch
import taichi as ti
from ..substrate.substrate import Substrate

def regen_ports(ports, period, port_id_map, resources):
    # TODO: take into account resource size and obstacles?
    port_id_map = ports.metadata["port_id_map"]
    resources = ports.metadata["resources"]
    for resource in resources:
        ports.contents.squeeze(0)[port_id_map == resource.id] += resource.regen_func(period)
        torch.clamp(ports.contents.squeeze(0)[port_id_map == resource.id],
                    ports.allowed_range[0], ports.allowed_range[1],
                    out=ports.contents.squeeze(0)[port_id_map == resource.id])
    num_regens = ports.metadata.get("num_regens", 0)
    ports.metadata.update({"num_regens": num_regens + 1})


@ti.func
def grow_muscle_csa_ti(capital:         ti.f32,
                       muscle_radius:   ti.f32,
                       radius_delta:    ti.f32,
                       growth_eff:      ti.f32,
                       capital_density: ti.f32):
    ret_delta_cap = 0.0
    ret_delta_rad = 0.0
    # assert capital >= 0, "Capital cannot be negative (before growth)"
    csa_delta = (muscle_radius + radius_delta)**2 - muscle_radius**2
    if csa_delta < 0:
        ret_delta_cap = csa_delta * capital_density * growth_eff
        ret_delta_rad = ti.math.sqrt(csa_delta)
    if csa_delta > 0:
        cap_needed = csa_delta * capital_density
        if cap_needed > capital:
            cap_needed = capital
            csa_delta = capital / capital_density
        ret_delta_cap = cap_needed
        ret_delta_rad = csa_delta * ti.math.sign(radius_delta)
    return ret_delta_cap, ret_delta_rad


def activate_muscles(muscle_radii: torch.Tensor, activations: torch.Tensor):
    return (torch.sign(muscle_radii) * (muscle_radii ** 2)).mul(activations)    # Activation is a percentage (kinda) of CSA

@ti.func
def activate_port_muscles_ti(capital:            ti.f32,
                             port:               ti.f32,
                             obstacles:          ti.f32,
                             port_muscle_radius: ti.f32,
                             port_activation:    ti.f32,
                             capital_per_work:   ti.f32):
    port_activation *= (1-obstacles)
    desired_delta = port_muscle_radius ** 2 * port_activation * ti.math.sign(port_muscle_radius)
    delta_capital = 0.0
    delta_port = 0.0
    if desired_delta < 0: # Deposition of capital into port system
        desired_delta = ti.math.min(-desired_delta, capital)
        delta_capital = desired_delta
        delta_port = desired_delta - desired_delta * capital_per_work # TODO: Can this be negative?
    if desired_delta > 0: # Extraction of capital (or poison) from port system
        desired_delta = ti.math.min(desired_delta, abs(port))
        delta_port = -desired_delta * ti.math.sign(port)
        delta_capital = desired_delta - desired_delta * capital_per_work
    return delta_capital, delta_port

@ti.func
def activate_mine_muscles_ti(capital:               ti.f32,
                             obstacles:             ti.f32,
                             waste:                 ti.f32,
                             mine_muscle_radius:    ti.f32,
                             mine_activation:       ti.f32,
                             capital_per_work:      ti.f32):
    desired_delta = mine_muscle_radius ** 2 * mine_activation * ti.math.sign(mine_muscle_radius)
    desired_delta = ti.math.min(desired_delta, capital/capital_per_work)
    delta_capital = 0.0
    delta_obstacle = 0.0
    delta_waste = 0.0
    if desired_delta < 0: # Deposition of waste into obstacles
        desired_delta = ti.math.min(-desired_delta, waste)
        delta_obstacle = desired_delta
        delta_waste = -desired_delta
        delta_capital = -desired_delta * capital_per_work
    if desired_delta > 0: # Extraction of obstacle into waste (pumpable)
        desired_delta = ti.math.min(desired_delta, obstacles)
        delta_obstacle = -desired_delta
        delta_waste = desired_delta
        delta_capital = desired_delta * capital_per_work
    return delta_capital, delta_obstacle, delta_waste
                  
                  

def activate_flow_muscles(world: Substrate, flow_kernel, flow_cost):
    capital = world['capital'].squeeze(2)
    waste = world['waste'].squeeze(2)
    obstacles = world['obstacle'].squeeze(2)
    flow_muscle_radii = world[('muscles', 'flow')].permute(2, 0, 1)
    flow_activations = world[('muscle_acts', 'flow')].squeeze(2)

    total_capital_before = capital.sum()
    assert capital.min() >= 0, "Capital cannot be negative (before flow)"

    # Enforce obstacles
    flow_activations *= (1-obstacles)

    # Invert negative flows across kernel
    roll_shift = (flow_kernel.shape[0]-1)//2
    flows = activate_muscles(flow_muscle_radii, flow_activations)
    positive_flows = torch.where(flows[1:] > 0, flows[1:], torch.zeros_like(flows[1:]))
    negative_flows = torch.where(flows[1:] < 0, flows[1:], torch.zeros_like(flows[1:]))
    flows[1:] = positive_flows.sub(torch.roll(negative_flows, roll_shift, dims=0))
    flows[0] = torch.abs(flows[0]) # Self flow is always positive
    del positive_flows, negative_flows

    # Distribute cytoplasm AND WASTE across flows
    total_flow = torch.sum(flows, dim=0)
    total_flow_desired_safe = torch.where(total_flow == 0, torch.ones_like(total_flow), total_flow)
    flow_distribution = flows.div(total_flow_desired_safe.unsqueeze(0))
    del total_flow_desired_safe

    max_flow_possible = torch.min(capital + waste, capital.div(1+flow_cost))
    total_flow = torch.where(total_flow > max_flow_possible, max_flow_possible, total_flow)
    capital.sub_(total_flow.mul(flow_cost))  # enforce cost of flow before distributing

    mass = capital + waste
    percent_waste = waste.div(torch.where(mass == 0, torch.ones_like(mass), mass))
    del mass
    waste_out = percent_waste * total_flow
    capital_out = total_flow - waste_out
    capital.sub_(capital_out)
    waste.sub_(waste_out)

    capital_flows = flow_distribution.mul(capital_out.unsqueeze(0))
    waste_flows = flow_distribution.mul(waste_out.unsqueeze(0))
    del waste_out, capital_out

    received_capital = torch.sum(
                            torch.stack(
                                [torch.roll(capital_flows[i], shifts=tuple(map(int, flow_kernel[i])), dims=[0, 1])
                                for i in range(flow_kernel.shape[0])]
                            ), dim=0
                        )
    received_waste = torch.sum(
                            torch.stack(
                                [torch.roll(waste_flows[i], shifts=tuple(map(int, flow_kernel[i])), dims=[0, 1])
                                for i in range(flow_kernel.shape[0])]
                            ), dim=0
                        )
    del capital_flows, waste_flows
    capital.add_(received_capital)
    waste.add_(received_waste)
    del received_capital, received_waste

    # capital = torch.where(capital < 0.01, cfg.zero_tensor, capital) # should be non-negative before this, just for cleanup

    assert capital.min() >= 0, "Capital cannot be negative (after flow)"
    capital_diff = capital.sum() - total_capital_before
    assert capital_diff <= 0, f"Oops, capital was invented during flow. Diff: {capital_diff}"




# -----------



# def activate_mine_muscles(sim: Simulation,
#         capital_ch: Channel, obstacles_ch: Channel, waste_ch: Channel,
#         mine_muscle_radii_ch: Channel, mine_activation_ch: Channel, metadata: dict):
    
#     mining_cost = metadata["mining_cost"]
#     capital = capital_ch.contents.squeeze(0)
#     obstacles = obstacles_ch.contents.squeeze(0)
#     waste = waste_ch.contents.squeeze(0)
#     mine_muscle_radii = mine_muscle_radii_ch.contents.squeeze(0)
#     mine_activation = mine_activation_ch.contents.squeeze(0)

#     assert capital.min() >= 0, "Capital cannot be negative (before mine)"
#     desired_delta = activate_muscles(mine_muscle_radii, mine_activation)
#     torch.clamp(desired_delta,
#                 min=torch.max(-waste, -capital/mining_cost),
#                 max=torch.min(obstacles, capital/mining_cost),
#                 out=desired_delta)
#     capital -= desired_delta*mining_cost
#     obstacles -= desired_delta
#     waste += desired_delta
#     torch.clamp(capital, min=0, out=capital)

#     assert capital.min() >= 0, "Capital cannot be negative (after mine)"
#     assert obstacles.min() >= 0, "Obstacle cannot be negative (after mine)"

# def grow_muscle_csa(capital, muscle_radii, radii_deltas, growth_cost):
#     # TODO: Update to work on batches of environments (simple)
#     # TODO: Cost should be able to be lower than 1, right now it is 1:1 + cost

#     assert capital.min() >= 0, "Capital cannot be negative (before growth)"

#     csa_deltas = (muscle_radii + radii_deltas)**2 - muscle_radii**2  # cross-sectional area

#     negative_csa_deltas = torch.where(csa_deltas < 0, csa_deltas, torch.zeros_like(csa_deltas))
#     positive_csa_deltas = torch.where(csa_deltas > 0, csa_deltas, torch.zeros_like(csa_deltas))

#     # Atrophy muscle and convert to capital
#     new_csa_mags = muscle_radii**2.0
#     total_capital_before = capital.sum() + new_csa_mags.sum()
#     new_csa_mags.add_(negative_csa_deltas)

#     capital.sub_(torch.sum(negative_csa_deltas, dim=0).mul(1-growth_cost)) # gaining capital from atrophy

#     # Grow muscle from capital, if possible
#     total_csa_deltas = torch.sum(positive_csa_deltas, dim=0)
#     total_csa_deltas_safe = torch.where(total_csa_deltas == 0, torch.ones_like(total_csa_deltas), total_csa_deltas)
#     csa_delta_distribution = positive_csa_deltas.div(total_csa_deltas_safe.unsqueeze(0))
#     del total_csa_deltas_safe # TODO: Free CUDA memory? Garbage Collect?  

#     capital_consumed = torch.where(total_csa_deltas > capital, capital, total_csa_deltas)
#     capital.sub_(capital_consumed)
#     capital_consumed.mul_(1-growth_cost)
#     # other than inefficiency, exchange rate between capital and muscle area is 1:1 
#     new_csa_mags.addcmul_(capital_consumed.unsqueeze(0), csa_delta_distribution) 
#     # This is allowed because even with no available capital a muscle may invert its polarity or go to 0 at only a cost of inefficiency
#     torch.copysign(torch.sqrt(new_csa_mags), muscle_radii.add(radii_deltas), out=muscle_radii) 

#     capital = torch.where(capital < 0.001, torch.zeros_like(capital), capital)
#     assert capital.min() >= 0, "Oops, a cell's capital became negative during growth"
#     capital_diff = (capital.sum() + new_csa_mags.sum()) - total_capital_before
#     assert capital_diff <= 0, f"Oops, capital was invented during growth. Diff: {capital_diff}"


# def activate_port_muscles(
#         sim: Simulation, capital_ch: Channel, ports_ch: Channel, obstacles_ch: Channel,
#         port_muscle_radii_ch: Channel, port_activations_ch: Channel, metadata: dict):
#     # TODO: REALLY? NEGATIVE CAPITAL
#     port_cost = metadata["port_cost"]
#     capital = capital_ch.contents.squeeze(0)
#     ports = ports_ch.contents.squeeze(0)
#     port_muscle_radii = port_muscle_radii_ch.contents.squeeze(0)
#     port_activations = port_activations_ch.contents.squeeze(0)

#     port_activations *= (1-obstacles_ch.contents.squeeze(0))
    
#     assert capital.min() >= 0, "Capital cannot be negative (before port)"
#     desired_delta = activate_muscles(port_muscle_radii, port_activations)
#     torch.clamp(desired_delta,
#                 min=-capital.div(1/port_cost),
#                 max=torch.min(capital/port_cost, torch.abs(ports)), # Poison costs the same to pump out
#                 out=desired_delta)
#     capital -= desired_delta * port_cost
#     capital += torch.copysign(desired_delta,torch.sign(ports)) # can produce negative capital
#     torch.clamp(capital, min=0, out=capital) # NOTE: This is a hack to prevent negative capital, fix later
#     ports -= desired_delta
#     del desired_delta
#     assert capital.min() >= 0, "Capital cannot be negative (after port)"