import random
from scipy import signal
import numpy as np
import torch
from coralai import (
    Simulation,
    Channel
)
from coralai.dynamics import ein_physics, pcg

# IDS
MUSCLES = 'all_muscle_radii'
FLOW_M = 'flow_muscle_radii'
PORT_M = 'port_muscle_radii'
MINE_M = 'mine_muscle_radii'
ALL_MUSCLE_ACT = 'all_muscle_activation'
FLOW_MACT = 'flow_muscle_activation'
PORT_MACT = 'port_muscle_activation'
MINE_MACT = 'mine_muscle_activation'
GROWTH_ACT = 'muscle_growth_activation'
COM = 'communication'
CAPITAL = 'capital'
WASTE = 'waste'
OBSTACLES = 'obstacles'
PORTS = 'ports'

num_communication_channels = 2

class EINCASM:
    def __init__(self, world_shape: tuple = (100, 100), verbose: bool = False):
        self.sim = Simulation.Simulation('EINCASM Experiment', world_shape, verbose=verbose)
        device = 'mps'
        if device == 'mps':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                raise ValueError("MPS is not available on this system")
        elif device == 'cuda':
            self.device = torch.device("cuda")
        elif device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.float_dtype = torch.float32

        kernel = torch.tensor([
            [0, 0],     # ORIGIN
            [-1, 0],    # UP
            [0, 1.0],   # RIGHT
            [1, 0],     # DOWN
            [0, -1]     # LEFT
        ], device=self.device, dtype=torch.int8)

        assert (kernel[0,0]==0).all(), "Kernel must have origin at index 0"
        assert kernel.shape[0] % 2 == 1, "Odd kernels (excluding origin) unimplemented"
        assert (torch.roll(kernel[1:], shifts=(kernel.shape[0]-1)//2, dims=0) - kernel[1:]).sum() == 0, "Kernel must be symmetric"
        self.kernel = kernel
        self.define_channels()


    def define_channels(self):
        self.sim.add_channel(COM, num_layers = num_communication_channels)
        self.sim.add_channel(MUSCLES, num_layers = self.kernel.shape[0] + 2, metadata = {'kernel': self.kernel})
        self.sim.add_subchannel(FLOW_M, MUSCLES, indices = list(range(self.kernel.shape[0])))
        self.sim.add_subchannel(PORT_M, MUSCLES, indices = [self.kernel.shape[0]])
        self.sim.add_subchannel(MINE_M, MUSCLES, indices = [self.kernel.shape[0]+1])
        self.sim.add_channel(ALL_MUSCLE_ACT, num_layers=3, allowed_range=[-1, 1])
        self.sim.add_subchannel(FLOW_MACT, ALL_MUSCLE_ACT, indices = 0)
        self.sim.add_subchannel(PORT_MACT, ALL_MUSCLE_ACT, indices = 1)
        self.sim.add_subchannel(MINE_MACT, ALL_MUSCLE_ACT, indices = 2)
        # flow, port, and mine muscles treated equally during growth
        self.sim.add_channel(GROWTH_ACT, num_layers = self.kernel.shape[0] + 2)
        self.sim.add_channel(CAPITAL, allowed_range = [0, 100])
        self.sim.add_channel(WASTE, allowed_range = [0, 100])
        self.sim.add_channel(OBSTACLES, init_func = pcg.init_obstacles_perlin)
        self.sim.add_channel(PORTS, init_func = pcg.init_ports_levy, 
                             allowed_range = [-1, 10],
                             metadata = {
                                 'num_resources': 2,
                                 'min_regen_amp': 0.5,
                                 'max_regen_amp': 2,
                                 'alpha_range': [0.4, 0.9],
                                 'beta_range': [0.8, 1.2],
                                 'num_sites_range': [2, 10]})

        self.sim.metadata.update({'period': 0.0})
        self.sim.add_rule('step_period',
            lambda sim, md: sim.metadata.update({'period': sim.metadata['period'] + 0.01}),
            req_sim_metadata = {'period': float})
        
        self.sim.add_rule('grow', ein_physics.grow_muscle_csa,
            input_channel_ids = [CAPITAL, MUSCLES, GROWTH_ACT],
            affected_channel_ids = [MUSCLES, CAPITAL],
            metadata = {'growth_cost': 0.2})
        
        self.sim.add_rule('flow', ein_physics.activate_flow_muscles,
            input_channel_ids = [CAPITAL, WASTE, OBSTACLES, FLOW_M, FLOW_MACT],
            affected_channel_ids = [CAPITAL],
            metadata = {'flow_cost': 0.2, 'kernel': self.kernel})
        
        self.sim.add_rule('eat', ein_physics.activate_port_muscles,
            input_channel_ids = [CAPITAL, PORTS, OBSTACLES, PORT_M, PORT_MACT],
            affected_channel_ids = [CAPITAL],
            metadata = {'port_cost': 0.2})
        
        self.sim.add_rule('dig', ein_physics.activate_mine_muscles,
            input_channel_ids = [CAPITAL, OBSTACLES, WASTE, MINE_M, MINE_MACT],
            affected_channel_ids = [CAPITAL, WASTE],
            metadata = {'mining_cost': 0.2})
        
        self.sim.add_rule('regen_resources', ein_physics.regen_ports,
            input_channel_ids = [PORTS], 
            affected_channel_ids = [PORTS],
            req_channel_metadata = {PORTS: ['port_id_map', 'port_sizes', 'resources']},
            req_sim_metadata = {'period': float})
        
        def seed_capital(sim, capital, ports, md):
            if capital.contents.sum() < md['threshold_sum']:
                coords = np.random.randint(0, capital.contents.shape[1], size=(md['num_sites'], 2))
                capital.contents[0, coords[:, 0], coords[:, 1]] = torch.rand(md['num_sites']) * 50 + 50

        self.sim.add_rule('seed_capital', seed_capital,
            input_channel_ids = [CAPITAL, PORTS],
            affected_channel_ids = [CAPITAL], metadata={
                "num_sites": 3,
                "threshold_sum": 2
            })

        self.sim.add_rule('random_agent', ein_physics.random_noise,
            input_channel_ids = [],
            affected_channel_ids = [GROWTH_ACT, ALL_MUSCLE_ACT, COM])

    def run(self):
        self.sim.init_all_channels()
        for _ in range(1000):
            self.sim.apply_all_rules()