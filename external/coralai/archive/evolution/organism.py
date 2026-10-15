import torch

class Organism:
    def __init__(self, substrate, kernel, sense_chs, act_chs, torch_device):
        self.substrate = substrate
        self.kernel = torch.tensor(kernel, device=torch_device)
        self.torch_device = torch_device
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)
        self.is_evolvable = False

    def forward(self, x):
        return x
    
    def mutate(self):
        return self