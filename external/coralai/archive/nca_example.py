import taichi as ti
import torch

from coralai.nca import NCA
from coralai.simulation.vis_old import Vis

ti.init(ti.gpu)  
ein = NCA(shape=(400, 400), torch_device=torch.device("mps"))

w = ein.world

vis = Vis(w, [('com', 'r'), ('com', 'g'), ('com', 'b')])

while vis.window.running:
    vps = vis.params[None]
    if vps.is_perturbing_weights:
        ein.organism.perturb_weights(vps.perturb_strength)
    if not vps.drawing:
        ein.world.mem = ein.organism.forward(ein.world.mem)
    vis.update()
