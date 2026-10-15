import taichi as ti

from ...dynamics.Organism import Organism

class DumbOrg(Organism):
    def __init__(self, world):
        super(DumbOrg, self).__init__(world, [], 0)
        self.initialized = False

    def forward(self, x): 
        inds = self.world.ti_indices[None]
        if not self.initialized:
            self.initialized = True
            height = self.world.h
            width = self.world.w
            for x in range(width):
                self.world.mem[0, inds.infra, x, height//2] = x / width
