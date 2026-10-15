import taichi as ti
import torch

cs = ti.types.struct(**{
    'a': ti.f32,
    'b': ti.types.vector(n=5, dtype=ti.f32),
    'c': ti.i8})


chshape = (4,4)
cfield = cs.field(shape=chshape)
cfield = cfield.to_torch()
blah = ti.field(ti.f32, shape=chshape)
f2 = ti.field(ti.f32, shape=chshape)

mem = torch.empty((*chshape,7), dtype = torch.float32)
mem[:,:,0] = cfield['a']
mem[:,:,1:6] = cfield['b']
mem[:,:,6] = f2.to_torch()

@ti.kernel
def memit(mem: ti.types.ndarray()):
    for i, j in ti.ndrange(chshape[0], chshape[1]):
        for k in ti.static(range(7)):
            mem[i,j,k] = ti.cast(k * i / j, ti.f32)

memit(mem)
print(mem)