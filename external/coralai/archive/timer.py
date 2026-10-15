import torch
import taichi as ti

ti.init(ti.gpu)
device = torch.device('mps:0')

depth_distr1 = torch.randn((128,128,32), device=device)
depth_distr2 = depth_distr1.clone()

@ti.kernel
def distribute(mem: ti.types.ndarray()):
    sum_total = 0.0
    for i,j,k in ti.ndrange(mem.shape[0], mem.shape[1], mem.shape[2]):
        sum_total += mem[i, j, k]
    for i,j,k in ti.ndrange(mem.shape[0], mem.shape[1], mem.shape[2]):
        mem[i, j, k] /= sum_total

from timeit import default_timer as timer

print(depth_distr1.sum())
print(depth_distr2.sum())

start = timer()
distribute(depth_distr1)
print(depth_distr1.sum())
end = timer()
print(end-start)

start = timer()
sum_half = torch.sum(depth_distr2, dim=2)
depth_distr2.div_(sum_half.unsqueeze(2))
print(depth_distr2.sum())
end = timer()
print(end-start)