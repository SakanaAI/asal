import taichi as ti
import torch

@ti.func
def ReLU(x):
    return x if x > 0 else 0

@ti.func
def sigmoid(x):
    return 1 / (1 + ti.exp(-x))

@ti.func
def inverse_gaussian(x):
    return -1./(ti.exp(0.89*ti.pow(x, 2.))+1.)+1.

def ch_norm(input_tensor):
    # Calculate the mean across batch and channel dimensions
    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)

    # Calculate the variance across batch and channel dimensions
    var = input_tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

    # Normalize the input tensor
    input_tensor.sub_(mean).div_(torch.sqrt(var + 1e-5))

    return input_tensor