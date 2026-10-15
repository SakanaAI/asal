import numpy as np
import noise
from scipy.stats import uniform
from scipy.stats import levy_stable
import torch
import random

def perlin2d(width, height, frequency=10.0, octaves=4, persistence=0.6,
                       lacunarity=3.0, x_offset=0, y_offset=0, channel=None, normalized=True):
    if channel is None:
        channel = torch.zeros((width, height))
    
    for x in range(width):
        for y in range(height):
            value = noise.pnoise2((x + x_offset) / frequency, (y + y_offset) / frequency, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            channel[x,y] = value
    
    if normalized:
        channel = (channel - channel.min()) / (channel.max() - channel.min())
    return channel

def discretize_levy_dust(shape: tuple, dust: np.array) -> np.array:
    """Discretize a levy dust cloud into a grid of shape shape,
    such that each position in the grid is the number of points in the cloud that fall in that position

    Returns:
        np.array: Grid of shape shape representing the density of points in the dust cloud
    """
    dust = np.array(dust, dtype=np.int64)
    points, density = np.unique(dust, axis=1, return_counts=True)
    out = np.zeros(shape)
    out[points[0,:], points[1,:]] = torch.from_numpy(density).float()
    return out

def levy_dust(shape: tuple, points: int, alpha: float, beta: float) -> np.array:
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi) 

    # Ensure alpha and beta are within the valid range for the levy_stable distribution
    alpha = max(min(alpha, 2), 0.1)
    beta = max(min(beta, 1), -1)

    step_length = abs(levy_stable.rvs(alpha, beta, size=points))

    x = np.cumsum(step_length * np.cos(angle)) % (shape[0])
    y = np.cumsum(step_length * np.sin(angle)) % (shape[1])

    return np.array([x, y])

def random_signal(num_components=2,
                    min_freq=np.pi/10, max_freq=np.pi*2, min_amp=0.1, max_amp=0.4,
                    min_start_period=0, max_start_period=np.pi*2):

    freqs = (max_freq - min_freq) * torch.rand(num_components) + min_freq
    # rots = (max_rot - min_rot) * torch.rand(num_signals).to(device) + min_rot
    amps = (max_amp - min_amp) * torch.rand(num_components) + min_amp
    start_periods = (max_start_period - min_start_period) * torch.rand(num_components) + min_start_period

    return lambda t: sum([amps[i] * torch.sin(freqs[i] * t + start_periods[i]) for i in range(num_components)]), freqs, amps, start_periods


def init_muscle_radii(shape: tuple, metadata: dict):
    return torch.zeros(shape, dtype=torch.float32), {}


def init_obstacles_perlin(shape: tuple, metadata: dict):
    # TODO: convert to taichi
    shape = shape[1:]
    # TODO: update to batches? 
    empty_threshold = metadata.get("empty_thresh", 0.4)
    full_threshold = metadata.get("full_thresh", 0.6)
    frequency = metadata.get("frequency", 15)
    octaves = metadata.get("octaves", 9)
    persistence = metadata.get("persistence", 0.6)
    lacunarity = metadata.get("lacunarity", 1.5)

    x_offset = random.randint(0, 10000)
    y_offset = random.randint(0, 10000)

    obstacles = perlin2d(shape[0], shape[1], frequency, octaves, persistence,
                    lacunarity, x_offset, y_offset, normalized=True)
    torch.where(obstacles > full_threshold, torch.tensor(1), obstacles, out=obstacles)
    torch.where(obstacles < empty_threshold,torch.tensor(0), obstacles, out=obstacles)

    return obstacles.unsqueeze(0), {"empty_threshold": empty_threshold,
                        "full_threshold": full_threshold,
                        "frequency": frequency,
                        "octaves": octaves,
                        "persistence": persistence,
                        "lacunarity": lacunarity,
                        "x_offset": x_offset,
                        "y_offset": y_offset}

class Resource:
    # a resource as an id (incremental), a min and max,
    # regeneration and dispersal functions of time,
    # a distribution function (levy dust usually)

    # a resource exists in the resource map as an id on a given cell along with a value
    # the id determines how to update it using the respective functions in this class
    def __init__(self, id, regen_func, metadata=None, dispersal_func=None):
        self.id = id
        self.regen_func = regen_func
        default_metadata = {
            'id': id
            }
        if metadata is None:
            metadata = {}
        metadata.update(default_metadata)
        self.metadata = metadata
        self.dispersal_func = dispersal_func

def init_ports_levy(shape: tuple, metadata: dict):
    shape = shape[1:]
    port_id_map = torch.zeros(shape, dtype=torch.int8)
    port_sizes = torch.zeros(shape)
    resources = []

    for port_id in range(1,metadata["num_resources"]+1):
        # regen_func, freqs, amps, start_periods
        signal_info = random_signal(
        min_amp = metadata["min_regen_amp"], max_amp=metadata["max_regen_amp"])
        regen_func = signal_info[0]

        resource = Resource(port_id, regen_func)
        resources.append(resource)
        alpha = random.uniform(*metadata["alpha_range"])
        beta = random.uniform(*metadata["beta_range"])
        num_sites = random.randint(*metadata["num_sites_range"])

        dust = levy_dust(shape, num_sites, alpha, beta)
        dust = discretize_levy_dust(shape, dust)
        port_id_map[dust > 0] = port_id
        port_sizes += dust

        resource.metadata.update({'regen_signal_info': {
                                        "frequencies": signal_info[1].tolist(),
                                        "amplitudes": signal_info[2].tolist(),
                                        "start_periods": signal_info[3].tolist()
                                        },
                                    'alpha': alpha,
                                    'beta': beta,
                                    'num_sites': num_sites,
                                    'object': resource})
    port_metadata = {
        'port_id_map': port_id_map,
        'port_sizes': port_sizes,
        'resources': resources
    }
    for resource in resources:
        port_metadata[f"resource_{resource.id}_init_info"] = resource.metadata

    return torch.zeros((1, *shape)), port_metadata
