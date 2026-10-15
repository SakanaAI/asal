
import numpy as np
from scipy.stats import uniform
from scipy.stats import levy_stable


def levy_dust(shape: tuple, points: int, alpha: float, beta: float, pad: int = 1) -> np.array:
    """Generate a levy dust cloud of points in a given shape padded by pad

    Parameters:
        shape (tuple): Shape of the grid
        points (int): Number of points in the cloud
        alpha (float): Alpha parameter of the levy distribution
        beta (float): Beta parameter of the levy distribution
        pad (int): Padding of the grid

    Returns:
        np.array: Levy dust cloud represented as a list of points
    """
    # uniformly distributed angles
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi)

    # Levy distributed step length
    r = abs(levy_stable.rvs(alpha, beta, size=points))

    x = np.cumsum(r * np.cos(angle)) % (shape[0]-pad)
    y = np.cumsum(r * np.sin(angle)) % (shape[1]-pad)

    return np.array([x, y])


def discretize_levy_dust(dust: np.array, shape: tuple, pad: int = 1) -> np.array:
    """Discretize a levy dust cloud into a grid of shape shape,
    such that each position in the grid is the number of points in the cloud that fall in that position

    Parameters:
        dust (np.array): Levy dust cloud
        shape (tuple): Shape of the grid
        pad (int): Padding of the grid

    Returns:
        np.array: Grid of shape shape representing the density of points in the dust cloud
    """
    dust = np.array(dust, dtype=np.int64)
    print(dust.shape)
    dust, density = np.unique(dust, axis=0, return_counts=True)

    channel = np.zeros(shape)
    channel[dust[:, 0]+pad, dust[:, 1]+pad] = density

    return channel

# def add_noise_to_ch(self, chs=[None], magnitude=0.1):
#     if chs[0] is None:
#         chs = [self.hidden_i]
#     new_grid = (np.random.random(self.channels[chs].size)
#                 * magnitude).reshape((len(chs), *self.eshape[1:]))
#     self.channels[chs] += new_grid
