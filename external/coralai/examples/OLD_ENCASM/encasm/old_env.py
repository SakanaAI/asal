from operator import ge
from re import A
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import uniform
from scipy.stats import levy_stable
from utils.bcolors import bcolors

'''
Defines a cellular automata environment for an agent to interact with.
Is responsible for generating food and life, and for updating the environment
based on the agent's actions.
'''


class CAEnvironment:
    # put into a config file, save with each run
    food_i = 0  # Channel index of food
    life_i = 1  # Channel index of life
    hidden_i = 2  # Index of first hidden channel

    n_hidden = 4
    n_channels = 2 + n_hidden

    esize = 32  # Width/Height of environment
    pad = 1
    cutsize = esize-2*pad
    eshape = (n_channels, esize, esize)
    channels = np.zeros(eshape)

    alpha = 1  # Alpha for levy food distribution
    beta = 1  # Beta for levy food distribution
    food_amt = 16  # Number of food points created
    max_food = 4  # Densest food source -- necessary?

    vid_channels = (food_i, life_i)
    vid_cmaps = [None] * 2

    frames = None

    def __init__(self, id):
        self.id = id

    # Initializes a new environment copied from the given one

    def __init__(self, env):
        self.id = env.id
        self.food_i = env.food_i
        self.life_i = env.life_i
        self.hidden_i = env.hidden_i
        self.n_hidden = env.n_hidden
        self.n_channels = env.n_channels
        self.esize = env.esize
        self.cutsize = env.cutsize
        self.eshape = env.eshape
        self.channels = np.copy(env.channels)

        self.alpha = env.alpha
        self.beta = env.beta
        self.food_amt = env.food_amt
        self.max_food = env.max_food

    def update_shape(self, new_shape, n_hidden_chs=4):
        '''
        Shape is in form (n_channels, width, height)
        Resets all channels
        '''
        if new_shape[1] != new_shape[2]:
            # TODO: Update these to raise value errors etc.
            print(bcolors.WARNING +
                  "ca_environment.py:update_shape: Unknown behavior for non-square envs" + bcolors.ENDC)
        if n_hidden_chs >= new_shape[0]:
            print(bcolors.WARNING +
                  "ca_environment.py:update_shape: Shape must have at least n_hidden_chs+1 channels" + bcolors.ENDC)
            return
        if (new_shape[0] - n_hidden_chs) == 1:
            self.hidden_i = 1
            self.life_i = None

        self.n_channels = new_shape[0]
        self.n_hidden = n_hidden_chs
        self.hidden_i = new_shape[0] - n_hidden_chs
        if self.hidden_i < 2:
            self.life_i = None
            if self.hidden_i == 0:
                self.food_i = None
            else:
                self.food_i = 0
        self.esize = new_shape[1]
        self.cutsize = self.esize-2*self.pad
        self.eshape = new_shape
        self.channels = np.zeros(self.eshape)

    def norm_center(self, x, w):
        x -= x.min()
        x *= w/x.max()
        return x

    def add_noise_to_ch(self, chs=[None], magnitude=0.1):
        if chs[0] is None:
            chs = [self.hidden_i]
        new_grid = (np.random.random(self.channels[chs].size)
                    * magnitude).reshape((len(chs), *self.eshape[1:]))
        self.channels[chs] += new_grid

    def get_levy_dust(self, shape: tuple, points: int, alpha: float, beta: float) -> np.array:
        # uniformly distributed angles
        angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi)

        # Levy distributed step length
        r = abs(levy_stable.rvs(alpha, beta, size=points))

        x = np.cumsum(r * np.cos(angle)) % (shape[0]-1)
        y = np.cumsum(r * np.sin(angle)) % (shape[1]-1)

        return np.array([x, y])

    # Generates a levy dust cloud and adds it to the environment to represent
    # an ecologically viable distribution of food
    def gen_padded_food(self) -> np.array:
        dust = self.get_levy_dust(
            (self.esize-self.pad*2, self.esize-self.pad*2), self.food_amt, self.alpha, self.beta).T
        dust = np.array(dust, dtype=np.int64)
        dust, density = np.unique(dust, axis=0, return_counts=True)

        self.channels[self.food_i, dust[:, 0]+1, dust[:, 1]+1] = density
        return self.channels

    def img_to_grid(self, image):
        img = Image.open(image)
        img = np.asarray(img)
        mask = img[..., -1] != 0
        a = np.zeros(mask.shape)
        a[mask] = 1
        return a

    # Used to load a grid from an image representing an channel
    def set_channel(self, i, grid):
        if isinstance(grid, str):
            grid = self.img_to_grid(grid)
        self.channels[i] = grid

    # Innoculate the cell with the most food - this is likely to be surrounded
    # by more food given def of levy dust

    def innoculate(self):
        maxfood = np.unravel_index(
            np.argmax(self.channels[self.food_i]), self.channels[self.food_i].shape)

        self.channels[self.life_i][maxfood[0], maxfood[1]] = 1

    def update_chunk(self, i, j, update):
        self.channels[:, i, j] = update

    def display(self, channels=None, cmaps=None):
        if channels is None:
            channels = (0, 1)
        if cmaps is None:
            cmaps = (cm.copper, cm.gray)

        for i in range(len(channels)):
            fig, axs = plt.subplots(ncols=1, figsize=(12, 6))
            if i >= len(cmaps):
                cmap = cm.gray
            else:
                cmap = cmaps[i]
            im = axs.matshow(self.channels[channels[i]], cmap=cmap)
            fig.colorbar(im, fraction=0.045, ax=axs)
            axs.set_title(self.id)
