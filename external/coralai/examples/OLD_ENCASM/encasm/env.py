import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from configparser import SafeConfigParser
import encasm.utils as utils


class CAEnvironment:
    """A Petri dish is a multi channeled container for populations of cells.

    Attributes:
        map (np.array): A 2d array (W x H) of the map of the environment, where:
            0: empty
            1: food
            2: water
            3: poison
            4: sink
        life (np.array): A 2d array (W x H) of the life of the environment, where:
            0: empty
            1: alive
        resv (np.array): A 2d array (W x H) of the resevoir of life cells
        hidden (np.array): A 3d array (N x W x H) of the hidden channels of the environment

        attributes stored in the config file:
            width (int): The width of the environment
            height (int): The height of the environment
            n_hidden (int): The number of hidden channels
            food_amt (int): The amount of food to generate
            alpha (float): The alpha parameter for the Lévy dust distribution
            beta (float): The beta parameter for the Lévy dust distribution
            pad (int): The padding for the Lévy dust distribution
    """
   
    def __init__(self, id: str, config: SafeConfigParser):
        """Initializes a new Petri dish from a SafeConfigParser object.

        Parameters:
            config (SafeConfigParser): The SafeConfigParser object
        """
         # Index indicates the value present in map
        self.map_keys = ["empty", "food", "water", "poison", "sink"]
        # black, green, blue, red, purple hex
        self.map_colors = ["#000000", "#00FF00", "#0000FF", "#FF0000", "#800080"]
        # black, pastel green, pastel blue, pastel red, pastel purple
        # map_colors = ["#000000", "#b2f2b2", "#b2d8f2", "#f2b2b2", "#f2b2f2"]
        # map_colors = [""#77dd77", "#aec6cf", "#ff6961", "#dda0dd"]
        self.map_cmap = ListedColormap(self.map_colors)
        self.id = id
        self.config = config

        self.init_channels()

        
    
    @classmethod
    def from_config_file(cls, id: str, config_file: str):
        """Creates a new Petri dish from a config file.

        Parameters:
            config_file (str): The path to the config file
        """
        parser = SafeConfigParser()
        parser.read(config_file)
        return cls(id, parser["Environment"])
    
    @classmethod
    def from_env(cls, new_id: str, other_env: 'CAEnvironment'):
        """Creates a new Petri dish from an existing environment.

        Parameters:
            env (PetriDish): The environment to copy
        """
        env = cls(new_id, other_env.config)
        env.map = np.copy(other_env.map)
        env.life = np.copy(other_env.life)
        env.resv = np.copy(other_env.resv)
        env.hidden = np.copy(other_env.hidden)
        env.food = other_env.fd()
        env.water = other_env.wt()
        env.poison = other_env.ps()
        env.sink = other_env.sk()

        return env

    @classmethod
    def from_channels(cls, id: str, channels: dict, config: SafeConfigParser):
        """Creates a new Petri dish from a dictionary of channels.

        Parameters:
            channels (dict): A dictionary of channels
        """
        env = cls(id, config)
        for ch, grid in channels.items():
            env.set_channel(ch, grid)
        return env

    def fd(self):
        return self.map == self.map_keys.index("food")

    def wt(self):
        return self.map == self.map_keys.index("water")
    
    def ps(self):
        return self.map == self.map_keys.index("poison")
    
    def sk(self):
        return self.map == self.map_keys.index("sink")
    
    def init_channels(self):
        """Initializes the channels of the environment from the config dictionary."""
        self.width = self.config.getint("width", 32)
        self.height = self.config.getint("height", 32)
        self.n_hidden = self.config.getint("n_hidden", 4)
        self.n_channels = self.n_hidden + 3
        self.map = np.zeros((self.width, self.height))
        self.food = np.zeros((self.width, self.height))
        self.water = np.zeros((self.width, self.height))
        self.poison = np.zeros((self.width, self.height))
        self.sink = np.zeros((self.width, self.height))
        self.life = np.zeros((self.width, self.height))
        self.resv = np.zeros((self.width, self.height))
        self.hidden = np.zeros((self.n_hidden, self.width, self.height))

    def set_channel(self, ch: str, grid: np.array):
        """Sets the specified channel to the specified grid,
           if channel is in map_vals, updates values present in grid.

        Parameters:
            ch (str): The channel to set
            grid (np.array): The grid to set the channel to
        """
        if ch in self.map_keys:
            self.map[grid >= 1] = self.map_keys.index(ch)
            # sets attribute to grid
            setattr(self, ch, grid)
        else:
            # Otherwise just set the channel attribute
            setattr(self, ch, grid)

    def generate_food(self):
        """Generates a Lévy dust distribution of food in the environment
            as specified by the config dictionary.
        """
        dust = utils.levy_dust(
            (self.width, self.height),
            self.config.getint("food_amt", 16),
            self.config.getfloat("alpha", 1),
            self.config.getfloat("beta", 1),
            self.config.getint("pad", 1)
        )
        self.food += utils.discretize_levy_dust(
            dust, (self.width, self.height), self.config.getint("pad", 1))

    def display(self, chs: list = ["map", "life", "resv"], cmaps: list = ["copper", "gray", "hot"], cols: int = 3, retbuf: bool = False):
        """Displays the specified channels of the environment.

        Parameters:
            chs (list): The list of channels to display
            cmaps (list): The list of colormaps to use for each channel
            cols (int): The number of columns to display
            retbuf (bool): Whether to return the buffer or not
        """
        # Ensures each channel has a color, repeats the last color if necessary
        if len(cmaps) < len(chs):
            cmaps += [cmaps[-1]] * (len(chs) - len(cmaps))
        
        if len(chs) == 1:
            # If only one channel, just display it
            if chs[0] == "map":
                plt.matshow(getattr(self, chs[0]), cmap=self.map_cmap)
                plt.colorbar(fraction=0.045)
            else:
                plt.matshow(getattr(self, chs[0]), cmap=cmaps[0])
                plt.colorbar(fraction=0.045)

        else:
            rows = int(np.ceil(len(chs) / cols))
            # Displays a grid of matshow subplots for each channel, with the specified colormap
            # each plot with a title of the channel name and a colorbar
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            for i, ch in enumerate(chs):
                # adds padding between plots to account for colorbars
                fig.subplots_adjust(hspace=0.5, wspace=0.5)

                ax = axs[i]
                # If channel is map, use the map_colors
                if ch == "map":
                    ax.matshow(getattr(self, ch), cmap=ListedColormap(self.map_colors))
                else:
                    ax.matshow(getattr(self, ch), cmap=cmaps[i])
                ax.set_title(self.id + ": " + ch)
                # Creates a colorbar for each subplot that is the same size as the subplot with padding
                fig.colorbar(ax.images[0], ax=ax, fraction=0.045)
                # fig.colorbar(ax.images[0], ax=ax, pad=0.01)
            if retbuf:
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                return img_buf


# rows = int(np.ceil(len(chs) / cols))
# fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
# for i, ch in enumerate(chs):
#     ax = axs[i // cols, i % cols]
#     ax.matshow(getattr(self, ch), cmap=cmaps[i])
#     ax.set_title(ch + " channel, " + self.id)
#     ax.colorbar()
        