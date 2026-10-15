
# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import yaml

import importlib

import apply_physics
importlib.reload(apply_physics)

# %% Load Check Congif --------------------------------------------------------
def load_check_config(config_object):
    if isinstance(config_object, str):
        with open(config_object) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    elif isinstance(config_object, dict):
        config = config_object
    else:
        raise TypeError("config_object must be either a path (str) or a config (dict)")

    all_channels = (config["environment"]["channels"]
                    + config["physiology"]["channels"]
                    + config["physiology"]["perception_channels"]
                    + config["physiology"]["actuators"])

    # Ensure all perception and action channels exist
    for channel in config["physiology"]["perception_channels"] + config["physiology"]["actuators"]:
        assert channel in all_channels, f"Channel {channel} not found in all_channels"
        
    return config

def create_dumb_physiology(config):
    perception_channels = config["physiology"]["perception_channels"]
    actuators = config["physiology"]["actuators"]

    def dumb_physiology(perception):
        return np.random.random(len(actuators))
    
    return dumb_physiology


def act(config, input, physiology):
    return physiology(input)


def perceive(config, cell, env_channels, live_channels):
    kernel_type = config["physiology"]["perception_kernel"]["kernel_type"]

    if kernel_type == "moore":
        kernel_radius = config["physiology"]["perception_kernel"]["kernel_radius"]
        kernel = np.ones((kernel_radius*2+1, kernel_radius*2+1))
    elif kernel_type == "von_neumann":
        kernel = np.zeros((3, 3))
        kernel[1, :] = 1
        kernel[:, 1] = 1
    else:
        kernel = np.ones((3, 3))
    
    kernel = np.argwhere(kernel)

    perception_channels = config["physiology"]["perception_channels"]
    perception = np.zeros((len(perception_channels), kernel.shape[0], kernel.shape[1]))

    # TODO: Calculate all this beforehand
    for i, channel in enumerate(perception_channels):
        if config["environment"]["channels"].count(channel) > 0:
            perception[i] = env_channels[config["environment"]["channels"].index(channel)][cell[0], cell[1]]
        else:
            perception[i] = live_channels[config["physiology"]["channels"].index(channel)][cell[0], cell[1]]
    return perception


def stoppage_condition_met(config, update_num, init_conditions):
    if config["lifecycle"]["stoppage"]["condition"] == "iterations":
        if init_conditions["num_iterations"] is None:
            raise ValueError("Stoppage condition 'iterations' not initialized.")
        return update_num > init_conditions["num_iterations"]
    else:
        raise ValueError(f"Stoppage condition {config['lifecycle']['stoppage']['condition']} not recognized.")


def init_stoppage_condition(config):
    init_conditions = {}
    if config["lifecycle"]["stoppage"]["condition"] == "iterations":
        low = config["lifecycle"]["stoppage"]["stoppage_range"][0]
        high = config["lifecycle"]["stoppage"]["stoppage_range"][1]
        assert low < high, "Stoppage range must be increasing."
        num_iterations = np.random.randint(low, high)
        init_conditions["num_iterations"] = num_iterations
    return init_conditions


def identify_cells_to_update(config, env_channels, live_channels):
    # for now just all cells with storage above minimum
    storage_channel = live_channels[config["physiology"]["channels"].index("storage")]
    return np.argwhere(storage_channel > config["physics"]["min_storage"])


def run_lifecycle(config, env_channels, live_channels, physiology):
    """
    Perform the main lifecycle loop.
    """
    init_conditions = init_stoppage_condition(config)

    update_num = 0
    while not stoppage_condition_met(config, update_num, init_conditions):
        update_num += 1
        cells_to_update = identify_cells_to_update(config, env_channels, live_channels)
        
        for cell in cells_to_update:
            input = perceive(config, cell, env_channels, live_channels)
            output = act(config, input, physiology)
            apply_physics.apply_local_physics(config, cell, output, env_channels, live_channels) 
        
        # Visualization, if needed


def inoculate_env(config, env_channels, live_channels):
    obstacle_channel = env_channels[config["environment"]["channels"].index("obstacle")]
    poison_channel = env_channels[config["environment"]["channels"].index("poison")]
    food_channel = env_channels[config["environment"]["channels"].index("food")]

    valid_positions = np.logical_not(np.logical_or(obstacle_channel, poison_channel))
    
    # Choose where to place based on configuration
    if config["lifecycle"]["inoculation"]["where_to_place"] == "random":
        candidates = np.argwhere(valid_positions)
    elif config["lifecycle"]["inoculation"]["where_to_place"] == "on_food":
        candidates = np.argwhere(np.logical_and(food_channel, valid_positions))
    elif config["lifecycle"]["inoculation"]["where_to_place"] == "next_to_food":
        # Get positions adjacent to food
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        adj_to_food = signal.convolve2d(food_channel, kernel, mode='same') > 0
        candidates = np.argwhere(np.logical_and(adj_to_food, valid_positions))
    else:
        raise ValueError(f"Inoculation place method {config['lifecycle']['inoculation']['where_to_place']} not recognized.")

    # Selection procedure (for now, just random. More can be added)
    if config["lifecycle"]["inoculation"]["selection_procedure"] != "random":
        raise ValueError(f"Selection procedure {config['lifecycle']['inoculation']['selection_procedure']} not implemented yet.")

    selected_positions = candidates[np.random.choice(candidates.shape[0], config["lifecycle"]["inoculation"]["num_spores"], replace=False)]
    
    # Place the spores at selected_positions. More logic can be added to initialize spore properties like muscle, storage, etc.
    for position in selected_positions:
        # environment_matrix[some_spore_channel, position[0], position[1]] = some_initial_value
        # set storage to 1
        live_channels[config["physiology"]["channels"].index("storage"), position[0], position[1]] = 1


def init_live_channels(config):
    width = config["environment"]["width"]
    height = config["environment"]["height"]
    n_live_channels = len(config["physiology"]["channels"])
    live_channels = np.zeros((n_live_channels, width, height))
    return live_channels

    

def simulate_lifecycle(config_file, env_channels, physiology):
    config = load_check_config(config_file)
    live_channels = init_live_channels(config)
    inoculate_env(config, env_channels, live_channels)
    run_lifecycle(config, env_channels, live_channels, physiology)
    
    return env_channels, live_channels

if __name__ == "__main__":
    import visualize    
    importlib.reload(visualize)
    import pcg
    importlib.reload(pcg)
    # TEST: 0
    # ---------------
    env_channels = pcg.generate_env("./config.yaml", visualize=True)
    config = load_check_config("./config.yaml")
    live_channels = init_live_channels(config)
    inoculate_env(config, env_channels, live_channels)
    run_lifecycle(config, env_channels, live_channels, create_dumb_physiology(config))
    # ---------------


    # %%
    # KERNEL PLAYGROUND
    # 
    #  

    world_size = 20
    world = np.zeros((world_size, world_size))
    world[world_size//2, world_size//2] = 1


    # Cellular automata that spreads outwards
    kernel = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])

    # convolve world with kernel
    world = signal.convolve2d(world, kernel, mode='same')

    # visualize.show_image(visualize.channel_to_image(world, cmap="viridis")).show()



"""
- Inoculate
- All cells with storage present
	- Perceive neighborhood of
		- Obstacle
		- Chemoattractant
		- Chemorepellant
		- ^ environment; v life
		- Muscle
		- capital density/fill
		- Storage
		- Communication channels
	- Generate desired
		- Storage delta
		- Muscle contraction and
		- Reservoir gate open percentages
			- From my store, what percentage to give to (softmax)
				- left,right,up,down
		- Communication channels
- Apply Physics
- Repeat until stoppage parameter (config)
- Measure reproduction metric after iterations or weather cycle:
	- Amount of converted energy
		- Total volume of capital in your genome
	- Spore production
		- Given a signal, maximize capital volume

Cells decide where to shed capital by contracting their reservoirs, this is already described.
capital can flow into empty space (every cell is capable of carrying 1 or fewer units of capital,
even if it has no muscle to pump it.
"""
# %%