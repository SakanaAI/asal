import re
import os
import configparser
import numpy as np
from PIL import Image
from encasm.env import PetriDish
import matplotlib.pyplot as plt


map_channels = {
    "F": "food",
    "L": "life",
    "P": "poison",
    "W": "water",
    "S": "sink",
}


def get_env_config(folder):
    # Parses the first .config file contained in the given folder,
    # returns a tuple of the environment shape and the config object
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.name.endswith(b".config"):
            config = configparser.ConfigParser()
            config.read(entry.path.decode())
            # (int(config["Environment"]["width"]), int(config["Environment"]["height"])),
            return config


def img_to_grid(img):
    # Converts an image to a grid of 0s and 1s
    img = np.asarray(Image.open(img))
    if len(img.shape) > 2:
        img = img[:, :, -1]  # Get the alpha channel
    return img > 0


def gen_env_dict(folder, config):
    # Loads folder into a dictonary of environments of the given shape

    # Pattern matches files like F_0-T_1.png where 01 would be the environment key, T would be the channel type of the image
    # These are returned in groups
    pattern = re.compile(r"(\d+)-([A-Z])(_(\d+))?.png")
    envs = {}
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.name.endswith(b".png"):
            match = pattern.match(entry.name.decode())
            if match:
                map_key = match.group(1)
                terrain_type = map_channels[match.group(2)]
                test_n = match.group(4)
                env_key = map_key + \
                    ("-" + test_n if test_n else "default")
                if map_key not in envs:
                    envs[map_key] = {}
                if test_n is None:
                    if "default" not in envs[map_key]:
                        envs[map_key]["default"] = {}
                    envs[map_key]["default"][terrain_type] = img_to_grid(
                        entry.path.decode())
                else:
                    if test_n not in envs[map_key]:
                        envs[map_key][test_n] = {}
                        envs[map_key][test_n]["name"] = env_key
                        envs[map_key][test_n]["channels"] = {}
                    envs[map_key][test_n]["channels"][terrain_type] = img_to_grid(
                        entry.path.decode())

    # For all environments, if they are missing a channel, add the default, or if no default, zeros
    for env in envs:
        for test in envs[env]:
            if test == "default":
                continue
            for channel in map_channels.values():
                if channel not in envs[env][test]:
                    if "default" in envs[env] and channel in envs[env]["default"]:
                        envs[env][test]["channels"][channel] = envs[env]["default"][channel]
                    # else:
                    #     envs[env][test]["channels"][channel] = np.zeros(
                    #         (int(config["Environment"]["width"]), int(config["Environment"]["height"])))
    # prints keys
    dishes = {}
    # turns envs into a dictionary of PetriDishes, with the key being the environment name
    for env in envs:
        for test in envs[env]:
            if test != "default":
                dishes[envs[env][test]["name"]] = PetriDish.from_channels(envs[env][test]["name"],
                                                                          envs[env][test]["channels"],
                                                                          config["Environment"])
    return dishes


def load_tests(folder, flat=False):
    # Walks through the subfolders, gathers their metadata, and adds their environments
    # to a dictionary whose key is the metadata title
    tests = {}
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.is_dir():
            config = get_env_config(entry.path.decode())
            tests[entry.name.decode()] = gen_env_dict(
                entry.path.decode(), config)

    if flat:
        # Flatten the dictionary
        flat_tests = {}
        for test in tests:
            for env in tests[test]:
                flat_tests[test + "_" + env] = tests[test][env]
        return flat_tests
    return tests
