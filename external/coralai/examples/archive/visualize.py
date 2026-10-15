
# %% 

import colorsys
import traceback
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2  # You might need OpenCV for HSV conversions
import yaml

import importlib
# import src.tester as tester
# importlib.reload(tester)

# %% Load Config---------------------------------------------------------------
def load_check_config(config_object):
    if isinstance(config_object, str):
        with open(config_object) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    elif isinstance(config_object, dict):
        config = config_object
    else:
        raise TypeError("config_object must be either a path (str) or a config (dict)")
    
    # asserts colormaps are of right length
    assert len(config["visualize"]["colormaps"]) == len(config["environment"]["channels"])
    # asserts colormaps are real matplotlib colormaps
    for cmap in config["visualize"]["colormaps"].values():
        assert cmap in plt.colormaps()

    return config

# if __name__ == "__main__":
#     verbose = True # For testing
#     channels = np.array([
#             [[0.2, 0, 0], [0.8, 1, 0]],   # food channel
#             [[0, 0, 0], [0.1, 0, 1]],     # poison channel
#             [[0, 1, 1], [0, 0, 0]]        # obstacle channel, binary
#         ])
#     config = {
#         "environment": {
#             "channels": ["food", "poison", "obstacle"]
#         },
#         "visualize": {
#             "colormaps": {
#                 "food": "Greens",
#                 "poison": "Oranges",
#                 "obstacle": "binary"
#             }
#         }
#     }
#     config = load_check_config(config)


# %% Blend Colors--------------------------------------------------------------
def blend_colors(color1, color2, weight):
    # Convert RGB colors to HSV
    hsv1 = colorsys.rgb_to_hsv(*color1)
    hsv2 = colorsys.rgb_to_hsv(*color2)

    # Blend the colors based on the weight
    blended_hsv = (
        (1 - weight) * hsv1[0] + weight * hsv2[0],    # Weighted average of the hues
        (1 - weight) * hsv1[1] + weight * hsv2[1],    # Weighted average of the saturations
        (1 - weight) * hsv1[2] + weight * hsv2[2]     # Weighted average of the values/brightness
    )

    # Convert the blended color back to RGB
    blended_rgb = colorsys.hsv_to_rgb(*blended_hsv)

    return blended_rgb

# if __name__ == "__main__":
#     # Test blend colors
#     color1 = (1.0, 0, 0)
#     color2 = (0, 1.0, 0)
#     def show_blend_result(result, title):
#         plt.figure(figsize=(6, 2))
#         plt.subplot(1, 3, 1)
#         plt.imshow(np.full((10, 10, 3), color1))
#         plt.axis('off')
#         plt.title('Color 1')
#         plt.subplot(1, 3, 2)
#         plt.imshow(np.full((10, 10, 3), color2))
#         plt.axis('off')
#         plt.title('Color 2')
#         plt.subplot(1, 3, 3)
#         plt.imshow(np.full((10, 10, 3), result))
#         plt.axis('off')
#         plt.title('Blended')
#         plt.show()
#         print("\033[93mResult of test", title, ":", result, "\033[0m")

#     tester.test(lambda: blend_colors(color1, color2, 0.5),
#                 "Blend colors",
#                 verbose,
#                 show_blend_result)
    
# %% Show Image, Show Images, Channel to Image---------------------------------
def show_image(image, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    plt.show()
    return ax


def show_images(images, columns=3):
    n = len(images)
    rows = n // columns
    rows += n % columns
    position = range(1, n + 1)
    fig = plt.figure(figsize=(20, 20))
    for k, image in zip(position, images):
        ax = fig.add_subplot(rows, columns, k)
        ax.imshow(image)
        plt.axis('off')
    plt.show()


def channel_to_image(channel, cmap="gray"):
    norm = mpl.colors.Normalize(channel.min(), channel.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(channel)

# if __name__ == "__main__":

#     tester.test(lambda: channel_to_image(channels[0], "copper"),
#                 "Channel to Image",
#                 verbose,
#                 lambda result, title: show_image(result))  

# %% Channels to Images -------------------------------------------------------
def channels_to_images(config, channels, colormaps=None):
    if colormaps is None:
        # food, poison, obstacle
        colormaps = config["visualize"]["colormaps"]
    assert len(colormaps) == len(channels)

    images=[]
    for channel_name in config["environment"]["channels"]:
        index = config["environment"]["channels"].index(channel_name)
        colormap = config["visualize"]["colormaps"][channel_name]
        images.append(channel_to_image(channels[index], colormap))
  
    return images

# if __name__ == "__main__":
#     tester.test(lambda: channels_to_images(config, channels),
#                 "Channels to Images",
#                 verbose,
#                 lambda result, t: show_images(result))

# %% Visualize Stacked Channels -----------------------------------------------
def visualize_stacked_channels(config, channels):
    # In the order provided, stacks images on top of each other (first is the background)
    # Makes alpha 0 for overlapping pixels (so the top image is visible, taking precedence)
    images = channels_to_images(config, channels, config["visualize"]["colormaps"].values())
    super_image = None
    for i in range(len(images)):
        if super_image is None:
            super_image = images[i]
        else:
            # Mask for the current channel where its value is greater than 0
            mask = channels[i] > 0
            # Set the super image to the current image where the mask is true
            super_image[mask] = images[i][mask]
    
    return super_image
    
# if __name__ == "__main__":
#     show_images(channels_to_images(config, channels))
#     tester.test(lambda: visualize_stacked_channels(config, channels),
#                 "Visualize Stacked Channels",
#                 verbose,
#                 lambda result, title: show_image(result))


# %% Pixel Clusters Visualization ---------------------------------------------
def pixel_vis(config, channels):
    pass 