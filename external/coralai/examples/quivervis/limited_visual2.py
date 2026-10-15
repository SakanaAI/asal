from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import os

# Make the UV quiver thing a function so it can be any 3x3 kernel
# Muscle value can be > 1
# Why are some long arrows black? Should be colored?

# Load the 100 tensors from the directory
tensor_files = sorted(os.listdir('fake_data'))
tensors = [np.load(os.path.join('fake_data', file)) for file in tensor_files]

WIDTH = len(tensors[0][0][0])
HEIGHT = len(tensors[0][0])

plots = []

fig, ax = plt.subplots()

def muscle_activations(muscle_radii, activations):
    return muscle_radii * activations

def get_muscle_color(value):
    if value > 0:
        green = value * 255
        red = 0
    else:
        green = 0
        red = -value * 255
    return (red, green, 0)  # Return RGB tuple

ax.clear()

def plot_quiver(X, Y, U, V, color, scale):
    ax.quiver(X, Y, U, V, color=color, angles='xy', scale_units='xy', scale=scale)

def plot_scatter(X, Y, color, alpha, size):
    ax.scatter(X, Y, color=color, alpha=alpha, s=size)

def update_plot(frame):
    def muscle_activations(muscle_radii, activations):
        return muscle_radii * activations

    def get_muscle_color(value):
        if value > 0:
            green = value * 255
            red = 0
        else:
            green = 0
            red = -value * 255
        return (red, green, 0)  # Return RGB tuple

    ax.clear()

    X, Y = np.meshgrid(range(0,WIDTH), range(0,HEIGHT))
    quiver_scale = .01

    UP = muscle_activations(tensors[frame][3], tensors[frame][0])
    U = np.zeros_like(UP)
    V = np.abs(UP)
    color = [get_muscle_color(value) for value in UP.flatten()]
    plot_quiver(X, Y, U, V, color, quiver_scale)

    RIGHT = muscle_activations(tensors[frame][4], tensors[frame][0])
    U = np.abs(RIGHT)
    V = np.zeros_like(RIGHT)
    color = [get_muscle_color(value) for value in RIGHT.flatten()]
    plot_quiver(X, Y, U, V, color, quiver_scale)

    DOWN = muscle_activations(tensors[frame][5], tensors[frame][0])
    U = np.zeros_like(DOWN)
    V = -np.abs(DOWN)
    color = [get_muscle_color(value) for value in DOWN.flatten()]
    plot_quiver(X, Y, U, V, color, quiver_scale)

    LEFT = muscle_activations(tensors[frame][6], tensors[frame][0])
    U = -np.abs(LEFT)
    V = np.zeros_like(LEFT)
    color = [get_muscle_color(value) for value in LEFT.flatten()]
    plot_quiver(X, Y, U, V, color, quiver_scale)

    cytoplasm = tensors[frame][1]
    plot_scatter(X, Y, (.18, .97, 1), .3, [value*500 for value in cytoplasm.flatten()])

ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)    


ani = FuncAnimation(fig=fig, func=update_plot, frames=40, interval=100)
plt.show()


