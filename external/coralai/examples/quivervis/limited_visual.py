from matplotlib import animation 
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
    quiver_scale = WIDTH * .005
    
    # Quiver Plots (Muscle Radii * Activation)
    UP = muscle_activations(tensors[frame][3], tensors[frame][0])
    U = np.zeros_like(UP)
    V = np.abs(UP)
    ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in UP.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

    RIGHT = muscle_activations(tensors[frame][4], tensors[frame][0])
    U = np.abs(RIGHT)
    V = np.zeros_like(RIGHT)
    ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in RIGHT.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

    DOWN = muscle_activations(tensors[frame][5], tensors[frame][0])
    U = np.zeros_like(DOWN)
    V = -np.abs(DOWN)
    ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in DOWN.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

    LEFT = muscle_activations(tensors[frame][6], tensors[frame][0])
    U = -np.abs(LEFT)
    V = np.zeros_like(LEFT)
    ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in LEFT.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

    # Scatter Plot (Cytoplasm)
    cytoplasm = tensors[frame][1]
    ax.scatter(X, Y, color=(.18, .97, 1), alpha=.3, s=[value*5000 for value in cytoplasm.flatten()])

    ax.set_xlim(-1, WIDTH)
    ax.set_ylim(-1, HEIGHT)    


ani = animation.FuncAnimation(fig=fig, func=update_plot, frames=40, interval=100)
f = r"./limited_eincasm_20.gif" 
writergif = animation.PillowWriter(fps=20) 
ani.save(f, writer=writergif)


