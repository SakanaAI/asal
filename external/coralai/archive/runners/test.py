import torch
import numpy as np
import matplotlib.pyplot as plt


t = torch.tensor([[[[0.0,0,1],
                   [0,10,0],
                   [0,0,1]],
                  [[1,0,3],
                   [3,0,0],
                   [2,0,10]],
                  [[0,0,0],
                   [0,100,0],
                   [0,0,0]]
                  ]])

print(t[0].eq(231))
t[0,t[0].eq(231)] = 9999
print(t)

exit()
t = torch.tensor([[[[0.0,0,1],
                   [0,10,0],
                   [0,0,1]],
                  [[1,0,3],
                   [3,0,0],
                   [2,0,10]],
                  [[0,0,0],
                   [0,100,0],
                   [0,0,0]]
                  ]])
print(torch.softmax(t[0,[0,1]],dim=1))



exit()



# Predefined sequence of offsets in a clockwise order starting from [1, 0]
dir_kernel = torch.tensor([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])

def produce_order(kernel_len):
    order = []
    ind = 0
    i = 0
    while i < kernel_len:
        order.append(ind)
        if ind > 0:
            ind = -ind
        else:
            ind = (-ind + 1)
        i += 1
    return order

dir_order = produce_order(dir_kernel.shape[0])
print(dir_order)
kernel_len = len(dir_kernel)
rot = 3
max_act_i = 2

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
import matplotlib.cm as cm

# Plot the origin
ax.plot(0, 0, 'ok')  # 'ok' plots a black dot

colors = cm.rainbow(np.linspace(0, 1, dir_kernel.shape[0]))

print(f"Rot: {rot}, Dir: {dir_kernel[rot]}")
print(f"max_act_i: {max_act_i}, max_act_rot: {(rot + dir_order[max_act_i]) % dir_kernel.shape[0]}," +
      f"max_act_dir: {dir_kernel[(rot + dir_order[max_act_i]) % dir_kernel.shape[0]]}")
for i in range(dir_kernel.shape[0]):
    ind = (rot+dir_order[i]) % dir_kernel.shape[0]
    print(f"i {i}, dir_ind: {ind}, dir: {dir_kernel[ind]}")
    dx, dy = dir_kernel[ind].numpy()
    color = colors[i]
    ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.2, fc=color, ec=color)
    ax.text(dx * 1.1, dy * 1.1, str(i), color='black')
plt.grid(True)
plt.show()
exit()



# Function to map a given direction to its starting index in the offset_sequence
def get_starting_index_for_direction(direction):
    # Calculate angle for the given direction
    angle = np.arctan2(direction[1], direction[0])
    
    # Normalize the angle to be within [0, 2*pi)
    if angle < 0:
        angle += 2 * np.pi

    # Each segment in the circle corresponds to 45 degrees (pi/4 radians)
    # Calculate the index by dividing the angle by the segment size
    index = int(round(4 * angle / np.pi)) % len(offset_sequence)
    return index

def get_alternating_offsets(offsets):
    i = 0
    new_offsets = []
    ind = 0
    while i < len(offsets):
        new_offsets.append(offsets[ind])
        if ind >= 0:
            ind = -(ind + 1)
        else:
            ind = -ind
        i += 1
    return new_offsets

new_dir = [-1,-1]
starting_index = get_starting_index_for_direction(new_dir)

def rotate_sequence(starting_index, sequence):
    return sequence[starting_index:] + sequence[:starting_index]

rotated_sequence = rotate_sequence(starting_index, offset_sequence)

print(offset_sequence)
print(f"new_dir: {new_dir},\nstarting_index: {starting_index},\nrotated_sequence: {rotated_sequence}")
print(get_alternating_offsets(rotated_sequence))
