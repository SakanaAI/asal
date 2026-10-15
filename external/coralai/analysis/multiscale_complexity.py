import os
import torch
import numpy as np
import taichi as ti
from PIL import Image
from coralai.substrate.substrate import Substrate
from coralai.visualization import compose_visualization, VisualizationData
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
from PIL import Image

@ti.kernel
def avg_neighs(mem: ti.types.ndarray(), out_mem: ti.types.ndarray()):
    # Relies on 2^n dimensions
    for batch, ch, i, j in ti.ndrange(mem.shape[0], mem.shape[1], mem.shape[2], mem.shape[3]):
        out_mem[batch, ch, i//2, j//2] += mem[batch, ch, i, j] / 4.0

def renormalize_patterns(patterns: torch.Tensor):
    out_mem = torch.zeros((patterns.shape[0], patterns.shape[1], patterns.shape[2]//2, patterns.shape[3]//2), device=patterns.device)
    avg_neighs(patterns, out_mem)
    return out_mem

def calc_rg_flow_ragged(patterns: torch.Tensor, renorm_steps: int):
    rg_flow = []
    rg_flow.append(patterns)
    for _ in range(renorm_steps):
        rg_flow.append(renormalize_patterns(rg_flow[-1]))
    return rg_flow

@ti.kernel
def upscale(pattern_coarse: ti.types.ndarray(), out_mem_fine: ti.types.ndarray()):
    for batch, ch, i, j in ti.ndrange(out_mem_fine.shape[0], out_mem_fine.shape[1], out_mem_fine.shape[2], out_mem_fine.shape[3]):
        out_mem_fine[batch, ch, i, j] = pattern_coarse[batch, ch, i//2, j//2]

def match_sizes(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor):
    coarse_upscaled = torch.zeros_like(patterns_fine)
    upscale(patterns_coarse, coarse_upscaled)
    return coarse_upscaled

def calc_pixel_complexities(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor):
    patterns_coarse_upscaled = match_sizes(patterns_coarse, patterns_fine)
    return torch.abs((patterns_coarse_upscaled * patterns_fine) -
                      ((patterns_coarse_upscaled * patterns_coarse_upscaled) +
                       (patterns_fine * patterns_fine))/2.0)

def calc_pixel_complexities_ragged(rg_flow: list):
    pixel_complexities = []
    for step in range(1, len(rg_flow)):
        pixel_complexities.append(calc_pixel_complexities(rg_flow[step], rg_flow[step-1]))
    return pixel_complexities

def calc_overlaps(patterns1: torch.Tensor, patterns2: torch.Tensor):
    return torch.mean(patterns1 * patterns2, dim=(2, 3))

def calc_partial_complexities(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor):
    patterns_coarse_upscaled = match_sizes(patterns_coarse, patterns_fine)
    return torch.abs(calc_overlaps(patterns_coarse_upscaled, patterns_fine) -
                     (calc_overlaps(patterns_fine, patterns_fine) + calc_overlaps(patterns_coarse, patterns_coarse))/2.0)

def calc_all_partial_complexities(patterns: torch.Tensor, renorm_steps: int):
    """
    Patterns: (batch_size, n_chs, w, h)
    Returns: (batch_size, n_chs, renorm_steps)
    """
    all_partial_complexities = []
    scaling_factors = []
    for step in range(1, renorm_steps):
        patterns_coarse = renormalize_patterns(patterns)
        all_partial_complexities.append(calc_partial_complexities(patterns_coarse, patterns))
        scaling_factors.append((1,2**step))
        patterns = patterns_coarse
    return torch.stack(all_partial_complexities), scaling_factors

def calc_complexities(patterns: torch.Tensor, renorm_steps: int):
    """
    Patterns: (batch_size, n_chs, w, h)
    Returns: (batch_size, n_chs)
    """
    all_partial_complexities, _ = calc_all_partial_complexities(patterns, renorm_steps)
    return torch.sum(all_partial_complexities, dim=0)



if __name__ == "__main__":
    ti.init(ti.metal)
    renorm_steps = 10

    msc_path = os.path.join(os.path.dirname(__file__), 'msc')
    image_names = sorted([f for f in os.listdir(msc_path) if f.endswith('.png')])  # Sort the images
    patterns = torch.zeros((len(image_names), 3, 1024, 1024))
    for idx, image_name in enumerate(image_names):
        image = Image.open(os.path.join(msc_path, image_name))
        image = image.convert("RGB")
        np_image = np.array(image) / 255.0  # Normalize RGB values to 0-1
        torch_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()  # Convert to torch tensor and adjust dimensions
        patterns[idx] = torch_image


    rg_flow = calc_rg_flow_ragged(patterns, renorm_steps)
    pixel_complexities = calc_pixel_complexities_ragged(rg_flow)
    print(pixel_complexities[0].shape)


    # Resize the figure and reduce margins
    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust figsize to your preference
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)  # Minimize margins

    # Display the first image and its first renormalization level as a heatmap
    img_display = ax.imshow(patterns[0].permute(1, 2, 0).numpy(), extent=[0, 1024, 0, 1024])
    heatmap_display = ax.imshow(pixel_complexities[0][0][0].numpy(), cmap='viridis', alpha=0.75, extent=[0, 1024, 0, 1024], interpolation='nearest')

    # Slider for selecting the renormalization level
    axcolor = 'lightgoldenrodyellow'
    ax_renorm = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider_renorm = Slider(ax_renorm, 'Renorm Level', 0, len(pixel_complexities)-1, valinit=0, valstep=1)

    # Slider for selecting the image index
    ax_img = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    slider_img = Slider(ax_img, 'Image Index', 0, len(patterns)-1, valinit=0, valstep=1)

    def update(val):
        renorm_level = int(slider_renorm.val)
        img_index = int(slider_img.val)
        img = patterns[img_index].permute(1, 2, 0).numpy()
        heatmap = pixel_complexities[renorm_level][img_index][0].numpy()

        # Update image display
        img_display.set_data(img)

        # Upscale heatmap to match original image size and update heatmap display
        heatmap_upscaled = np.kron(heatmap, np.ones((2**renorm_level, 2**renorm_level)))
        heatmap_display.set_data(heatmap_upscaled)
        heatmap_display.set_extent([0, img.shape[1], 0, img.shape[0]])

        fig.canvas.draw_idle()

    # Call update function on slider value change
    slider_renorm.on_changed(update)
    slider_img.on_changed(update)

    # # Define the update function for the animation
    # def animate(frame):
    #     # Calculate the slider value using a sinusoidal function of the frame number
    #     t = np.linspace(0, 2 * np.pi, 100)  # 100 frames for a full cycle
    #     # img_index = int((np.sin(t[frame]) + 1) / 2 * (len(patterns) - 1))
    #     percent_animated = 1 - abs(np.sin(t[frame]))
    #     print(percent_animated)
    #     print(int(percent_animated * (len(pixel_complexities) - 1)))
    #     renorm_level = int(percent_animated * (len(pixel_complexities) - 1))
        
    #     # slider_img.set_val(img_index)
    #     slider_renorm.set_val(renorm_level)

    # # Create the animation
    # ani = FuncAnimation(fig, animate, frames=100, repeat=True)

    plt.show()
