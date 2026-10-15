from matplotlib import gridspec, pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import numpy as np
import torch
import src.utils as utils

# TODO: Incorporate Visualizers into Simulation Class
# class Visualizer:
#     def __init__(self, id: str, ax_func: callable, axs_titles: list[str] = [], input_channel_ids: list[str] = [], 
#                  req_metadata: dict = {}, metadata: dict = {}):
#         self.id = id
#         self.ax_func = ax_func
#         self.axs_titles = axs_titles
#         self.input_channel_ids = input_channel_ids
#         self.metadata = {"id": id,
#                          "input_channel_ids": input_channel_ids,
#                          "req_metadata": req_metadata,
#                          "axs_titles": axs_titles
#                          }
#         self.metadata.update(metadata)

#     def assert_compatibility(self, sim):
#         for id in self.input_channel_ids:
#             assert id in sim.channels.keys(), f"Input channel \"{id}\" for Visualizer \"{self.id}\" not in simulation \"{sim.id}\""
#         for k, req_meta in self.req_metadata.items():
#             if k in sim.channels:
#                 assert (utils.check_subdict(sim.channels[k].metadata, req_meta),
#                 f"Visualizer (\"{self.id}\"): required channel metadata ({req_meta}) not found in channel {k} metadata ({sim.channels[k].metadata}).")
#             else:
#                 assert (utils.check_subdict(sim.metadata, {k:req_meta}),
#                 f"Visualizer (\"{self.id}\"): required simulation metadata ({req_meta}) not found in simulation metadata ({sim.metadata}).")

    # def update_ax(self, sim):
    #     self.ax_func(sim, axs, *[sim.channels[id] for id in self.input_channel_ids], self.metadata)

def vis_weather(sim, ax_port, ax_rmap, ax_regen, port_ch, metadata):
        period = sim.metadata['period'] 
        port_id_map = port_ch.metadata['port_id_map']
        resources = port_ch.metadata['resources']

        # TODO: Update to work with more resources
        resource_colors = ['red', 'green', 'blue']

        ax_port.clear()
        ax_port.set_title("Resource Values (Yl Good)")
        midmap = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=10)
        ax_port.imshow(port_ch.contents.squeeze(), cmap="viridis", norm=midmap)

        ax_rmap.clear()
        ax_rmap.set_title('Resource Map')
        cmap = mcolors.ListedColormap(['white'] + resource_colors)
        ax_rmap.imshow(port_id_map, cmap=cmap)

        # Bottom: Regeneration Functions
        ax_regen.clear()
        ax_regen.set_title("Regeneration functions")
        for idx, resource in enumerate(resources):
            t = torch.linspace(0, 2*np.pi, 1000)
            y = resource.regen_func(t+period-np.pi/2).numpy()
            ax_regen.plot(t+period, y, color=resource_colors[idx], label=f"Resource {idx+1}")

        ax_regen.axvline(x=period + np.pi/2, color='black', linestyle='--')  # Static vertical line
        ax_regen.axvspan(period, period + np.pi/2, facecolor='gray', alpha=0.2)  # Highlight indicating past
        ax_regen.legend(loc="upper right")


# regen_vis = Visualizer("regen_vis", vis_regen, input_channel_ids=["ports"], req_metadata={"period": 0.0})