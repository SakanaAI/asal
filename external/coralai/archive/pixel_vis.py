# import taichi as ti
# import numpy as np
# from matplotlib.cm import get_cmap
# from coralai.sim.World import World
# from coralai.Experiment import coralai as coralai
# from .Vis import Vis
# import time

# LEVELS = 20
# BORDER_SIZE = 2

# @ti.data_oriented
# class PixelVis(Vis):
#     def __init__(self, coralai: coralai, ch_cmaps: dict, scale=None, out_path=None):
#         super().__init__(coralai, ch_cmaps, scale=scale, out_path=out_path)

#         if scale is None:
#             max_dim = max(self.world.shape)
#             desired_max_dim = 800 // 2  # PixelVis uses 2 pixels per cell
#             scale = desired_max_dim // max_dim

#         self.scale = scale
#         self.cluster_size = 2 * self.scale + BORDER_SIZE
#         self.img_w = self.world.shape[0] * self.cluster_size + 1
#         self.img_h = self.world.shape[1] * self.cluster_size + 1
#         self.cmaps = self.process_ch_cmaps(ch_cmaps)

#     @ti.func
#     def write_to_image(self, mem: ti.types.ndarray(), chindices: ti.types.vector()):
#         for i,j in self.image:
#             xind = i//self.cluster_size
#             yind = j//self.cluster_size

#             in_border = (i % self.cluster_size < BORDER_SIZE) or (j % self.cluster_size < BORDER_SIZE)
#             if in_border:
#                 self.image[i, j] = ti.Vector([0.25, 0.25, 0.25])  # Black for border
#             else:
#                 # Adjusted calculation for channel index
#                 adjusted_i = (i % self.cluster_size) - BORDER_SIZE
#                 adjusted_j = (j % self.cluster_size) - BORDER_SIZE
#                 chx = adjusted_i // self.scale
#                 chy = adjusted_j // self.scale
#                 chnum = chx + 2 * chy
#                 chindex = chindices[chnum]
#                 low = self.chlims[chnum, 0]
#                 high = self.chlims[chnum, 1]
#                 chval = ti.math.clamp(mem[xind, yind, chindex], low, high)
#                 chval_norm = (chval - low) / (high - low)
#                 self.image[i, j] = self.cmaps[chnum, int(chval_norm * (LEVELS - 1))]

#     def process_ch_cmaps(self, ch_cmaps):
#         ti_cmaps = ti.Vector.field(n=3, dtype=ti.f32, shape=(4, LEVELS))
#         cmap_arrs = np.zeros((4, LEVELS, 3), dtype=np.float32)
#         for i in range(4):
#             cmap = ch_cmaps[self.chids[i]]
#             cmap_arrs[i, :, :] = get_cmap(cmap)(np.linspace(0, 1, LEVELS))[:, :3]
#         ti_cmaps.from_numpy(cmap_arrs)
#         return ti_cmaps
    
    # def validate_channels(self, ch_cmaps):
    #     chs = list(ch_cmaps.keys())
    #     if len(chs) != 4:
    #         raise ValueError("PixelVis: currently only supports exactly 4 channels. Pass None to color black.")
    #     for ch in chs:
    #         if len(self.world.windex_obj[ch]) != 1: 
    #             raise ValueError(f"PixelVis: currently only supports channels with exactly 1 index. Channel {ch} has {len(self.world.windex_obj[ch])} indices. Please pass singular subchannels")
    #     return chs

    # def get_channel_indices(self):
    #     chids = np.array(self.world.windex_obj[self.chids], dtype=np.int32)
    #     if len(chids) != 4:
    #         raise ValueError("PixelVis: Oops, Error in indexing channels via world, got more than 4 Channel IDs - should have been caught above")
    #     chids_field = ti.field(dtype=ti.i32, shape=4)
    #     chids_field.from_numpy(chids)
    #     return chids_field

    # def get_channel_limits(self):
    #     ch_lims = np.zeros((4,2), dtype=np.float32)
    #     for i, ch in enumerate(self.chids):
    #         if isinstance(ch, tuple):
    #             ch_lims[i] = self.world.channels[ch[0]].lims
    #         else:
    #             ch_lims[i] = self.world.channels[ch].lims
    #     ch_lims_field = ti.field(dtype=ti.f32, shape=(4,2))
    #     ch_lims_field.from_numpy(ch_lims)
    #     return ch_lims_field