import time
import numpy as np
import taichi as ti
import warnings
import torch
import torch.nn as nn
from timeit import Timer


@ti.func
def ReLU(x):
    return x if x > 0 else 0

@ti.func
def sigmoid(x):
    return 1 / (1 + ti.exp(-x))

@ti.func
def inverse_gaussian(x):
    return -1./(ti.exp(0.89*ti.pow(x, 2.))+1.)+1.

def ch_norm(input_tensor):
    # Calculate the mean across batch and channel dimensions
    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)

    # Calculate the variance across batch and channel dimensions
    var = input_tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

    # Normalize the input tensor
    input_tensor.sub_(mean).div_(torch.sqrt(var + 1e-5))

    return input_tensor


class TaichiStructFactory:
    """This class is a factory for creating Taichi structures."""

    def __init__(self):
        self.val_dict = {}
        self.type_dict = {}
        self.struct_type = None

    def add(self, name, val, dtype):
        self.val_dict[name] = val
        self.type_dict[name] = dtype

    def add_i(self, name, val):
        self.add(name, val, ti.i32)

    def add_f(self, name, val):
        self.add(name, val, ti.f32)

    def add_nparr_float(self, name, val):
        self.add(name, val, ti.types.vector(n=val.shape[0], dtype=ti.f32))

    def add_nparr_int(self, name, val):
        self.add(name, val, ti.types.vector(n=val.shape[0], dtype=ti.i32))

    def add_tivec_f(self, name, val):
        self.add(name, val, ti.types.vector(n=val.n, dtype=ti.f32))

    def add_tivec_i(self, name, val):
        self.add(name, val, ti.types.vector(n=val.n, dtype=ti.i32))

    def add_timat_f(self, name, val):
        self.add(name, val, ti.types.matrix(n=val.n, m=val.m, dtype=ti.f32))

    def add_timat_i(self, name, val):
        self.add(name, val, ti.types.matrix(n=val.n, m=val.m, dtype=ti.i32))

    def build(self):
        struct_type = ti.types.struct(**self.type_dict)
        val_field = struct_type.field(shape=())
        for k, v in self.val_dict.items():
            val_field[None][k] = v
        self.struct_type = struct_type
        return val_field


class WorldIndex:
    def __init__(self, index_tree):
        self.index_tree = index_tree

    def _get_tuple_indices(self, key_tuple):
        chid = key_tuple[0]
        subchid = key_tuple[1]
        if isinstance(subchid, list):
            indices = []
            for subchid_single in key_tuple[1]:
                indices += self.index_tree[chid]["subchannels"][subchid_single][
                    "indices"
                ]
        else:
            indices = self.index_tree[chid]["subchannels"][subchid]["indices"]
        return indices


    def __getitem__(self, key):
        if isinstance(key, tuple):
            return np.array(self._get_tuple_indices(key))

        elif isinstance(key, list):
            indices = []
            for chid in key:
                if isinstance(chid, tuple):
                    indices += self._get_tuple_indices(chid)
                else:
                    indices += self.index_tree[chid]["indices"]
            return np.array(indices)
        else:
            return np.array(self.index_tree[key]["indices"])

    def __setitem__(self, key, value):
        raise ValueError(
            f"World: World indices are read-only. Cannot set index {key} to {value} - get/set to the world iteself"
        )
        

class Channel:
    def __init__(
            self, chid, world, ti_dtype=None,
            lims=None,
            metadata: dict=None, **kwargs):
        self.chid = chid
        self.world = world
        self.lims = np.array(lims) if lims else np.array([-1, 1], dtype=np.float32)
        self.ti_dtype = ti_dtype if ti_dtype is not None else ti.f32
        self.memblock = None
        self.indices = None
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        field_md = {
            'id': self.chid,
            'ti_dtype': self.ti_dtype,
            'lims': self.lims,
        }
        self.metadata.update(field_md)
    
    def link_to_mem(self, indices, memblock):
        self.memblock = memblock
        indices = np.array(indices)
        if len(indices) == 1:
            indices = indices[0]
        self.indices = indices
        self.metadata['indices'] = indices
    
    def add_subchannel(self, chid, ti_dtype=ti.f32, **kwargs):
        subch = Channel(chid, self.world, ti_dtype=ti_dtype, **kwargs)
        subch.metadata['parent'] = self
        self.metadata[chid] = subch
        self.metadata['subchids'] = self.metadata.get('subchids', [])
        self.metadata['subchids'].append(chid)
        return subch

    def get_data(self):
        if self.memblock is None:
            raise ValueError(f"Channel: Channel {self.chid} has not been allocated yet.")
        else:
            return self.memblock[self.indices]

    def __getitem__(self, key):
        return self.metadata.get(key)

    def __setitem__(self, key, value):
        self.metadata[key] = value


@ti.data_oriented
class World:
    # TODO: Support multi-level indexing beyond 2 levels
    # TODO: Support mixed taichi and torch tensors - which will be transferred more?
    def __init__(self, shape, torch_dtype, torch_device, channels: dict = None):
        self.w = shape[0]
        self.h = shape[1]
        self.shape = (*shape, 0) # changed in malloc
        self.mem = None
        self.windex = None
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device
        self.channels = {}
        if channels is not None:
            self.add_channels(channels)
        self.ti_ind_builder = TaichiStructFactory()
        self.ti_lims_builder = TaichiStructFactory()
        self.ti_indices = -1
        self.ti_lims = -1


    def add_channel(self, chid: str, ti_dtype=ti.f32, **kwargs):
        if self.mem is not None:
            raise ValueError(
                f"World: When adding channel {chid}: Cannot add channel after world memory is allocated (yet)."
            )
        self.channels[chid] = Channel(chid, self, ti_dtype=ti_dtype, **kwargs)


    def add_channels(self, channels: dict):
        if self.mem is not None:
            raise ValueError(
                f"World: When adding channels {channels}: Cannot add channels after world memory is allocated (yet)."
            )
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, dict):
                self.add_channel(chid, **ch)
            else:
                self.add_channel(chid, ch)


    def check_ch_shape(self, shape):
        lshape = len(shape)
        if lshape > 3 or lshape < 2:
            raise ValueError(
                f"World: Channel shape must be 2 or 3 dimensional. Got shape: {shape}"
            )
        if shape[:2] != self.shape[:2]:
            print(shape[:2], self.shape[:2])
            raise ValueError(
                f"World: Channel shape must be (w, h, ...) where w and h are the world dimensions: {self.shape}. Got shape: {shape}"
            )
        if lshape == 2:
            return 1
        else:
            return shape[2]


    def stat(self, key):
        # Prints useful metrics about the channel(s) and contents
        minval = self[key].min()
        maxval = self[key].max()
        meanval = self[key].mean()
        stdval = self[key].std()
        shape = self[key].shape
        print(
            f"{key} stats:\n\tShape: {shape}\n\tMin: {minval}\n\tMax: {maxval}\n\tMean: {meanval}\n\tStd: {stdval}"
        )


    def _transfer_to_mem(self, mem, tensor_dict, index_tree, channel_dict):
        for chid, chindices in index_tree.items():
            if "subchannels" in chindices:
                for subchid, subchtree in chindices["subchannels"].items():
                    if tensor_dict[chid][subchid].dtype != self.torch_dtype:
                        warnings.warn(
                            f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m",
                            stacklevel=3,
                        )
                    if len(tensor_dict[chid][subchid].shape) == 2:
                        tensor_dict[chid][subchid] = tensor_dict[chid][
                            subchid
                        ].unsqueeze(2)
                    mem[:, :, subchtree["indices"]] = tensor_dict[chid][subchid].type(
                        self.torch_dtype
                    )
                    channel_dict[chid].add_subchannel(
                        subchid, ti_dtype=channel_dict[chid].ti_dtype
                    )
                    channel_dict[chid][subchid].link_to_mem(subchtree["indices"], mem)
                channel_dict[chid].link_to_mem(chindices["indices"], mem)
            else:
                if tensor_dict[chid].dtype != self.torch_dtype:
                    warnings.warn(
                        f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m",
                        stacklevel=3,
                    )
                if len(tensor_dict[chid].shape) == 2:
                    tensor_dict[chid] = tensor_dict[chid].unsqueeze(2)
                mem[:, :, chindices["indices"]] = tensor_dict[chid].type(
                    self.torch_dtype
                )
                channel_dict[chid].link_to_mem(chindices["indices"], mem)
        return mem, channel_dict


    def add_ti_inds(self, key, inds):
        if len(inds) == 1:
            self.ti_ind_builder.add_i(key, inds[0])
        else:
            self.ti_ind_builder.add_nparr_int(key, np.array(inds))


    def _index_subchannels(self, subchdict, start_index, parent_chid):
        end_index = start_index
        subch_tree = {}
        for subchid, subch in subchdict.items():
            if not isinstance(subch, torch.Tensor):
                raise ValueError(
                    f"World: Channel grouping only supported up to a depth of 2. Subchannel {subchid} of channel {parent_chid} must be a torch.Tensor. Got type: {type(subch)}"
                )
            subch_depth = self.check_ch_shape(subch.shape)
            indices = [i for i in range(end_index, end_index + subch_depth)]
            self.add_ti_inds(parent_chid + "_" + subchid, indices)
            self.ti_lims_builder.add_nparr_float(
                parent_chid + "_" + subchid, self.channels[parent_chid].lims
            )
            subch_tree[subchid] = {
                "indices": indices,
            }
            end_index += subch_depth
        return subch_tree, end_index - start_index


    def malloc(self):
        if self.mem is not None:
            raise ValueError("World: Cannot allocate world memory twice.")
        celltype = ti.types.struct(
            **{chid: self.channels[chid].ti_dtype for chid in self.channels.keys()}
        )
        tensor_dict = celltype.field(shape=self.shape[:2]).to_torch(
            device=self.torch_device
        )

        index_tree = {}
        endlayer_pointer = self.shape[2]
        for chid, chdata in tensor_dict.items():
            if isinstance(chdata, torch.Tensor):
                ch_depth = self.check_ch_shape(chdata.shape)
                indices = [
                    i for i in range(endlayer_pointer, endlayer_pointer + ch_depth)
                ]
                self.add_ti_inds(chid, indices)
                self.ti_lims_builder.add_nparr_float(chid, self.channels[chid].lims)
                index_tree[chid] = {"indices": indices}
                endlayer_pointer += ch_depth
            elif isinstance(chdata, dict):
                subch_tree, total_depth = self._index_subchannels(
                    chdata, endlayer_pointer, chid
                )
                indices = [
                    i for i in range(endlayer_pointer, endlayer_pointer + total_depth)
                ]
                self.add_ti_inds(chid, indices)
                self.ti_lims_builder.add_nparr_float(chid, self.channels[chid].lims)
                index_tree[chid] = {
                    "subchannels": subch_tree,
                    "indices": indices,
                }
                endlayer_pointer += total_depth

        self.shape = (*self.shape[:2], endlayer_pointer)
        mem = torch.zeros(self.shape, dtype=self.torch_dtype, device=self.torch_device)
        self.mem, self.channels = self._transfer_to_mem(
            mem, tensor_dict, index_tree, self.channels
        )
        self.windex = WorldIndex(index_tree)
        self.ti_indices = self.ti_ind_builder.build()
        self.ti_lims = self.ti_lims_builder.build()
        self.mem = self.mem.permute(2, 0, 1).unsqueeze(0).contiguous()
        self.shape = self.mem.shape


    def __getitem__(self, key):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot get {key}")
        val = self.mem[self.windex[key], :, :]
        return val
    
    def __setitem__(self, key, value):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot set {key}")
        raise NotImplementedError("World: Setting world values not implemented yet. (Just manipulate memory directly)")


    def get_inds_tivec(self, key):
        indices = self.windex[key]
        itype = ti.types.vector(n=len(indices), dtype=ti.i32)
        return itype(indices)


    def get_lims_timat(self, key):
        lims = []
        if isinstance(key, str):
            key = [key]
        if isinstance(key, tuple):
            key = [key[0]]
        for k in key:
            if isinstance(k, tuple):
                lims.append(self.channels[k[0]].lims)
            else:
                lims.append(self.channels[k].lims)
        if len(lims) == 1:
            lims = lims[0]
        lims = np.array(lims, dtype=np.float32)
        ltype = ti.types.matrix(lims.shape[0], lims.shape[1], dtype=ti.f32)
        return ltype(lims)


@ti.data_oriented
class Organism(nn.Module):
    def __init__(self, world, sensors, n_actuators):
        super(Organism, self).__init__()
        self.world = world
        self.w = world.w
        self.h = world.h
        self.sensors = sensors
        self.sensor_inds = self.world.windex[self.sensors]
        self.n_sensors = self.sensor_inds.shape[0]
        self.n_actuators = n_actuators
        
        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_sensors,
            self.n_actuators,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=self.world.torch_device,
            bias=False
        )


    def forward(self, x=None):
        with torch.no_grad():
            if x is None:
                x=self.world.mem[:, self.sensor_inds, :, :]
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = ch_norm(x)
            return torch.sigmoid(x)


    def perturb_weights(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data) 


@ti.data_oriented
class NCA:
    def __init__(
        self,
        shape=None,
        torch_device=None,
    ):
        if shape is None:
            shape = (100, 100)
        if torch_device is None:
            torch_device = torch.device("cpu")

        self.shape = shape
        self.torch_device = torch_device
        self.world = self.world_def()
        self.world.malloc()
        self.sensors = ['com']
        self.organism = Organism(self.world,
                                 sensors = self.sensors,
                                 n_actuators = self.world.windex['com'].shape[0])
        self.actions = None

    def world_def(self):
        return World(
            shape=self.shape,
            torch_dtype=torch.float32,
            torch_device=self.torch_device,
            channels={
                "com": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            },
       )


ti.init(ti.metal)

w, h = 400, 400
n_ch = 3
n_im = n_ch//3  # 3 channels per image, nim images next to each other widthwise
cell_size = 2
img_w, img_h = w * cell_size * n_im, h * cell_size
render_buffer = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=(img_w, img_h)
        )

@ti.kernel
def add_one(
        pos_x: ti.f32,
        pos_y: ti.f32,
        radius: ti.i32,
        mem: ti.types.ndarray()
    ):
    ind_x = int(pos_x * w)
    ind_y = int(pos_y * h)
    offset = int(pos_x * n_im) * 3
    for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
        for ch in ti.static(range(3)):
            if (i**2) + j**2 < radius**2:
                mem[0, offset+ch, (i + ind_x * n_im) % w, (j + ind_y) % h] +=1

@ti.kernel
def write_to_renderer(mem: ti.types.ndarray()):
    for i, j in render_buffer:
        subimg_index = i // (w * cell_size)
        offset = subimg_index * 3
        xind = (i//cell_size) % w
        yind = (j//cell_size) % h
        for ch in ti.static(range(3)):
            render_buffer[i, j][ch] = mem[0, offset+ch, xind, yind]

ein = NCA(shape=(w, h), torch_device=torch.device("mps"))

window = ti.ui.Window('NCA Visualization', (w, h), fps_limit=200, vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()
paused = False
brush_radius = 3
perturbing_weights = False
perturbation_strength = 0.1
drawing = False
while window.running:
    if not paused:
        ein.world.mem = ein.organism.forward(ein.world.mem)
        for e in window.get_events(ti.ui.PRESS):
            if e.key in  [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and window.is_pressed(ti.ui.SHIFT):
                drawing = True
            elif e.key == ti.ui.SPACE:
                ein.world.mem *= 0.0
        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                drawing = False
                
        if drawing:
            pos = window.get_cursor_pos()
            add_one(pos[0], pos[1], brush_radius, ein.world.mem)
        write_to_renderer(ein.world.mem)

        if perturbing_weights:
            ein.organism.perturb_weights(perturbation_strength)

    canvas.set_background_color((1, 1, 1))
    opt_w = min(480 / img_w, img_w)
    opt_h = min(180 / img_h, img_h)
    with gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
        brush_radius = sub_w.slider_int("Brush Radius", brush_radius, 1, 200)
        perturbation_strength = sub_w.slider_float("Perturbation Strength", perturbation_strength, 0.0, 5.0)
        paused = sub_w.checkbox("Pause", paused)
        perturbing_weights = sub_w.checkbox("Perturb Weights", perturbing_weights)
    canvas.set_image(render_buffer)
    window.show()

while window.running:
    ein.world.mem = ein.organism.forward(ein.world.mem)

    write_to_renderer(ein.world.mem)
    canvas.set_image(render_buffer)

    window.show()

# vis = Vis(w, [('com', 'r'), ('com', 'g'), ('com', 'b')])

# %% ------------------
# def update():
#     ein.world.mem = ein.organism.forward(ein.world.mem)

# t = Timer(update)

# # Measure time taken for 1000 calls
# time_taken = t.timeit(number=1000)

# # Calculate FPS
# fps = 1000 / time_taken

# print(f"Frames per second: {fps}")

# %% ------------------

# while vis.window.running:
#     vps = vis.params[None]
#     if vps.is_perturbing_weights:
#         ein.organism.perturb_weights(vps.perturb_strength)
#         ein.world.mem = ein.organism.forward(ein.world.mem)
#     vis.update()