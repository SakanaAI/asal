import json
import warnings
import torch
import os
import taichi as ti
import numpy as np
from .utils.ti_struct_factory import TaichiStructFactory
from .channel import Channel
from .substrate_index import SubstrateIndex


@ti.data_oriented
class Substrate:
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

    def save_metadata_to_json(self, filepath):
        """
        Saves the substrate configuration (channels, channel metadata, dimensions, dtypes, etc.) to a file.
        """
        config = {
            "shape": self.shape,
            "torch_dtype": str(self.torch_dtype),
            "torch_device": str(self.torch_device),
            "channels": {chid: {"ti_dtype": str(ch.ti_dtype), **ch.metadata} for chid, ch in self.channels.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    def save_mem_to_pt(self, filepath):
        # Saves channels, channel metadata, dims, dtypes, etc
        torch.save(self.mem, filepath)

    def index_to_chname(self, index):
        return self.windex.index_to_chname(index)


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
        self.windex = SubstrateIndex(index_tree)
        self.ti_indices = self.ti_ind_builder.build()
        self.ti_lims = self.ti_lims_builder.build()
        self.mem = self.mem.permute(2, 0, 1).unsqueeze(0).contiguous()
        self.shape = self.mem.shape


    def __getitem__(self, key):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot get {key}")
        val = self.mem[:, self.windex[key], :, :]
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
