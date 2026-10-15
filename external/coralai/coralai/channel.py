import numpy as np
import taichi as ti

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