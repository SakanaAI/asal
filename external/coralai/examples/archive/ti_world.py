import torch
import taichi as ti
import numpy as np
import warnings

class Channel:
    def __init__(
            self, id=None, dtype=ti.f32,
            init_func=None,
            lims=None,
            metadata: dict=None, **kwargs):
        self.id = id
        self.lims = lims if lims else (-1, 1)
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.init_func = init_func
        self.dtype = dtype

    def __getitem__(self, key):
        return self.metadata.get(key)

    def __setitem__(self, key, value):
        self.metadata[key] = value
            
@ti.data_oriented
class World:
    def __init__(self, shape, channels: dict=None, metadata: dict=None, **kwargs):
        self.shape = shape # shape is just (w,h), unlike torch world wich is (channels, w, h)
        self.channels = {}
        self.memory_allocated = False
        if channels is not None:
            self.add_channels(channels)
        self.mem = None

        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)

    def add_channel(self, id: str, dtype=ti.f32, **kwargs):
        if self.memory_allocated:
            raise ValueError("When adding channel {id}: Cannot add channel after world memory is allocated (yet).")
        self.channels[id] = Channel(id=id, dtype=dtype, **kwargs)

    def add_channels(self, channels: dict):
        if self.memory_allocated:
            raise ValueError("When adding channels {channels}: Cannot add channels after world memory is allocated (yet).")
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, Channel):
                 if ch.id is None:
                     ch.id = chid
                 self.channels[id] = ch
            elif isinstance(ch, dict):
                self.add_channel(chid, **ch)
            else:
                self.add_channel(chid, ch)    

    def malloc(self):
        if self.memory_allocated:
            raise ValueError(f"Cannot allocate world memory twice.")
        
        celltype = ti.types.struct(**{chid: self.channels[chid].dtype for chid in self.channels.keys()})
        self.mem = celltype.field(shape=self.shape[:2])
        self.memory_allocated = True
        return self.mem
    
    def __getitem__(self, key):
        if key in self.channels:
            return self.channels.get(key)
        else:
            return self.metadata.get(key)
    
    def __setitem__(self, key, value):
        if isinstance(value, Channel):
            if self.memory_allocated:
                raise ValueError(f"Cannot add channel {key} after world memory is allocated.")
            else:
                self.channels[key] = value
        else:
            self.metadata[key] = value