import numpy as np
import torch
import json
from pprint import pformat

class Channel:
    def __init__(self, id, shape, init_func=None, metadata: dict=None,
                 allowed_range=(-np.inf, np.inf), dtype=torch.float32,
                 device=torch.device("cpu"), verbose: bool = False):
        
        self.id = id
        self.shape = shape
        self.verbose = verbose
        self.subchannels = []
        self.parent_channel = None
        
        if init_func is None:
            init_func = lambda shape, md: (torch.zeros(shape, dtype=dtype, device=device), md)
        
        self.init_function = init_func
        self.allowed_range = allowed_range if allowed_range else (-np.inf, np.inf)
        
        default_metadata = {
            'id': id,
            'shape': self.shape,
            'allowed_range': self.allowed_range,
            'dtype': dtype,
            'device': device,
        }
        self.metadata = {**metadata, **default_metadata} if metadata else default_metadata

        self.dtype = dtype
        self.device = device
        self.initialized = False
        self.contents=torch.Tensor()    
    
    def init_contents(self):
        contents, init_metadata = self.init_function(self.shape, self.metadata)
        assert contents.shape == self.shape, f"init_function {self.id} must return tensor of shape {self.shape}, got {contents.shape}"
        assert contents.min() >= self.allowed_range[0], f"init_function for {self.id} must return tensor with min value >= {self.allowed_range[0]}, got {contents.min()}"
        assert contents.max() <= self.allowed_range[1], f"init_function for {self.id} must return tensor with max value <= {self.allowed_range[1]}, got {contents.max()}"
        
        self.metadata.update(init_metadata)
        self.contents = contents
        self.initialized=True
        return self.contents
    
    # def add_subchannel(self, id, slice):
    #     subchannel = Channel(id, )

    # def add_subchannel(self, channel):
    #     self.subchannels.append(channel)
    
    # def update_sub_channels(self, new_content=None):
    #     if new_content is None:
    #         new_content = self.contents
    #     for subchannel in self.subchannels:
    #         subchannel.contents = new_content[subchannel.metadata['parent_indices']]
    #         subchannel.initialized = True

    def __str__(self):
        return f"Channel(id={self.id}, shape={self.shape}, dtype={self.dtype}, device={self.device}, initialized={self.initialized})"
    
    def __repr__(self):
        contents_str = pformat(self.contents.tolist(), width=50, compact=True)
        if self.metadata.get('is_subchannel', False):
            new_metadata = self.metadata.copy()
            new_metadata["parent_channel"] = new_metadata["parent_channel"].id
            obj_title = "subChannel"
        else:
            new_metadata = self.metadata
            obj_title = "Channel"

        metadata_str = json.dumps(new_metadata, default=lambda o: repr(o), indent=2).replace("\\n", "\n")
        return (
            f"{obj_title}(\n"
            f"\tid={self.id},\n"
            f"\tshape={self.shape},\n"
            f"\tdtype={self.dtype},\n"
            f"\tdevice={self.device},\n"
            f"\tinitialized={self.initialized},\n"
            f"\tcontents=\n{contents_str},\n"
            f"\tmetadata={metadata_str}\n"
            f")"
        )