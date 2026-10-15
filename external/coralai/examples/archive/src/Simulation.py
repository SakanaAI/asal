import torch
import numpy as np
from src.Rule import Rule
from src.Channel import Channel
import src.utils as utils

class Simulation:
    def __init__(
            self,
            id: str = "default",
            world_shape: tuple = (100, 100),
            metadata: dict = {},
            device: torch.device = torch.device("cpu"),
            verbose: bool = False
        ):
        self.id = id
        self.world_shape = world_shape
        self.world_layers = 0
        self.channels = {}
        self.rules = {}
        # self.visualizers = {}
        self._deferred_groups = {}
        self.device = device
        self.verbose = verbose

        self.metadata = {
            'id': self.id,
            'world_shape': self.world_shape,
        }
        self.metadata.update(metadata)

    def add_channel(self, id: str, init_func: callable = None, num_layers: int = 1, metadata: dict={},
                    allowed_range: tuple = None):
        if id in self.channels:
            raise ValueError(f"Channel with ID {id} already exists.")
        self.world_layers += num_layers
        shape = (num_layers, *self.world_shape)
        self.channels[id] = Channel(id, shape, init_func, metadata, allowed_range, verbose=self.verbose)
    
    def add_subchannel(self, id: str, parent_id: str, indices: list[int], metadata: dict={}):
        if id in self.channels:
            raise ValueError(f"Channel with ID {id} already exists, cannot add as subchannel.")
        if parent_id not in self.channels:
            raise ValueError(f"Parent channel with ID {parent_id} does not exist, cannot add subchannel.")
        if isinstance(indices, int):
            indices = [indices]
        for index in indices:
            if index < 0 or index >= self.channels[parent_id].shape[0]:
                raise ValueError(f"Index {index} is out of bounds for parent channel {parent_id} with {self.channels[parent_id].shape[0]} layers.")

        parent_channel = self.channels[parent_id]
        parent_subchannels = parent_channel.metadata.get('subchannels', [])
        parent_subchannels.append(id)
        metadata['parent_channel'] = parent_channel
        metadata['parent_indices'] = indices
        metadata['parent_id'] = parent_id
        metadata['is_subchannel'] = True
        shape = (len(indices), *self.world_shape)
        init_func = lambda shp, md: (md['parent_channel'].contents[indices], md)
        self.channels[id] = Channel(id, shape, init_func, metadata, parent_channel.allowed_range, verbose=self.verbose)
        # parent_channel.add_subchannel(self.channels[id])

    # TODO: add channels not of world shape and add associations to other world/other channels (for hierarchical channels)
    # EG: define a set of NN channels (and their genomes) that are associated with a world channel via a genome map

    def add_rule(self, id, function, input_channel_ids: list[str] = [], affected_channel_ids: list[str] = [],
                            metadata: dict = None, req_sim_metadata: dict = {},
                            req_channel_metadata: dict = {}):
        if id in self.rules:
            raise ValueError(f"Update function with ID {id} already exists.")
        rule = Rule(id, function, input_channel_ids, affected_channel_ids,
                                     metadata, req_sim_metadata, req_channel_metadata, verbose=self.verbose)
        rule.assert_compatability(sim=self)
        self.rules[id] = rule
    
    def add_visualizer(self, visualizer):
        visualizer.assert_compatibility(sim=self)
        self.visualizers[visualizer.id] = visualizer
    
    def init_all_channels(self):
        for channel in self.channels.values():
            try:
                if not channel.initialized:
                    if channel.metadata.get('is_subchannel', False):
                        channel.metadata['parent_channel'].init_contents()
                    channel.init_contents()
            except Exception as e:
                if self.verbose:
                    channel_info = repr(self.channels[channel.id])
                else:
                    channel_info = str(self.channels[channel.id])
                raise RuntimeError(f"An error occurred while initializing channel {channel.id} " +
                                   channel_info +
                                   f"\nError:\n{e}") from e

    def apply_all_rules(self):
        for _, rule in self.rules.items():
            rule.apply(sim=self)
    
    def apply_rule(self, rule_id: str):
        if rule_id not in self.rules:
            raise ValueError(f"Rule with ID {rule_id} does not exist.")
        try:
            self.rules[rule_id].apply(sim=self)
        except Exception as e:
            if self.verbose:
                rule_info = repr(self.rules[rule_id])
            else:
                rule_info = str(self.rules[rule_id])
            raise RuntimeError(f"An error occurred while applying rule {rule_id} " +
                               rule_info +
                               f"\nError:\n{e}") from e

    def __repr__(self):
        channel_str = "-------------------------------------------------------------------------------\n"
        for channel in self.channels.values():
            channel_str += f"\n---\n{channel.id}\n\n{repr(channel)}\n---\n"

        rule_str = "-------------------------------------------------------------------------------\n"
        for rule in self.rules.values():
            rule_str += f"\n---\n{rule.id}\n\n{repr(rule)}\n---\n"
        
        return (
            f"Simulation(\nid: {self.id}\n"
            f"Metadata:\n{utils.dict_to_str(self.metadata)}\n"
            f"---\nChannels:\n{channel_str}\n"
            f"---\nRules:\n{rule_str}\n"
        )
