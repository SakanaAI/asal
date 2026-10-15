from coralai import utils
# from src.Channel import Channel
import json
from pprint import pformat
import src.utils as utils

class Rule:
    def __init__(self, id: str, func: callable,
                 input_channel_ids: list[str] = [], affected_channel_ids: list[str] = [],
                 metadata: dict = None, req_sim_metadata: dict = {},
                 req_channel_metadata: dict = {}, verbose: bool = False):
        self.id = id
        self.func = func
        self.input_channel_ids = input_channel_ids
        self.affected_channel_ids = affected_channel_ids
        self.req_sim_metadata = req_sim_metadata
        self.req_channel_metadata = req_channel_metadata
        self.verbose = verbose
        
        default_metadata = {
            'id': id,
            'input_channel_ids': input_channel_ids,
            'affected_channel_ids': affected_channel_ids,
            'req_sim_metadata': req_sim_metadata,
            'req_channel_metadata': req_channel_metadata,
        }
        if metadata is None:
            metadata = {}
        metadata.update(default_metadata)
        self.metadata = metadata

    def assert_compatability(self, sim):
        for id in self.input_channel_ids:
            assert id in sim.channels.keys(), f"Input channel \"{id}\" for Rule \"{self.id}\" not in simulation \"{sim.id}\""
        for id in self.affected_channel_ids:
            assert id in sim.channels.keys(), f"Affected channel \"{id}\" for Rule \"{self.id}\" not in simulation \"{sim.id}\""
        
        utils.check_subdict(sim.metadata, self.req_sim_metadata)
        all_channel_metadata = {}
        for channel_id, channel in sim.channels.items():
            all_channel_metadata[channel_id] = channel.metadata
        utils.check_subdict(all_channel_metadata, self.req_channel_metadata)


    def apply(self, sim):
        if self.verbose:
            pass
            # print(f"Rule \"{self.id}\" updating...")
        try:
            self.func(sim, *[sim.channels[id] for id in self.input_channel_ids], self.metadata)
        except Exception as e:
            if self.verbose:
                extra_info = f"\nSim Metadata:\n{utils.dict_to_str(sim.metadata)}\n"
                extra_info += f"\n{self.__repr__()}"
            else:
                extra_info = ""
            raise e from RuntimeError(f"Error in Rule: \"{self.id}\"" +
                               extra_info +
                               f"\nError in Rule: \"{self.id}\"\n")
        
        for ch in self.affected_channel_ids:
            channel = sim.channels[ch]
            if channel.shape != channel.contents.shape:
                raise RuntimeError(f"Rule \"{self.id}\": Affected Channel \"{ch}\" has shape {channel.shape}, but updated contents have shape {channel.contents.shape}")

        for ch in self.affected_channel_ids:
            subchannels = sim.channels[ch].subchannels


    def __str__(self) -> str:
        return f"Rule(id={self.id}, input_channel_ids={self.input_channel_ids}, affected_channel_ids={self.affected_channel_ids})"
    
    def __repr__(self) -> str:
        metadata_str = utils.dict_to_str(self.metadata)
        input_channel_ids_str = pformat(self.input_channel_ids, width=50, compact=True)
        affected_channel_ids_str = pformat(self.affected_channel_ids, width=50, compact=True)
        req_sim_metadata_str = utils.dict_to_str(self.req_sim_metadata)
        req_channel_metadata_str = utils.dict_to_str(self.req_channel_metadata)
        return (
            f"Rule(\n"
            f"\tid={self.id},\n"
            f"\tinput_channel_ids={input_channel_ids_str},\n"
            f"\taffected_channel_ids={affected_channel_ids_str},\n"
            f"\treq_sim_metadata={req_sim_metadata_str},\n"
            f"\treq_channel_metadata={req_channel_metadata_str},\n"
            f"\tmetadata={metadata_str}\n"
            f")"
        )
