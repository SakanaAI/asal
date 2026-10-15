import numpy as np

class SubstrateIndex:
    """
    Returns indices of substrate channels in queried order.
    Usage:
    - substrate.windex['red'] (returns: [0])
    - substrate.windex[['energy', 'infra']] (returns: [0,1])
    - substrate.windex['rgb'] (returns: [1,2,3]) assuming rgb is a taichi vector or struct
    - substrate.windex[('acts', ['invest', 'liquidate'])] (returns: [4, 3]) assuming acts is a taichi struct
    """
    def __init__(self, index_tree):
        self.index_tree = index_tree

    def index_to_chname(self, index):
        for channel, details in self.index_tree.items():
            if index in details['indices']:
                if 'subchannels' in details:
                    for subchannel, subdetails in details['subchannels'].items():
                        if index in subdetails['indices']:
                            return f"{channel}_{subchannel}"
                    return "err??"
                else:
                    return channel
        return "Ch not found"

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
        