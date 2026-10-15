import json

def check_subdict(check_dict, req_dict):
    if isinstance(req_dict, dict):
        for k, v in req_dict.items():
            if k not in check_dict:
                return False
            
            # If the value is a dictionary or list, recursively check
            if isinstance(v, (dict, list)):
                if not check_subdict(check_dict[k], v):
                    return False
            # If the value is neither a dict nor a list, it's checking type compatibility
            else:
                if not isinstance(check_dict.get(k), v):
                    return False
    elif isinstance(req_dict, list):
        if not isinstance(check_dict, dict):
            return False
        for item in req_dict:
            if item not in check_dict:
                return False
    else:
        # If it's neither a list nor a dictionary, it's a base-level type.
        # This case should never be reached due to the above logic, but is kept for completeness.
        if not isinstance(check_dict, req_dict):
            return False
            
    return True

def test_check_subdict():
    # Example use
    sim_metadata = {"a": {"a1a": 2, "a1b": 3}, "b": 4, "c": "5"}
    req_sim_metadata1 = {"a": ["a1a", "a1b"], "b": int}
    req_sim_metadata2 = ["a", "c"]
    req_sim_metadata3 = {"a": {"a1a": ["a1a1a"]}}

    assert check_subdict(sim_metadata, req_sim_metadata1)  # True
    assert check_subdict(sim_metadata, req_sim_metadata2)  # True
    assert not check_subdict(sim_metadata, req_sim_metadata3)  # False


def dict_to_str(metadata):
    return json.dumps(metadata, default=lambda o: repr(o), indent=2).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '\"')