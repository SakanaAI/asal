"""
Start environments for the coral (thesis) experiment.

The thesis version only needs the flat (homogeneous) default.
Non-flat environments are available in coral_dev.
"""

from coralai.environment import StartEnvironment


class FlatEnvironment(StartEnvironment):
    """Homogeneous environment — no spatial structure (thesis default)."""
    name = "flat"


ENVIRONMENTS = {
    "flat": FlatEnvironment,
}


def make_env(env_name: str = "flat", param=None) -> StartEnvironment:
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"coral experiment only supports env='flat'. Got: {env_name!r}")
    return ENVIRONMENTS[env_name]()
