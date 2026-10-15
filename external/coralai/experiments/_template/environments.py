"""
Template environments for this experiment.

Add environment classes here as you need spatial structure. The flat
(homogeneous) default is inherited from coralai.environment.StartEnvironment.

See experiments/coral_dev/environments.py for a complete example with hole,
stripes, and corners.
"""

from coralai.environment import StartEnvironment


class FlatEnvironment(StartEnvironment):
    """Homogeneous environment — no spatial structure (default)."""
    name = "flat"


ENVIRONMENTS = {
    "flat": FlatEnvironment,
}


def make_env(env_name: str = "flat", param=None) -> StartEnvironment:
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown env {env_name!r}. Available: {list(ENVIRONMENTS)}")
    return ENVIRONMENTS[env_name]()
