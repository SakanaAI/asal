"""
Experiment base class.

An Experiment bundles everything needed to run and replay a simulation:
substrate channel layout, spatial kernel, NEAT config, physics pipeline,
and evolution strategy.

Subclass this in your experiment directory's experiment.py. Export a
module-level EXPERIMENT instance so coralai/replay.py can import it from
the snapshot without instantiation arguments:

    # In experiments/myexp/experiment.py:
    from coralai.experiment import Experiment

    class MyExperiment(Experiment):
        name = "myexp"
        channels = {...}
        kernel = [...]
        ...
        def run_physics(self, substrate, evolver): ...
        def run_evolution(self, substrate, evolver, step): ...

    EXPERIMENT = MyExperiment()
"""

import os

import torch


class Experiment:
    """Base class for CoralAI experiments.

    Subclass to define channel layout, kernel, and physics/evolution pipelines.
    The base class provides default make_substrate and make_evolver
    implementations; override make_vis or make_env to customise the GUI or
    starting conditions.
    """
    name: str = "unnamed"
    channels: dict = {}
    kernel: list = []
    dir_order: list = []
    sense_chs: list = []
    act_chs: list = []
    neat_config_filename: str = "neat.config"

    def make_substrate(self, shape, device):
        """Create and malloc a Substrate for this experiment."""
        from coralai.substrate import Substrate
        sub = Substrate(shape, torch.float32, device, self.channels)
        sub.malloc()
        return sub

    def make_evolver(self, substrate):
        """Create a SpaceEvolver wired to this experiment's config.

        Looks for neat.config in self._exp_dir, which subclasses must set to
        os.path.dirname(os.path.abspath(__file__)) at module level. This
        ensures the correct snapshot copy is used during replay.
        """
        from coralai.evolver import SpaceEvolver
        exp_dir = getattr(self, "_exp_dir", None)
        if exp_dir is None:
            raise AttributeError(
                f"{type(self).__name__} must define a class attribute:\n"
                "    _exp_dir = os.path.dirname(os.path.abspath(__file__))")
        config_path = os.path.join(exp_dir, self.neat_config_filename)
        return SpaceEvolver(config_path, substrate, self.kernel,
                            self.dir_order, self.sense_chs, self.act_chs)

    def make_env(self, env_name: str = "flat", param=None):
        """Return a StartEnvironment instance for this experiment.

        Override in subclass to support experiment-specific environments.
        Default returns a flat (no-op) environment.
        """
        from coralai.environment import StartEnvironment
        return StartEnvironment()

    def make_vis(self, substrate, evolver):
        """Create a Visualization for the run GUI.

        Override in subclass to provide a richer panel with experiment-specific
        channel presets and stats. Default uses the first three sense channels.
        """
        from coralai.visualization import Visualization
        vis_chs = self.sense_chs[:3]
        return Visualization(substrate, vis_chs)

    def run_physics(self, substrate, evolver):
        """Apply one step of physics. Must be implemented by subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement run_physics()")

    def run_evolution(self, substrate, evolver, step: int):
        """Apply one step of evolution (noise injection, mutation, etc.).

        Must be implemented by subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run_evolution()")
