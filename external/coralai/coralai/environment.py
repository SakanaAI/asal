"""
StartEnvironment base class.

Defines how a substrate is initialized and optionally constrained each step.
Subclass in your experiment's environments.py. The snapshot captures that
file so replay always uses the env that generated the run.

Usage:
    # In your experiment's environments.py:
    from coralai.environment import StartEnvironment

    class HoleEnvironment(StartEnvironment):
        name = "hole"
        def seed(self, substrate): ...
        def step(self, substrate): ...
        @property
        def persist(self): return True
"""


class StartEnvironment:
    """Base class for experiment starting environments.

    Default is flat/homogeneous (seed and step are no-ops, persist=False).
    Subclass and override to add spatial structure: resource gradients, barren
    zones, periodic terrain, etc.
    """
    name: str = "flat"

    def seed(self, substrate):
        """Initialize substrate state. Called once after evolver setup.

        Override to seed resources only in habitable zones, cut out barren
        regions, place obstacles, etc. Default: no-op (substrate stays at
        whatever SpaceEvolver.init_substrate() left it as).
        """
        pass

    def step(self, substrate):
        """Apply per-step environmental constraints.

        Called every step only when persist=True. Use to enforce permanent
        spatial structure — e.g. zero energy/infra/genome in barren cells so
        organisms can't colonise them. Default: no-op.
        """
        pass

    @property
    def persist(self) -> bool:
        """If True, step() is called every iteration to re-enforce constraints.

        False (default): the environment only seeds the initial state; life
        can later colonise any zone. True: barren zones stay barren forever.
        """
        return False

    def to_dict(self) -> dict:
        """Serialisable description for meta.json. Override to include params."""
        return {"name": self.name}
