"""Content provenance for headless CoralDev references, including uncommitted code."""

import ast
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys

CORALAI_ROOT = Path(__file__).resolve().parents[1]


def simulator_fingerprint():
    """Hash simulator/vendored source and NEAT configuration, independent of HEAD."""
    files = set((CORALAI_ROOT / "coralai").rglob("*.py"))
    files.update((CORALAI_ROOT / "experiments/coral_dev").glob("*.py"))
    files.add(CORALAI_ROOT / "experiments/coral_dev/neat.config")
    loaded = sys.modules.get("coralai.evolver")
    if loaded is not None and Path(loaded.__file__).resolve() != CORALAI_ROOT / "coralai/evolver.py":
        raise ValueError("Controller source is not this repository's evolver.py")
    return {"schema_version": 1,
            "source_sha256": {str(p.relative_to(CORALAI_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(files)},
            "package_versions": {name: version(name) for name in ("torch", "taichi", "neat-python", "numpy")},
            "python_version": list(sys.version_info[:3])}


def effective_configuration(config):
    """Use the runner's literal defaults without importing/executing its CLI."""
    tree = ast.parse((CORALAI_ROOT / "experiments/coral_dev/headless_coral.py").read_text())
    defaults = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in node.targets))
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown headless settings: {sorted(unknown)}")
    result = defaults | config
    result.pop("output_dir")  # Destination path has no simulation effect.
    return result
