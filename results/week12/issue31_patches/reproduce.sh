#!/bin/sh
# Run from the repository root with an activated Python 3.11 CPU environment.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
ISSUE31_PYTHON=${ISSUE31_PYTHON:-python}
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/external/coralai:$PWD/external/coralai/coralai/dependencies/PyTorch-NEAT:$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -ne 1 ]; then
    echo "Usage: sh results/week12/issue31_patches/reproduce.sh --check | FRESH_OUTPUT_DIRECTORY" >&2
    exit 2
fi
"$ISSUE31_PYTHON" -B - <<'PY'
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import jax
import coralai.evolver as evolver
from huggingface_hub import snapshot_download
root = Path.cwd()
folder = root / "results/week12/issue31_patches"
record = json.loads((folder / "execution.json").read_text())
source = Path(evolver.__file__).resolve()
assert source == root / "external/coralai/coralai/evolver.py", source
for rel, sha in record["source_sha256"].items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == sha, rel
for name, expected in record["versions"].items():
    assert version(name) == expected, (name, version(name), expected)
for filename in ("default", "search"):
    saved = "default_config.json" if filename == "default" else "search_settings.json"
    assert json.loads((root / f"configs/coralai_supervised.{filename}.json").read_text()) == json.loads((folder / saved).read_text())
settings = json.loads((folder / "search_settings.json").read_text())
model = Path(snapshot_download(settings["model"], revision=settings["revision"], local_files_only=True))
weights = hashlib.sha256((model / "flax_model.msgpack").read_bytes()).hexdigest()
assert weights == json.loads((folder / "scorer.json").read_text())["weights_sha256"]
assert all(device.platform == "cpu" for device in jax.devices())
print("Preflight passed: repository controller, source/config hashes, package versions, pinned model, CPU.")
PY
"$ISSUE31_PYTHON" -B main_opt.py --help >/dev/null
if [ "$1" = "--check" ]; then
    exit 0
fi
if [ -e "$1" ]; then
    echo "Choose a fresh output directory: $1 already exists." >&2
    exit 2
fi
exec "$ISSUE31_PYTHON" -B main_opt.py \
    --substrate coralai \
    --coralai_config configs/coralai_supervised.default.json \
    --coralai_search_config configs/coralai_supervised.search.json \
    --save_dir "$1" --pop_size 6 --n_iters 5
