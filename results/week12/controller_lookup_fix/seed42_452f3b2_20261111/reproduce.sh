#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$repo_dir/external/coralai:$repo_dir/external/coralai/coralai/dependencies/PyTorch-NEAT:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TI_CPU_MAX_NUM_THREADS=1
exec "${TRAIL_PYTHON:-python}" -B scripts/demo_controller_lookup_fix.py --output-dir "${1:?Pass a fresh output directory}"
