#!/bin/sh
# Run from any directory with a prepared novelty environment and cached model.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
case "${1:-}" in
    smoke) population=4; iterations=1 ;;
    full) population=8; iterations=8 ;;
    *) printf '%s\n' 'Usage: PYTHON=/path/to/python sh reproduce.sh smoke|full FRESH_OUTPUT_DIR' >&2; exit 2 ;;
esac
output_dir=${2:?Pass a fresh output directory}
cd "$repo_dir"
if [ -e "$output_dir" ]; then
    printf '%s\n' "Output already exists: $output_dir" >&2
    exit 2
fi
if [ -n "${PYTHON:-}" ]; then
    python_bin=$PYTHON
elif [ -x "$repo_dir/.venv-novelty/bin/python" ]; then
    python_bin=$repo_dir/.venv-novelty/bin/python
else
    python_bin=python3
fi
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TI_CPU_MAX_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$repo_dir/.cache/issue37/matplotlib"
export TI_OFFLINE_CACHE_FILE_PATH="$repo_dir/.cache/issue37/taichi"
export PYTHONPATH="$repo_dir:$repo_dir/external/coralai:$repo_dir/external/coralai/coralai/dependencies/PyTorch-NEAT"
exec "$python_bin" -B main_opt.py \
    --substrate coralai --coralai_objective temporal_novelty_multiseed \
    --coralai_config configs/dynamic_energy_demo/dynamic.json \
    --coralai_search_config configs/asal_novelty_pilot/search_multiseed.json \
    --coef_prompt 0 --coef_oe 1 --time_sampling 10 --seed 17 \
    --pop_size "$population" --n_iters "$iterations" --sigma 0.25 \
    --save_dir "$output_dir"
