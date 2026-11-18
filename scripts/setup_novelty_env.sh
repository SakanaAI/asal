#!/bin/sh
# Create a separate, persistent Python 3.11 CPU environment; leave .venv intact.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime_dir="$repo_dir/.runtime-novelty"
venv_dir="$repo_dir/.venv-novelty"
mkdir -p "$runtime_dir"
if [ ! -x "$runtime_dir/python/bin/python3.11" ]; then
    archive="$runtime_dir/python.tar.gz"
    if [ -n "${NOVELTY_PYTHON_ARCHIVE:-}" ]; then
        cp "$NOVELTY_PYTHON_ARCHIVE" "$archive"
    else
        curl --fail --location --retry 2 --output "$archive" 'https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.11.18%2B20261110-x86_64-unknown-linux-gnu-install_only.tar.gz'
    fi
    printf '%s  %s\n' ac0b758b6ae4ba1cb87f7091eb6bb17c9a383a9bcdb5c18493cdd4f34e35ae8c "$archive" | sha256sum --check -
    tar -xzf "$archive" -C "$runtime_dir"
fi
if [ ! -x "$venv_dir/bin/python" ]; then
    "$runtime_dir/python/bin/python3.11" -m venv "$venv_dir"
fi
export PIP_CACHE_DIR="$repo_dir/.cache/novelty/pip"
"$venv_dir/bin/python" -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$venv_dir/bin/python" -m pip install -r "$repo_dir/configs/asal_novelty_pilot/requirements.lock.txt"
"$venv_dir/bin/python" -m pip install --no-deps "$repo_dir/external/coralai" "$repo_dir/external/coralai/coralai/dependencies/PyTorch-NEAT"
"$venv_dir/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', allow_patterns=['*.json','*.txt','flax_model.msgpack'])"
printf '\nReady: %s/bin/python\n' "$venv_dir"
