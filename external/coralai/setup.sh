#!/usr/bin/env bash
# setup.sh — One-command setup for Coralai
# Usage:  ./setup.sh          (auto-detect conda or venv)
#         ./setup.sh conda    (force conda)
#         ./setup.sh venv     (force venv)
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[coralai]${NC} $*"; }
warn()  { echo -e "${YELLOW}[coralai]${NC} $*"; }
error() { echo -e "${RED}[coralai]${NC} $*"; exit 1; }

MODE="${1:-auto}"

# ── Detect environment manager ──────────────────────────────────────
if [ "$MODE" = "auto" ]; then
    if command -v conda &>/dev/null; then
        MODE="conda"
    else
        MODE="venv"
    fi
fi

info "Setting up Coralai with: $MODE"

# ── Conda path ──────────────────────────────────────────────────────
if [ "$MODE" = "conda" ]; then
    if ! command -v conda &>/dev/null; then
        error "conda not found. Install Miniconda/Anaconda first, or run: ./setup.sh venv"
    fi

    ENV_NAME="coralai"
    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Conda env '$ENV_NAME' already exists. Updating..."
        conda env update -f environment.yml --prune
    else
        info "Creating conda env '$ENV_NAME'..."
        conda env create -f environment.yml
    fi

    info ""
    info "Setup complete! Activate with:"
    info "  conda activate coralai"
    info ""
    info "Then run:"
    info "  make run-minimal    # headless NCA demo"
    info "  make run-xor        # XOR evolution demo"
    info "  python headless_repl.py --experiment minimal --shape 64"

# ── Venv path ───────────────────────────────────────────────────────
elif [ "$MODE" = "venv" ]; then
    PYTHON=""
    for cmd in python3.12 python3.11 python3.10 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            major=$("$cmd" -c "import sys; print(sys.version_info.major)")
            minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
            if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
                PYTHON="$cmd"
                break
            fi
        fi
    done
    [ -z "$PYTHON" ] && error "Python 3.10+ required. Found none."
    info "Using $PYTHON ($($PYTHON --version))"

    if [ ! -d ".venv" ]; then
        info "Creating virtual environment..."
        $PYTHON -m venv .venv
    fi
    source .venv/bin/activate

    info "Installing dependencies..."
    pip install --upgrade pip setuptools wheel -q
    pip install -e "coralai/dependencies/PyTorch-NEAT" -q
    pip install -e ".[dev]" -q

    info ""
    info "Setup complete! Activate with:"
    info "  source .venv/bin/activate"
    info ""
    info "Then run:"
    info "  make run-minimal    # headless NCA demo"
    info "  make run-xor        # XOR evolution demo"
    info "  python headless_repl.py --experiment minimal --shape 64"

else
    error "Unknown mode: $MODE. Use 'conda' or 'venv'."
fi
