# AGENTS.md

---

## Logs and Output Convention

All analysis, design thinking, and substantial AI responses belong in the **logs directory** at `logs/` inside this repo.

### Directory structure

```
logs/
├── YY-MM-DD ddd - Session Title.md         ← human session log (one per day, append-only)
└── YYYY-MM-DD ddd/                         ← AI output directory for that day
    └── YYYY-MM-DD ddd HH.MM descriptive title.md
```

### Human session log (`YY-MM-DD ddd - Title.md`)
- One file per day, named with short year + 3-letter weekday: `26-02-26 Thu - Title.md`
- Title represents the full day's content; update it as the day progresses
- Sections separated by `---`, ordered chronologically
- Captures questions, decisions, and directions to explore

### AI output files (`YYYY-MM-DD ddd/YYYY-MM-DD ddd HH.MM descriptive title.md`)
- Written to the dated subdirectory: `YYYY-MM-DD ddd/`
- Filename: full ISO 8601 date + ddd + `HH.MM` time + space-separated descriptive title
- Note: use `HH.MM` (period separator) not `HH:MM`; colons are invalid in filenames on macOS/Windows
- Example: `2026-02-26 Thu/2026-02-26 Thu 14.30 energy dynamics and architecture.md`
- Write a new file for any substantial analysis, design exploration, or technical write-up

---

## Project Overview

CoralAI is a Python research framework for simulating emergent ecosystems of evolved Neural Cellular Automata (NCA). It uses PyTorch for tensor operations, Taichi Lang for GPU-parallel kernels (Mac Metal, CUDA, or CPU), and NEAT-Python for neuroevolution. It was developed as a Master's Thesis (2024), building on EINCASM (ALIFE 2023).

**Active experiment**: `experiments/coral_dev/` — energy/infra economy with directional colonization.
**Active entry point**: `python experiments/coral_dev/run.py`
**Active evolver**: `coralai/evolver.py` (SpaceEvolver — continuous spatial NEAT, no explicit generations).
**Replay**: `python -m coralai.replay --run-dir runs/coral_dev_TIMESTAMP`

See `logs/2026-02-26 Thu/` for architecture write-ups and `logs/2026-02-28 Sat/` for the current refactor notes.

---

## Setup

Run all commands from the **repo root** (the directory containing `setup.sh` and `Makefile`).

### One-command install (auto-detects conda or venv)
```bash
./setup.sh            # auto
./setup.sh conda      # force conda
./setup.sh venv       # force venv
```

- **Conda** (recommended): creates/updates the `coralai` env from `environment.yml`
- **Venv**: requires Python 3.10+, installs from `requirements.txt`, installs the package with `pip install -e .`

PyTorch-NEAT is vendored at `coralai/dependencies/PyTorch-NEAT/` — no submodule init needed. It is installed as a local editable package via the `-e coralai/dependencies/PyTorch-NEAT` line in `requirements.txt`.

### Activate
```bash
conda activate coralai          # conda
source .venv/bin/activate       # venv (Linux/macOS)
.venv\Scripts\activate          # venv (Windows)
```

### Make targets
```bash
make help           # show all targets
make setup          # run setup.sh
make run-minimal    # headless REPL, minimal 1-channel NCA, 64×64
make run-nca        # headless REPL, NCA with RGB + hidden channels, 48×48
make run-coral      # headless REPL, coral ecosystem, 16×16
make run-xor        # XOR NEAT evolution (no GUI)
make demo           # non-interactive 300-step demo, saves video to runs/
make test           # run PyTorch-NEAT tests
make lint           # flake8 critical errors only
make clean          # remove runs/, history/, __pycache__, etc.
```

---

## Running

### GUI (Mac Metal)
```bash
python experiments/coral_dev/run.py
python experiments/coral_dev/run.py --env hole --env-param 0.35
python experiments/coral_dev/run.py --env stripes --env-param 6
```

### Headless
```bash
python experiments/coral_dev/run.py --no-gui --steps 1000
python experiments/coral_dev/run.py --no-gui --steps 1000 --profile
python experiments/coral_dev/run.py --benchmark --steps 200 --shape 400
python experiments/coral_dev/run.py --backend cpu --device cpu --no-gui --steps 100
```

### Checkpointing and resume
```bash
# Dense checkpoints for scrubbing in replay:
python experiments/coral_dev/run.py --no-gui --steps 1000 --checkpoint-interval 100

# Resume from a checkpoint (--steps = additional steps from that point):
python experiments/coral_dev/run.py --resume-from runs/coral_dev_TIMESTAMP/checkpoint_0000500

# Replay with GUI — Reset/Prev/Next buttons to navigate checkpoints:
python -m coralai.replay --run-dir runs/coral_dev_TIMESTAMP
python -m coralai.replay --run-dir runs/coral_dev_TIMESTAMP --step 500
```

Run output (`runs/coral_dev_TIMESTAMP/`):
- `meta.json` — shape, seed, env, start_step, timestamps
- `initial_state.pt` — substrate at step 0
- `step_log.csv` — per-step energy, infra, genome counts, FPS
- `snapshot/` — frozen copy of experiment source (replay uses this)
- `checkpoint_NNNNNNN/` — substrate.pt + population.pkl + meta.json

---

## Profiling and Benchmarking

### Steps per second
The headless REPL prints `steps/s` after every `step [N]` command and after `--auto` runs:
```bash
python headless_repl.py --experiment minimal --shape 64 --auto 1000
# Prints: "1000 steps in 4.23s (236 steps/s)"
```

Use `--steps-per-frame 10` or higher to reduce rendering overhead and get pure sim throughput:
```bash
python headless_repl.py --experiment minimal --shape 128 --auto 500 --steps-per-frame 10
```

### Comparing backends
The headless REPL uses `ti.cpu` and `torch.device("cpu")`. To benchmark Metal or CUDA, run the GUI runner or modify the device init at the top of `headless_repl.py`:
```python
ti.init(ti.metal)                    # Mac
ti.init(ti.cuda)                     # NVIDIA
TORCH_DEVICE = torch.device("mps")   # Mac
TORCH_DEVICE = torch.device("cuda")  # NVIDIA
```

### Substrate memory scaling
Grid size scales as O(W×H×C) where C is total channel count. For coral: C=15 channels, 400×400 grid ≈ 2.4M floats ≈ 9.6MB per substrate tensor. The stacked weight tensor for SpaceEvolver scales as O(n_genomes × n_acts × n_inputs) — with 100 genomes, n_acts=10, n_inputs=12 this is ~12K floats, negligible.

---

## Known Issues and Vestigial Code

### Broken / not runnable
- `coralai/instances/coral/coral_evolver.py` — empty stub, imports a deleted `simulation.evolver` module
- `coralai/instances/coral/dumb_test_org.py` — imports deleted `dynamics.Organism`
- `coralai/instances/eincasm/eincasm.py` — references `World`, `dynamics.pcg`, `dynamics.ein_physics`, all deleted

### Known design issues
- `activate_outputs` in coral_dev uses `tanh` for trade signals (replaces the old softmax invest/liquidate); cells can still "waste" signal. See `logs/2026-02-28 Sat/` for design rationale.
- `explore_physics` uses `argmax` on direction activations — discards magnitude, forces single-direction expansion per step
- No spatially-structured energy sources — energy injection is spatially homogeneous (uniform noise + sinusoidal offset)
- `LinearNet` in SpaceEvolver ignores NEAT hidden nodes — the network is always purely linear regardless of `num_hidden` in the config

### Taichi CPU gotcha
On the CPU backend, any cell with `genome = -1` causes out-of-bounds array access in `apply_weights_and_biases` (negative index into the weight array). The GPU backends handle negative indices differently (wrap/clamp). Always initialize genome values to valid keys (≥ 0) when using CPU.

### Replay non-determinism
Resuming from a checkpoint does not produce bitwise-identical results on MPS or CPU. Two sources: (1) Taichi atomic race conditions in `explore_physics` — thread scheduling is non-deterministic; (2) MPS parallel float reduction ordering. All RNG states (torch/MPS/numpy/python) are correctly saved and restored — the divergence is in kernel execution order. CPU is also non-deterministic for the same reason. Checkpoints are useful for exploring the state space but not for exact reproduction.

---

## Tests and Lint

```bash
python tests/smoke_test.py   # 10 smoke tests: syntax, imports, run, checkpoint, replay
make test    # runs test_cppn.py and test_multi_env_eval.py from PyTorch-NEAT
make lint    # flake8 E9/F63/F7/F82 (critical errors only)
```

- `tests/smoke_test.py` — 10 tests covering: syntax (12 files), replay_utils importable without Taichi, StartEnvironment base class, basic headless run + file structure, meta.json keys, snapshot contents, hole env persist, checkpointing, load_experiment_from_snapshot, replay one step without NaN. All pass.
- 5/6 PyTorch-NEAT tests pass on CPU-only (`test_cppn_unconnected` fails — hardcoded CUDA device in test, not a real issue)
- `make clean` removes `runs/`, `history/`, all `__pycache__`
