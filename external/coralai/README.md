# Coralai: Emergent Ecosystems of Evolved Neural Cellular Automata

Coralai is a research framework for studying emergent collective behavior in evolving neural cellular automata. Each cell in a 2D toroidal grid runs a NEAT-evolved neural network, competing for energy and territory. Selection is purely spatial — cells that can't hold energy die, and successful strategies spread through directional colonization and in-place mutation. No fitness function. No explicit generations.

Builds on **EINCASM** (ALIFE 2023). Active development resumed 2026.

**Key design:**
- GPU-accelerated via [Taichi Lang](https://docs.taichi-lang.org/) + PyTorch (Mac Metal, CUDA, CPU)
- Continuous spatial NEAT — mutation and crossover happen as in-place radiation events
- Physics-as-experiment: the physics pipeline *is* the research question
- Pluggable experiments — each defines its own channel layout, physics, and NEAT config

---

## Setup

```bash
./setup.sh          # auto-detects conda or venv
conda activate coralai    # if conda
source .venv/bin/activate # if venv
```

Requires Python 3.10+. See `AGENTS.md` for full setup details and known issues.

---

## Running

```bash
# Interactive GUI (Mac Metal):
python experiments/coral_dev/run.py

# Headless — N steps, profiling breakdown:
python experiments/coral_dev/run.py --no-gui --steps 1000 --profile

# Benchmark:
python experiments/coral_dev/run.py --benchmark --steps 200 --shape 400

# Different starting environments:
python experiments/coral_dev/run.py --env hole --env-param 0.35
python experiments/coral_dev/run.py --env stripes --env-param 6

# CPU backend (slower but portable):
python experiments/coral_dev/run.py --backend cpu --device cpu --no-gui --steps 100

# Resume from a checkpoint (--steps counts additional steps from that point):
python experiments/coral_dev/run.py --resume-from runs/coral_dev_TIMESTAMP/checkpoint_0000500

# Replay saved run with GUI (navigate checkpoints with Reset/Prev/Next):
python -m coralai.replay --run-dir runs/coral_dev_TIMESTAMP
```

Run dirs are written to `runs/coral_dev_TIMESTAMP/` and contain:
- `meta.json` — experiment name, shape, seed, env, start time
- `initial_state.pt` — substrate state at step 0
- `step_log.csv` — per-step energy, infra, genome counts, FPS
- `snapshot/` — frozen copy of the experiment's source code (used by replay)
- `checkpoint_NNNNNNN/` — substrate + population + RNG state at that step

---

## Structure

```
coralai/                    ← repo root
│
├── experiments/            ← create and run experiments here
│   ├── _template/          ← copy this to start a new experiment
│   │   ├── run.py
│   │   ├── experiment.py   ← Experiment subclass + EXPERIMENT instance
│   │   ├── environments.py ← StartEnvironment subclasses
│   │   ├── physics.py
│   │   └── neat.config
│   ├── coral_dev/          ← active development experiment
│   │   ├── run.py          ← CLI entry point (thin driver)
│   │   ├── experiment.py   ← CoralDevExperiment + EXPERIMENT
│   │   ├── environments.py ← Flat/Hole/Ring/Stripes/Corners
│   │   ├── physics.py      ← physics pipeline
│   │   ├── evolution.py    ← radiation, culling helpers
│   │   └── neat.config
│   └── coral/              ← thesis frozen version (same structure)
│
├── coralai/                ← Python package (the engine — don't edit per-experiment)
│   ├── substrate.py        ← world memory manager
│   ├── evolver.py          ← SpaceEvolver + apply_weights_and_biases kernel
│   ├── experiment.py       ← Experiment base class
│   ├── environment.py      ← StartEnvironment base class
│   ├── replay.py           ← generic checkpoint replay GUI
│   ├── replay_utils.py     ← discover_checkpoints, load_experiment_from_snapshot
│   ├── visualization.py    ← Taichi GGUI real-time renderer
│   ├── channel.py
│   ├── nn_lib.py
│   ├── substrate_index.py
│   └── utils/
│       └── ti_struct_factory.py
│
├── tests/
│   └── smoke_test.py       ← 10 smoke tests (syntax, run, checkpoint, replay)
│
├── analysis/               ← post-hoc tools for understanding saved runs
│   ├── multiscale_complexity.py
│   └── population_analysis.py
│
├── pyproject.toml
├── Makefile
├── setup.sh
│
├── runs/                   ← gitignored — written by experiment runs
├── logs/                   ← session notes (human + AI)
└── archive/                ← frozen history
```

**To create a new experiment:** copy `experiments/_template/`. Subclass `Experiment` in `experiment.py`, define environments in `environments.py`, write physics in `physics.py`. The engine (`coralai/`) stays unchanged. Run with `python experiments/myexp/run.py` and replay any saved run with `python -m coralai.replay --run-dir runs/myexp_TIMESTAMP`.

---

## Architecture notes

For a full write-up of the architecture, design issues, and research direction see:
- `logs/2026-02-26 Thu/2026-02-26 Thu 19.21 coralai deep dive.md` — full system overview
- `logs/2026-02-26 Thu/2026-02-26 Thu 20.17 energy dynamics exploration and architecture.md` — physics analysis
- `logs/2026-02-26 Thu/2026-02-26 Thu 22.09 session capstone and next steps.md` — current state and priorities

---

## EINCASM

Predecessor to CoralAI, published at ALIFE 2023 in Sapporo, Japan. Featured muscles, ports, capital mining, and waste physics. Design preserved in `archive/instances/eincasm/`.

- **Paper:** https://direct.mit.edu/isal/proceedings/isal/35/82/116945
- **Talk:** https://www.youtube.com/watch?v=RuLQRgi6YSU&t=514s

---

aidanbx@gmail.com
