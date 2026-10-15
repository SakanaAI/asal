# Coralai Pilot Config and Result Format

Shared config and result format for the headless runner, supervised target search, and validation.&#x20;

## Files

- Default config: `configs/coralai_pilot.default.json`
- Config schema: `schemas/coralai_pilot_config.schema.json`
- Result schema: `schemas/coralai_pilot_result.schema.json`
- Example result: `examples/coralai_pilot_result.json`
- Validator script: `scripts/validate_coralai_contract.py`

## Format checks

Run these checks from the repo root. They require `jsonschema==4.23.0` but do not run Coralai or download CLIP.

```sh
python scripts/validate_coralai_contract.py config configs/coralai_pilot.default.json
python scripts/validate_coralai_contract.py result examples/coralai_pilot_result.json
python -m unittest discover -s tests -p 'test_coralai_contract.py'
```

In Python, use `load_json`, `validate_config`, and `validate_result` from `scripts/validate_coralai_contract.py` to load and validate files before running or scoring.

## Configuration settings

Copy `configs/coralai_pilot.default.json` and edit the values you need. All fields are required, and unknown fields are rejected.

### Simulation (`simulation`)

- `seed`: Random seed for Python, NumPy, and Torch. Default: `42`.
- `shape`: Grid side length. A value of `100` creates a `100 × 100` grid. Default: `100`.
- `steps`: Number of simulation steps to run. Default: `1000`.
- `env`: Environment type, either `"patches"` or `"flat"`. Default: `"patches"`.
- `env_param`: Number of energy patches for `"patches"`, or `null` for `"flat"`. Default: `8`.
- `backend` and `device`: Hardware pair, either `"cpu"/"cpu"`, `"cuda"/"cuda"`, or `"metal"/"mps"`. Default: `"cpu"/"cpu"`.
- `infra_decay`: Infrastructure fraction lost per step, from `0` to `1`. Default: `0.0`.
- `defense_coeff`: Infrastructure defense multiplier. Default: `0.0`.
- `radiate_interval`: Steps between radiation mutations. Default: `50`.
- `cleanup_interval`: Steps between extinct-genome cleanup. Default: `100`.
- `emergency_pop_cap`: Maximum live genomes, or `0` to disable. Default: `1000`.
- `checkpoint_interval`: Steps between saved checkpoints. Default: `100`.
- `log_interval`: Steps between log entries. Default: `10`.

For `"patches"` runs, the remaining environment settings use the experiment defaults: radius fraction `0.07`, capacity `30.0`, injection rate `0.06`, spawn interval `250`, and seed infrastructure `0.4`.

### Evaluation and output (`evaluation` and `output`)

- `evaluation.prompt`: Text prompt for ASAL CLIP scoring. Default: `"branching colonies"`.
- `evaluation.bin_width`: Bin width for resource entropy calculations. Default: `1.0`.
- `output.root`: Relative folder path for run outputs. Default: `"data/raw/pilot"`.

Each run saves its files in a new `<output.root>/<run_id>/` folder without overwriting past runs. During search, copy the full config for each candidate and change only the searched parameter so the seed, grid size, step count, and prompt stay the same.

## Runner result format

The runner saves a summary dictionary as `<run_dir>/result.json`. The file `examples/coralai_pilot_result.json` demonstrates the format; it is not a real simulation result.

- `schema_version`: Format version (`1`).
- `run_id`: Unique folder name for the run.
- `status`: `"success"` or `"failed"`.
- `effective_config`: Full config dictionary used for the run.
- `completed_steps`: Actual number of completed simulation steps.
- `artifacts.checkpoints`: List of `{completed_steps, directory}` entries for saved checkpoints.
- `artifacts.final_checkpoint`: Path to the final checkpoint folder, named `checkpoint_<steps padded to 7 digits>`, or `null` on failure.
- `artifacts.final_image`: Path to the rendered `224 × 224` PNG from the final checkpoint, or `null`.
- `artifacts.log`: Path to the step log CSV, or `null`.
- `artifacts.metadata`: Path to the run metadata JSON, or `null`.
- `provenance`: Git commit, dirty flag, hardware used, and package versions.
- `evaluation.clip`: CLIP score results added during evaluation, or `null`.
- `evaluation.entropy`: Entropy results added during evaluation, or `null`.
- `error`: `null` on success, or `{code, message}` if the run failed.

All paths inside `artifacts` are relative to the folder containing `result.json`. Always save the final checkpoint when a run finishes, even if `checkpoint_interval` is `0`.

## Rendering and scoring

The runner leaves `evaluation.clip` and `evaluation.entropy` as `null` so scoring can run separately on finished runs.

- Rendering (`substrates/coralai.py`): Converts the final `substrate.pt` state into a `224 × 224` RGB PNG. Red represents energy, green represents infrastructure, and blue shows living cells.
- CLIP scoring (`scripts/score_coralai.py`): Scores `final_image` against `evaluation.prompt` using `openai/clip-vit-base-patch32`. Records `model: "ASAL CLIP"`, `asal_target_loss`, and `alignment`, where `alignment = -asal_target_loss`. Lower loss is better.
- Entropy scoring (`scripts/score_entropy.py`): Scores the final checkpoint and records `alive_cells`, `occupied_genome_ids`, `genome_shannon_bits` for living cells, `energy_binned_entropy_bits` and `infrastructure_binned_entropy_bits` for the whole grid, and `bin_width`.
