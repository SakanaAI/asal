# Pilot Validation

These checks confirm that our existing saved files and scoring scripts reproduce the Week 7 results.

## Validation results
- Config and result format commands: Passed.
- Contract and artifact unit tests: 24 passed.
- Saved patch checkpoint rendering: Exact pixel match with the Week 7 image.
- Saved patch checkpoint entropy: Matches all recorded values.
- Ten saved runs at step 1000: All 10 run rows and 10 summary rows match within `1e-12`.
- CPU CLIP score on the rendered image: Exact match at `-0.246830552816391` loss.

## Reference checkpoint details
- Checkpoint path: `data/raw/runs/coral_dev_20261001_174547/checkpoint_0001000/substrate.pt`
- Settings: `100x100` grid, seed `42`, and `8` energy patches.
- Genome entropy: `5.735160218300853` bits across living cells only.
- Energy entropy: `2.7636674667850962` bits across the whole grid with `1.0` bin width.
- Infrastructure entropy: `1.8909831577412344` bits across the whole grid with `1.0` bin width.

## Files and environment
- `reproduction_checks.json`: Commands used, comparison results, file hashes, and package versions.
- `requirements-used.txt`: Package versions from the CPU Python environment used for these checks.
- Raw validation outputs are saved in `data/raw/validation/week8/` (ignored by Git) so the files in `results/week7/` stay unchanged.

## Notes
- Matching saved images and scores confirms that the evaluation scripts work on existing data, though it does not test running a fresh simulation or resuming from a checkpoint.
- The artifact-validation tests were not tested on actual pilot runs yet and instead used small synthetic tensors to check failure handling and comparisons, but the validation pipeline should work as expected once the live runner and search are implemented.
- See `docs/coralai_pilot_validation.md` for the commands to run these saved-data checks and validate pilot `result.json` files.