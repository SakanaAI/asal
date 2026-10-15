# Coralai + ASAL

CSC 4880 project connecting Coralai ecosystem simulations to ASAL supervised target search.

## Current progress
- Added Coralai source under external/coralai/.
- Organized existing experiment results under results/.
- Collected project documentation under docs/.
- Defined the [Coralai pilot configuration and result contract](docs/coralai_pilot_contract.md), with JSON schemas, a default configuration, and an illustrative result.

## Pilot format checks

The shared format can be validated without running Coralai or downloading CLIP.
Use a Python environment with the existing `jsonschema==4.23.0` dependency:

```sh
python scripts/validate_coralai_contract.py config configs/coralai_pilot.default.json
python scripts/validate_coralai_contract.py result examples/coralai_pilot_result.json
python -m unittest discover -s tests -p 'test_coralai_contract.py'
```

The config-driven runner and target search are planned in issues #3 and #4.

See [pilot validation](docs/coralai_pilot_validation.md) for saved-artifact checks,
reproduction commands, and [the recorded checks](results/week8/validation/README.md).

## Next steps
- Compute Shannon and discretized entropy on saved runs.
- Connect Coralai simulation frames to ASAL scoring.

## Upstream projects
- ASAL: https://github.com/SakanaAI/asal
- Coralai: https://github.com/aidanbx/coralai
