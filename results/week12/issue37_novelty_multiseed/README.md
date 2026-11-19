# Issue 37 temporal-novelty results

- Source: `cfd00c23964c41486fcc4ba9fe33a2401a385807`; original 48×48, 200-update dynamic-energy protocol retained.
- Search: 64 proposals, 45 unique rounded configurations; seeds 303/7/13; optimizer seed 17, population 8, eight iterations, sigma 0.25.
- Runtime: smoke 2m35s; full 15m53s. Completed raw-artifact validation is retained in `../issue37_execution/`.
- Minimize mean temporal occupancy-CLIP loss plus sample standard deviation. Selected interval 26/shift 6 improved the criterion from 0.895192434 to 0.893072908.
- The improvement did not hold up: selected improved on 1/3 unseen seeds (101/202/404); mean hand-minus-selected raw loss was −0.000248611.
- `selection.json`, `candidates.csv`/`candidates.json`, and per-rollout scores/configurations/embeddings substantiate selection; `heldout/` contains paired results.
- `comparison.csv`, `trajectory.png`, and `comparison.gif` show search seed 303, not held-out evidence. Cell count and spatial behavior are distinct; this finite test does not establish indefinite open-endedness.
- Exact commands, runtimes, source/model hashes, versions, and the curated inventory are in `../issue37_execution/` (`sharing_files.txt`); native `execution.json` and `requirements-used.txt` preserve provenance.
- Reproduce: `PYTHON=/path/to/python sh results/week12/issue37_execution/reproduce.sh full results/week12/issue37_full_new` from the repository root; use `smoke` first with another fresh output path.
- Requires a prepared Python 3.11 CPU environment with recorded package versions and cached pinned CLIP revision (`scorer.json`). Local paths in `requirements-used.txt` are provenance, not portable installation instructions.
- `scripts/setup_novelty_env.sh` provides setup; fresh installation was not tested. The portable entry point was checked with a stub interpreter, without rerunning simulations.
- Compact check: `python results/week12/issue37_execution/verify_compact.py` (NumPy). Prior `verify_results.py` additionally needs complete local raw artifacts and recorded paths; excluded data remain local.
- Known limitation: held-out evaluation is skipped when the hand configuration wins; this case did not occur here. CPU scheduling may cause small reproduction differences.
