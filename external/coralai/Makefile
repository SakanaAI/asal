.PHONY: setup run-minimal run-nca run-coral run-xor test lint clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install everything (auto-detects conda or venv)
	./setup.sh

setup-conda: ## Install using conda
	./setup.sh conda

setup-venv: ## Install using venv
	./setup.sh venv

run-minimal: ## Run minimal NCA experiment (interactive REPL)
	python headless_repl.py --experiment minimal --shape 64

run-nca: ## Run NCA experiment (interactive REPL)
	python headless_repl.py --experiment nca --shape 48

run-coral: ## Run coral ecosystem experiment (interactive REPL)
	python headless_repl.py --experiment coral --shape 16

run-xor: ## Run XOR NEAT evolution demo
	python xor_runner.py

demo: ## Run a quick auto demo (no interaction needed)
	python headless_repl.py --experiment minimal --shape 64 --auto 300 --fps 20
	@echo ""
	@echo "Video saved to runs/*/simulation.mp4"

test: ## Run tests
	python -m pytest coralai/dependencies/PyTorch-NEAT/tests/test_cppn.py \
		coralai/dependencies/PyTorch-NEAT/tests/test_multi_env_eval.py -v

lint: ## Run linter
	flake8 coralai/ headless_repl.py --select=E9,F63,F7,F82

clean: ## Remove generated files
	rm -rf runs/ neat_output/ history/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
