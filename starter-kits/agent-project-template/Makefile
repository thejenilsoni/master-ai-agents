# The only commands anyone needs. `make check` is what CI runs and what you run
# before pushing -- keeping them identical is the whole point.

.DEFAULT_GOAL := help
.PHONY: help setup check test lint fmt types run demo clean docker

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and dev tools
	pip install -e ".[dev]"

check: lint types test  ## Everything CI runs

test:  ## Run the tests (live evals are deselected)
	pytest

lint:  ## Check formatting and lint rules
	ruff check .
	ruff format --check .

fmt:  ## Fix what can be fixed automatically
	ruff format .
	ruff check --fix .

types:  ## Static type check
	mypy

run:  ## Ask a question (offline, no key needed)
	python -m agentapp.cli --offline "what is the weather in Bergen?"

demo:  ## Same, but show the full trace
	python -m agentapp.cli --offline --json "convert 250 to eur"

docker:  ## Build the container
	docker build -t agentapp:local .

clean:  ## Remove caches and build output
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
