# Makefile for easy test execution
.PHONY: test lint format smoke install-dev doc publish help

default: help

MAPSERVER_OUTPUT ?= mapserver_new
STYLES_DIR ?= ${HOME}/DATA/Derivations/output/R16/2026-01-28/styles/



install-dev: ## Install development dependencies
	pip install -e ".[dev]"


smoke: # Smoke tests only (fastest) no coverage
	pytest tests/test_imports.py tests/test_cli_smoke.py -v --no-cov


format: ## Format code
	ruff format src/


lint: format ## Linting
	ruff check src/ --fix


test: ## Run all tests
	pytest tests/ -v

doc:  ## Generating doc
	pdoc src/gcover  gcover.config.models gcover.publish.style_config  gcover.publish.esri_classification_extractor  --docformat google


check: lint smoke ## Linting and tests
	@echo "✅ Basic checks passed!"

publish: ## Publish as mapfiles
	gcover --env production publish mapserver    \
        --use-symbol-field  \
        --output-dir $(MAPSERVER_OUTPUT)  \
        --generate-combined \
        --styles-dir $(STYLES_DIR) \
        --pattern-file config/patterns_catalog.yaml  \
        --no-scale \
        --gml-items label  \
        config/esri_classifier_denormalized_geocover.yaml

# Show help
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
        awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
