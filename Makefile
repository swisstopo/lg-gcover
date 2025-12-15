# Makefile for easy test execution
.PHONY: test lint format smoke install-dev doc

# Install development dependencies
install-dev:
	pip install -e ".[dev]"

# Run smoke tests only (fastest) no ocverage
smoke:
	pytest tests/test_imports.py tests/test_cli_smoke.py -v --no-cov


format:
	ruff format src/

# Run linting
lint: format
	ruff check src/ --fix

# Run all tests
test:
	pytest tests/ -v

doc:
	pdoc src/gcover --docformat google

# Run everything
check: lint smoke
	@echo "✅ Basic checks passed!"
