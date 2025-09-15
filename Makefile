# Makefile for easy test execution
.PHONY: test lint smoke install-dev

# Install development dependencies
install-dev:
	pip install -e ".[dev]"

# Run smoke tests only (fastest) no ocverage
smoke:
	pytest tests/test_imports.py tests/test_cli_smoke.py -v --no-cov

# Run linting
lint:
	ruff check src/ --fix
	ruff format src/

# Run all tests
test:
	pytest tests/ -v

# Run everything
check: lint smoke
	@echo "✅ Basic checks passed!"