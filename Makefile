.PHONY: help install install-dev test test-cov lint format benchmark demo clean docs

# Default target
help:
	@echo "PyAlgo Development Commands"
	@echo "=========================="
	@echo "install      - Install production dependencies"
	@echo "install-dev  - Install development dependencies"
	@echo "test         - Run test suite"
	@echo "test-cov     - Run tests with coverage report"
	@echo "lint         - Run linting (flake8, mypy)"
	@echo "format       - Format code (black, isort)"
	@echo "benchmark    - Run performance benchmarks"
	@echo "demo         - Run algorithm demonstrations"
	@echo "clean        - Clean cache and build files"
	@echo "docs         - Generate documentation"

# Installation
install:
	poetry install --only=main

install-dev:
	poetry install --with=dev,benchmark

# Testing
test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ --cov=. --cov-report=html --cov-report=term

# Code quality
lint:
	poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	poetry run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	poetry run mypy . --ignore-missing-imports

format:
	poetry run black .
	poetry run isort .

# Performance and demos
benchmark:
	poetry run python benchmark.py

demo:
	poetry run python demo.py

# Maintenance
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Documentation (placeholder for future)
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Consider adding Sphinx or MkDocs in the future"
