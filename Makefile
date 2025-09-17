# Makefile for MoA Prediction Framework

.PHONY: help install install-dev test lint format clean data train predict

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting (flake8, mypy)"
	@echo "  format       - Format code (black)"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  data         - Download and process data"
	@echo "  train        - Train model with default configuration"
	@echo "  predict      - Run prediction on sample data"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,notebooks]"

# Testing and code quality
test:
	pytest tests/ -v --cov=moa --cov-report=html

lint:
	flake8 moa/ tests/
	mypy moa/

format:
	black moa/ tests/ scripts/
	isort moa/ tests/ scripts/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Data pipeline
data:
	python scripts/download_data.py
	python scripts/process_data.py

# Training and prediction
train:
	moa-train --config configs/config.yaml

predict:
	moa-predict --model models/best_model.ckpt --input data/sample_compounds.smi

# Development utilities
setup-dirs:
	mkdir -p data/{raw,processed,splits}
	mkdir -p models
	mkdir -p results
	mkdir -p cache
	mkdir -p logs

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Docker (if needed)
docker-build:
	docker build -t moa-prediction .

docker-run:
	docker run -it --rm -v $(PWD):/workspace moa-prediction
