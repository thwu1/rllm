.PHONY: help install install-dev test test-cov lint format clean docker-build docker-run

help:
	@echo "RLLM LiteLLM Proxy - Development Commands"
	@echo ""
	@echo "  install       Install dependencies"
	@echo "  install-dev   Install dev dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo "  run           Run proxy server locally"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r tests/requirements.txt
	pip install black flake8 mypy isort

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=sdk.proxy --cov-report=html --cov-report=term

lint:
	flake8 sdk/proxy
	mypy sdk/proxy

format:
	black sdk/proxy
	isort sdk/proxy

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t rllm-proxy:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

run:
	python -m sdk.proxy.server

run-dev:
	uvicorn sdk.proxy.server:create_app --factory --reload --host 0.0.0.0 --port 8000
