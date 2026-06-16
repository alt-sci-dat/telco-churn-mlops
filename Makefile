# A Makefile gives you short, memorable commands for common tasks. Instead of
# remembering long invocations, you type `make train`, `make test`, etc.
# Run `make help` to see everything.

.PHONY: help install train test lint serve docker-build docker-run mlflow clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create a venv and install everything (dev + runtime)
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt && pip install -e .

train:  ## Train the model (override trials: make train TRIALS=50)
	python -m churn.train --n-trials $(or $(TRIALS),30)

test:  ## Run the test suite
	pytest

lint:  ## Lint with ruff
	ruff check .

serve:  ## Run the API locally with auto-reload
	uvicorn app.main:app --reload --port 8000

mlflow:  ## Open the MLflow experiment-tracking UI at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:  ## Build the Docker image
	docker build -t telco-churn-api .

docker-run:  ## Run the Docker image on port 8000
	docker run --rm -p 8000:8000 telco-churn-api

clean:  ## Remove caches and the local MLflow store
	rm -rf .pytest_cache .ruff_cache __pycache__ src/*.egg-info mlruns
