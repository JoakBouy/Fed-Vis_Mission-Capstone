.PHONY: install test lint format typecheck clean train-local train-federated inference

# Development
install:
	poetry install

test:
	poetry run pytest tests/ -v --cov=src/fedvis --cov-report=term-missing

lint:
	poetry run ruff check src/ tests/

format:
	poetry run ruff format src/ tests/

typecheck:
	poetry run mypy src/fedvis/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/

# Training
train-local:
	poetry run python -m fedvis.scripts.train_local data=$(DATASET)

train-federated:
	poetry run python -m fedvis.scripts.train_federated

# Inference
inference:
	poetry run python -m fedvis.scripts.inference checkpoint=$(CHECKPOINT)

# Docker
docker-build:
	docker build -f docker/Dockerfile.train -t fedvis:train .
	docker build -f docker/Dockerfile.inference -t fedvis:inference .

docker-fl:
	docker-compose up

# Pre-commit
pre-commit-install:
	poetry run pre-commit install

pre-commit-run:
	poetry run pre-commit run --all-files
