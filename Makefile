.PHONY: install lint test run-api run-pipeline run-all docker-up docker-down frontend-build

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check src tests

test:
	pytest

run-api:
	uvicorn mlforecast_realworld.api.main:app --host 0.0.0.0 --port 8000 --reload

run-pipeline:
	python scripts/run_pipeline.py

run-all: lint test frontend-build

frontend-build:
	cd frontend/angular-ui && npm run build

docker-up:
	docker compose up --build

docker-down:
	docker compose down
