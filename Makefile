.PHONY: install demo benchmark test lint typecheck check docker-build docker-demo docker-test docker-lint db-up db-check db-down

install:
	python -m pip install -e ".[dev]"

demo:
	python -m examples.smoke_demo

benchmark:
	python -m benchmarks.latency --iterations 1000

test:
	python -m pytest

lint:
	python -m ruff check .

typecheck:
	python -m mypy inference

check: lint typecheck test

docker-build:
	docker build -t land-valuation-inference .

docker-demo:
	docker compose run --rm demo

docker-test:
	docker compose run --rm test

docker-lint:
	docker compose run --rm lint

db-up:
	docker compose up -d postgres

db-check:
	docker compose run --rm db-check

db-down:
	docker compose down
