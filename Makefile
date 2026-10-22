.PHONY: install dev api worker test test-cov lint format typecheck eval eval-baseline docker-build docker-up docker-down clean seed help

PY := python
PIP := pip

install: ## Install runtime dependencies
	$(PIP) install -r requirements.txt

dev: ## Run the Streamlit UI
	$(PY) -m streamlit run app.py

api: ## Run the FastAPI service
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

worker: ## Run the background worker (single triage pass)
	$(PY) -m worker.scheduler --once

test: ## Run the test suite
	$(PY) -m pytest -q

test-cov: ## Run tests with coverage
	$(PY) -m pytest -q --cov=assistant --cov=api --cov-report=term-missing

lint: ## Lint the codebase
	$(PY) -m ruff check assistant api worker scripts
	$(PY) -m ruff format --check assistant api worker scripts

format: ## Format the codebase
	$(PY) -m ruff format assistant api worker scripts

typecheck: ## Type-check with mypy
	$(PY) -m mypy assistant api worker

eval: ## Run the offline evaluation harness
	$(PY) scripts/run_eval.py --offline

eval-baseline: ## Run eval and write a baseline for regression comparison
	$(PY) scripts/run_eval.py --offline --output eval-baseline.json

docker-build: ## Build the Docker image
	docker build -t ai-email-assistant .

docker-up: ## Start API + worker with Docker Compose
	docker compose up -d

docker-down: ## Stop Docker Compose services
	docker compose down

clean: ## Remove caches and local databases
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov *.db *.sqlite3

seed: ## Seed the database with sample emails
	$(PY) -c "from assistant.extractor import from_text; import json; from assistant.storage import get_db; db=get_db(); [db.upsert_email(from_text(**d)) for d in json.load(open('data/sample_emails.json'))]; print('seeded', db.count_emails(), 'emails')"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
