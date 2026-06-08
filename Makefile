.PHONY: setup test lint typecheck check run clean format

PYTHON = python
PIP = pip

setup:
	$(PIP) install -r requirements.txt --break-system-packages 2>/dev/null || $(PIP) install -r requirements.txt

lint:
	ruff check . --fix

format:
	ruff format .

typecheck:
	mypy agent/ rag/ workflow/ mcp_server/ api/ config/ --ignore-missing-imports

test:
	pytest tests/ -v --tb=short

check: lint typecheck test

run:
	uvicorn api.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
