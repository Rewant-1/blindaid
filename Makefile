PYTHON ?= python
VENV ?= .venv

.PHONY: setup dev lint format mypy pytest test run clean

setup:
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@. $(VENV)/bin/activate && pip install -r requirements.txt

dev: setup
	@. $(VENV)/bin/activate && pip install -r requirements-dev.txt

lint:
	@. $(VENV)/bin/activate && ruff check blindaid tests
	@. $(VENV)/bin/activate && black --check blindaid tests

format:
	@. $(VENV)/bin/activate && black blindaid tests

mypy:
	@. $(VENV)/bin/activate && mypy blindaid

pytest:
	@. $(VENV)/bin/activate && pytest

test: pytest

run:
	@. $(VENV)/bin/activate && python -m blindaid

clean:
	@rm -rf $(VENV)
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
