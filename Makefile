.PHONY: install dev lint format test demo

install:
	python -m pip install -U pip
	python -m pip install -e .

dev:
	python -m pip install -U pip
	python -m pip install -e .[dev]

lint:
	ruff check .
	python -m mypy src/mmds

format:
	black .
	ruff check . --fix

test:
	pytest

demo:
	python scripts/run_demo.py
