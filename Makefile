PYTHON ?= python

.PHONY: lint test

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest -q
