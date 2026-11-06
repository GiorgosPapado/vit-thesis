PYTHON ?= python

.PHONY: lint test test-all

lint:
	$(PYTHON) -m ruff check src tests scripts

# -rs prints a summary of skipped tests (e.g. torch tests skipped when torch is missing).
test:
	$(PYTHON) -m pytest -q -rs

# Fails (collection error in tests/conftest.py) if torch is missing; also fails on any skipped torch test.
test-all:
	REQUIRE_TORCH=1 $(PYTHON) -m pytest -q -rs -p no:cacheprovider
