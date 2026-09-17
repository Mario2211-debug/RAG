UV		:= uv
PYTHON	:= $(UV) run python 
SOURCES	:= src test conftest.py


FUNCTIONS ?= data/input/functions_definition.json
INPUT	?= data/input/function_calling_tests.json
OUTPUT 	?= data/output/functions_calling_results.json


.PHONY: all install run debug lint lint-strict clean fclean re

all: run

install:
	$(UV) sync

run:
	$(PYTHON) -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	$(PYTHON) -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	$(UV) run flake8 . --exclude=.venv,__pycache__,.git,.mypy_cache,.pytest_cache,llm_sdk
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
			--ignore-missing-imports --disallow-untyped-defs \
			--check-untyped-defs --exclude=llm_sdk

lint-strict:
	$(UV) run flake8 . --exclude=.venv,__pycache__,.git,.mypy_cache,.pytest_cache,llm_sdk
	$(UV) run mypy . --strict --exclude=llm_sdk

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

fclean: clean
	rm -rf data/output

re: fclean run