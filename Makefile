UV      := uv
PYTHON  := $(UV) run python

# tudo sobreponivel: make run DATASET=... K=...
CORPUS          ?= data/raw
INDEX           ?= data/processed/index.json
MAX_CHUNK_SIZE  ?= 2000
K               ?= 10
SCOPE           ?= UnansweredQuestions
DATASET         ?= data/datasets/$(SCOPE)/dataset_docs_public.json
GROUND_TRUTH    ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
SEARCH_DIR      ?= data/output/search_results/$(SCOPE)
ANSWER_DIR      ?= data/output/search_results_and_answer/$(SCOPE)
RESULTS         ?= $(SEARCH_DIR)/$(notdir $(DATASET))
QUERY           ?= How do I load a LoRA adapter in vLLM?

.PHONY: all install run debug index search search_dataset answer \
        answer_dataset evaluate lint lint-strict clean fclean re

all: run

install:
	$(UV) sync

# pipeline de retrieval de ponta a ponta: indexa e pesquisa o dataset
run: index search_dataset evaluate

index:
	$(PYTHON) -m src index \
		--max_chunk_size $(MAX_CHUNK_SIZE) \
		--corpus_root $(CORPUS) \
		--index_path $(INDEX)

search:
	$(PYTHON) -m src search "$(QUERY)" --k $(K) --index_path $(INDEX)

search_dataset:
	$(PYTHON) -m src search_dataset \
		--dataset_path $(DATASET) \
		--k $(K) \
		--save_directory $(SEARCH_DIR) \
		--index_path $(INDEX)

answer:
	$(PYTHON) -m src answer "$(QUERY)" --k $(K) --index_path $(INDEX)

answer_dataset:
	$(PYTHON) -m src answer_dataset \
		--student_search_results_path $(RESULTS) \
		--save_directory $(ANSWER_DIR)

evaluate:
	$(PYTHON) -m src evaluate \
		--student_search_results_path $(RESULTS) \
		--dataset_path $(GROUND_TRUTH)

debug:
	$(PYTHON) -m pdb -m src search "$(QUERY)" --k $(K) --index_path $(INDEX)

lint:
	$(UV) run flake8 . --exclude=.venv,.mypy_cache,data
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV) run flake8 . --exclude=.venv,.mypy_cache,data
	$(UV) run mypy . --strict

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -path ./.venv -prune -o -type d -name __pycache__ -print0 \
		| xargs -0 rm -rf

fclean: clean
	rm -rf data/processed data/output

re: fclean run
