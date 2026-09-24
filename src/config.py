"""Constantes partilhadas pelas varias fases do pipeline.

Nada aqui e calculado: sao so os valores por omissao que a CLI pode
sobrepor com flags.
"""

import re

# BM25
K1 = 1.5
B = 0.75
# peso do "file prior": quanto o score do ficheiro inteiro puxa cada chunk
FILE_PRIOR = 0.3
# quantas vezes os tokens do path entram no chunk
PATH_BOOST = 2
# quantos titulos acima do chunk entram nos seus tokens
HEADING_CONTEXT = 2

# chunking
OVERLAP = 200
MAX_CHUNK_SIZE = 2000

# retrieval / avaliacao
DEFAULT_K = 10
MIN_IOU = 0.05

# caminhos por omissao (todos sobreponiveis na CLI)
CORPUS_ROOT = "data/raw/vllm-0.10.1"
INDEX_PATH = "data/processed/index.json"

# leitura do corpus
TEXT_EXT = (".py", ".md", ".txt")
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}

# geracao
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 256
MAX_CONTEXT_SOURCES = 3
MAX_CONTEXT_CHARS = 6000

# fronteira de seccao markdown, e o texto do titulo
HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
HEADING_TEXT_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
# uma "palavra" bruta: identificador, numero ou palavra normal
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# as partes de um identificador: camelCase, snake_case e digitos
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")
