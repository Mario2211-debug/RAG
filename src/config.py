"""Shared constants used across the pipeline stages.

Nothing here is computed: these are only the default values that the CLI may
override with flags.
"""

import re

# BM25
K1 = 1.5
B = 0.75
# file prior weight: how strongly the whole-file score pulls each chunk up
FILE_PRIOR = 0.3
# how many times the path tokens are added to the chunk
PATH_BOOST = 2
# how many headings above the chunk are included in the token context
HEADING_CONTEXT = 2

# chunking
OVERLAP = 200
MAX_CHUNK_SIZE = 2000

# retrieval / evaluation
DEFAULT_K = 10
MIN_IOU = 0.05

# default paths (all overrideable in the CLI)
CORPUS_ROOT = "data/raw/vllm-0.10.1"
INDEX_PATH = "data/processed/index.json"

# corpus reading
TEXT_EXT = (".py", ".md", ".txt")
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}

# generation
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 256
MAX_CONTEXT_SOURCES = 3
MAX_CONTEXT_CHARS = 6000

# markdown section boundary and heading text
HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
HEADING_TEXT_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
# a raw "word": identifier, number, or ordinary word
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# identifier parts: camelCase, snake_case, and digits
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")
