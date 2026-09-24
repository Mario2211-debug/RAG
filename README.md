*This project has been created as part of the 42 curriculum by mafonso.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about a
codebase — here, the vLLM 0.10.1 repository.

The pipeline has four stages:

1. **Indexing** — every `.py`, `.md` and `.txt` file under `data/raw/` is split
   into chunks and folded into an inverted index persisted in
   `data/processed/`.
2. **Retrieval** — a question is scored against that index with BM25; the
   answer is a ranked list of *source locations* (`file_path` plus a character
   range no wider than 2000 characters).
3. **Augmentation** — the top sources are read back from disk and packed into a
   bounded context.
4. **Generation** — `Qwen/Qwen3-0.6B` reads that context and writes an answer
   grounded in it.

Retrieval quality is measured with recall@k against the reference datasets.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install          # uv sync
```

Expected layout (the corpus and the datasets are provided as attachments, not
committed to this repository):

```
data/raw/vllm-0.10.1/                       the indexed corpus
data/processed/                             the index produced by `index`
data/datasets/UnansweredQuestions/          questions to answer
data/datasets/AnsweredQuestions/            the same questions with ground truth
data/output/search_results/<Scope>/         output of `search_dataset`
data/output/search_results_and_answer/<Scope>/   output of `answer_dataset`
```

Every command is `uv run python -m src <command> [options]`:

| Command | What it does |
| --- | --- |
| `index --max_chunk_size <int>` | Ingest `data/raw/` and build the index |
| `search <query> --k <int>` | Top-k sources for a single question |
| `search_dataset --dataset_path <p> --k <int> --save_directory <d>` | Batch search, writes a `StudentSearchResults` JSON |
| `answer <query> --k <int>` | Retrieve, then generate an answer |
| `answer_dataset --student_search_results_path <p> --save_directory <d>` | Batch generation, writes a `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path <p> --dataset_path <p>` | Own recall@k against a ground-truth dataset |

All input and output paths are CLI arguments; nothing is hard-coded. The
Makefile wraps the same commands (`make index`, `make search_dataset`,
`make evaluate`, `make answer_dataset`), and `make run` chains index → search →
evaluate. `make lint` runs flake8 and mypy, `make lint-strict` runs
`mypy --strict`; both are clean.

## Example usage

```bash
# 1. build the index (about 7 s for the whole vLLM corpus)
uv run python -m src index --max_chunk_size 2000

# 2. one question
uv run python -m src search "How do I load a LoRA adapter?" --k 5
 1.   38.41  data/raw/vllm-0.10.1/docs/features/lora.md  [4695:6100]
 2.   31.02  data/raw/vllm-0.10.1/docs/features/lora.md  [6100:8080]
 ...

# 3. a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 4. score it
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Recall@1: 0.620  Recall@3: 0.760  Recall@5: 0.840  Recall@10: 0.910

# 5. generate answers from those results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# or a single question, end to end
uv run python -m src answer "How do I dynamically load a LoRA adapter?" --k 5
Answer: To dynamically load a LoRA adapter in vLLM, send a POST request to the
`/v1/load_lora_adapter` endpoint with the name and path of the adapter. [...]
```

## System architecture

```
data/raw/  ──▶ DataProcess ──▶ chunking ──▶ Index ──▶ data/processed/index.json
                                                            │
question ───────────────────────────────▶ Retriever ◀───────┘
                                              │
                                              ├─▶ MinimalSource list ─▶ JSON output
                                              │
                                              └─▶ Generator (Qwen3-0.6B) ─▶ answer
```

| Module | Responsibility |
| --- | --- |
| `src/__main__.py` | Fire entry point; turns expected failures into messages |
| `src/cli.py` | One method per subcommand: argument checking, I/O, reporting |
| `src/config.py` | Every default value and regex in one place |
| `src/models.py` | The pydantic models exchanged between stages |
| `src/data_process.py` | Walks the corpus and reads the text files |
| `src/chunking.py` | The two chunking strategies |
| `src/utils/tokenizer.py` | The single tokenizer, shared by indexing and search |
| `src/index.py` | Builds, saves and loads the inverted index |
| `src/retrieval.py` | BM25 scoring, file prior, ranking |
| `src/generation.py` | Context assembly and generation with Qwen3-0.6B |
| `src/evaluation.py` | IoU overlap and recall@k |

The index is a plain JSON document with four parts: `postings`
(token → list of `[chunk_id, frequency]`), `chunks` (chunk_id → `[path, start,
end]`), `doc_len` (chunk_id → token count) and the corpus-wide `avgdl`. That is
everything BM25 needs and nothing else.

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so
`chunk_document` dispatches on the extension — the only place in the code that
knows about file types.

**Python** (`chunk_python_file`): a sliding window of `max_chunk_size`
characters that backs up to the nearest blank line, or failing that to the
nearest newline, so a chunk does not end in the middle of a statement. Chunks
overlap by 200 characters so that a definition straddling a boundary still
appears whole in one of them.

**Markdown / text** (`chunk_markdown_file`): headings are the natural seams, so
the text is first cut at every `#`…`######` line. A section longer than
`max_chunk_size` is then split again at paragraph breaks.

Both chunkers share one guard: a candidate cut point is only accepted if it
falls past the middle of the chunk. Without it, a file with an early newline
and no later one makes the window advance a few characters at a time and the
index explodes into thousands of useless slivers. That bug cost an afternoon.

At indexing time each chunk is tokenized together with two extra pieces of
evidence:

* **its file path**, counted twice — a question about LoRA should be able to
  find `docs/features/lora.md` by name alone;
* **the two headings above it**, for text files — a chunk taken from the middle
  of a section no longer contains its own title, and the title is usually where
  the words of the question live.

## Retrieval method

Ranking is **BM25** (`k1 = 1.5`, `b = 0.75`) over the inverted index. For every
distinct query token, the posting list gives the chunks containing it; each
contributes `idf · tf · (k1+1) / (tf + k1 · (1 − b + b · len/avgdl))`. Only
chunks that share a token with the question are ever touched, which is why a
search costs milliseconds on a 14 000-chunk index.

Two additions on top of textbook BM25:

**Sub-token indexing.** The tokenizer keeps both the whole identifier and its
parts: `get_num_layers` is indexed as `get_num_layers`, `get`, `num` and
`layers`. Questions either quote an identifier verbatim or paraphrase it, and
this way both reach the same chunk. On code questions this alone moved
recall@5 from 0.717 to 0.828.

**File prior.** After scoring, the scores of all chunks of the same file are
summed, and each chunk is boosted in proportion to how well its file scored
overall (`score += 0.3 · score · file_share`). A documentation question is
usually about a topic, and a topic lives in a whole file: if several chunks of
the same file match, the right region is probably there too. This moved docs
recall@5 from 0.800 to 0.840.

The same tokenizer runs on the chunk at indexing time and on the question at
search time — the two sides must speak the same dialect, which is why it lives
in a single function.

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions),
`max_chunk_size = 2000`:

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Threshold |
| --- | --- | --- | --- | --- | --- |
| docs | 0.620 | 0.760 | **0.840** | 0.910 | 0.80 |
| code | 0.636 | 0.778 | **0.859** | 0.919 | 0.50 |

Both figures were confirmed by the reference `moulinette`
(`Student data is valid: True`).

Effect of chunk size on recall@5:

| `max_chunk_size` | chunks | index time | search (100 questions) | docs @5 | code @5 |
| --- | --- | --- | --- | --- | --- |
| 500 | 84 936 | 16.3 s | 16.2 s | 0.800 | 0.848 |
| 1000 | 31 200 | 8.9 s | 6.1 s | 0.820 | 0.848 |
| 2000 | 14 304 | 7.3 s | 2.8 s | 0.840 | 0.859 |

Smaller chunks pin the answer down more precisely but split the evidence: a
question whose terms are spread over a section no longer finds any single chunk
that carries all of them, and recall drops. 2000 is both the best and the
maximum the grader accepts, so it is the default.

Against the required budgets:

| Requirement | Limit | Measured |
| --- | --- | --- |
| Indexing the whole corpus | 300 s | **7.3 s** (1952 files, 14 304 chunks) |
| Retrieval, 200 questions | 90 s | **~5.6 s** |
| Recall@5 docs | 0.80 | **0.840** |
| Recall@5 code | 0.50 | **0.859** |

Generation is the slow stage: roughly 10 s per question on CPU, so a
100-question dataset takes about 17 minutes. It is bounded on purpose — at most
3 sources, 6000 characters of context and 256 new tokens.

## Design decisions

* **JSON for the index, not pickle.** It is readable, debuggable by hand and
  safe to load. The price is converting integer keys back from strings when
  reloading, which `Index.load_index` does explicitly.
* **The index is built in a single pass.** Nothing is searched while indexing:
  every chunk is tokenized once, counted, and appended to its posting lists.
* **Pydantic only at the boundaries.** Every file read or written is validated
  against a model; the internal services (indexer, retriever, generator) are
  plain classes, as the subject allows.
* **Lazy import of torch.** `transformers` and `torch` are imported inside
  `Generator.load`, so `index`, `search` and `evaluate` do not pay the several
  seconds the deep-learning stack costs to start.
* **Bounded context.** A 0.6B model drowns in context as easily as it starves
  without it, so only the best 3 sources are passed and the whole prompt is
  capped in characters.
* **Own `evaluate` command.** It reimplements recall@k (IoU ≥ 0.05, same file)
  to iterate quickly; the official score still comes from the moulinette, which
  this code never imports or calls. The two agree to the third decimal.
* **`--max_chunk_size` is a real parameter**, stored in the index, not a
  constant buried in the chunkers.

## Challenges faced

* **Degenerate chunking.** The first chunkers could advance one character at a
  time when the only available cut point sat right after the start of the
  window. The fix — refusing cut points before the midpoint of a chunk — is two
  lines and the reason the index has 14 000 chunks instead of 200 000.
* **Exact paths.** The grader compares `file_path` verbatim. Storing paths
  exactly as they are read from `data/raw/...`, instead of rebuilding them from
  a prefix, removed a whole class of silent zero-recall failures.
* **Docs recall stuck at 0.80.** Plain BM25 sat exactly on the threshold, with
  no margin at all. Rather than tuning `k1` and `b` — a grid search over them
  changed nothing useful — the gain came from giving the ranker better
  evidence: sub-tokens, path tokens and heading context, then the file prior.
* **Questions and code speak different languages.** A question says "number of
  layers", the code says `num_layers`. Indexing both the identifier and its
  parts is what closes that gap.
* **Never crashing.** The CLI is tested with empty queries, `k = 0`, missing
  files, malformed JSON and files that parse but do not match the schema. Every
  one of them produces a one-line message and a non-zero exit code, never a
  traceback.

## Resources

* S. Robertson & H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* — the paper the ranking function comes from.
* C. Manning, P. Raghavan & H. Schütze, *Introduction to Information
  Retrieval*, chapters 1–6 (inverted index, tf-idf, BM25).
* [vLLM documentation](https://docs.vllm.ai/) — the corpus itself.
* [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) and the
  [Transformers documentation](https://huggingface.co/docs/transformers/).
* [Python Fire](https://github.com/google/python-fire),
  [pydantic](https://docs.pydantic.dev/) and [uv](https://docs.astral.sh/uv/)
  documentation.

### Use of AI

AI was used as an assistant:

* **AI used as a measuring instrument:** the ranking experiments (sub-token
  indexing, path boost, heading context, file prior, chunk-size sweep) were run
  as scripted comparisons; the numbers in the table above come from those runs
  and were all reproduced through the CLI and the moulinette.
