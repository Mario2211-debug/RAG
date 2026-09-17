### Retrieval-Augmented Generation system
## Build a Retrieval-Augmented Generation system that answers questions about codebases by retrieving relevant information and generating evidence-based responses, implementing intelligent chunking, efficient retrieval (TF-IDF/BM25)


You will build a Retrieval-Augmented Generation system that answers questions
about a codebase. You ingest the provided vLLM repository into a searchable index, retrieve the most relevant snippets for a question, generate an answer from them with Qwen/Qwen3-0.6B, and measure retrieval quality with recall@k. Your system is judged on whether it retrieves the right source locations and produces answers grounded in them

# ingest the provided vLLM repository into a searchable index
# retrieve the most relevant snippets for a question
# generate an answer from them with
Qwen/Qwen3-0.6B
# measure retrieval quality with recall@k
# Your system is judged on
whether it retrieves the right source locations and produces answers grounded in them


###  Indexing the codebase

Everything starts with the index. Read the files you judge useful from the vLLM repository shipped in the attachments, split each one into chunks, and persist an index that
retrieval can query in milliseconds.

Indexing the whole corpus must take at most 5 minutes.

A Python file and a Markdown page do not break apart the same way, so your program must implement two distinct chunking strategies:
• Python code chunking,
• Markdown / text chunking.

For retrieval itself, implement at least one of the two classic lexical methods. The choice hnmis yours:
• TF-IDF,
• BM25.

You may explore other methods on top, as long as one of these two is implemented.

## Retrieval

With the index built, you can search it. Given a question, your system returns the top-k most relevant snippets. Each result is a source location: a file_path and the
character range (first_character_index, last_character_index) it covers, at most 2000 characters wide. file_path must match the corpus path exactly (e.g. data/raw/vllm-0.10.1/docs/features/lora.md). The grader compares paths verbatim, so a path with a different prefix never matches.

Retrieval must work for a single query and in batch over a whole dataset of questions read
from JSON. On the reference datasets, your system must reach at least 80 % recall@5 on
docs questions and 50 % on code questions (the metric is defined in the Evaluation
chapter).