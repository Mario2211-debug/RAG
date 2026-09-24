"""Command-line interface: one method per Fire subcommand."""

import os
import json
import time
from tqdm import tqdm
from typing import Any
from src.index import Index
from src.retrieval import Retriever
from pydantic import ValidationError
from src.generation import Generator
from src.evaluation import recall_at_k
from src.data_process import DataProcess
from src.config import (CORPUS_ROOT, DEFAULT_K, INDEX_PATH, MAX_CHUNK_SIZE,
                        MAX_NEW_TOKENS, MODEL_NAME)
from src.models import (MinimalAnswer, MinimalSearchResults, RagDataset,
                        StudentSearchResults, StudentSearchResultsAndAnswer,
                        UnansweredQuestion)

REPORT_KS = (1, 3, 5, 10)


class CliError(Exception):
    """Expected CLI error: the CLI prints this message
    instead of a traceback."""


def as_int(value: Any, name: str) -> int:
    """Convert a CLI argument to an integer, or explain why it is invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CliError(f"--{name} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError:
        raise CliError(f"--{name} must be an integer, got {value!r}")


def read_json(path: str) -> Any:
    """Read a JSON file from disk, translating expected failures."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise CliError(f"file not found: {path}")
    except json.JSONDecodeError as err:
        raise CliError(f"malformed JSON in {path}: {err}")
    except OSError as err:
        raise CliError(f"cannot read {path}: {err}")


def write_json(payload: Any, save_directory: str, filename: str) -> str:
    """Write a pydantic model to a file inside save_directory."""
    try:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload.model_dump_json(indent=2))
    except OSError as err:
        raise CliError(f"cannot write into {save_directory}: {err}")
    return path


def load_dataset(path: str) -> RagDataset:
    """Validate a question dataset against the pydantic model."""
    try:
        return RagDataset.model_validate(read_json(path))
    except ValidationError as err:
        raise CliError(f"{path} is not a valid RagDataset:\n{err}")


def load_search_results(path: str) -> StudentSearchResults:
    """Validate a search-results file."""
    try:
        return StudentSearchResults.model_validate(read_json(path))
    except ValidationError as err:
        raise CliError(f"{path} is not a valid StudentSearchResults:\n{err}")


def load_retriever(index_path: str) -> Retriever:
    """Load the index from disk and return a ready-to-use retriever."""
    try:
        return Retriever(Index.load_index(index_path))
    except FileNotFoundError:
        raise CliError(f"index not found: {index_path}\n"
                       "build it first: uv run python -m src index")
    except (json.JSONDecodeError, KeyError) as err:
        raise CliError(f"corrupt index in {index_path}: {err}\n"
                       "rebuild it: uv run python -m src index")


class Cli():
    """Commands available via `uv run python -m src <command>`."""

    def index(self,
              max_chunk_size: int = MAX_CHUNK_SIZE,
              corpus_root: str = CORPUS_ROOT,
              index_path: str = INDEX_PATH) -> None:
        """Ingest the corpus and save the index.

        Args:
            max_chunk_size: maximum chunk size in characters.
            corpus_root: root of the files to index.
            index_path: file where the index is saved.
        """
        max_chunk_size = as_int(max_chunk_size, "max_chunk_size")
        if max_chunk_size <= 0:
            raise CliError("--max_chunk_size must be a positive integer")
        if not os.path.isdir(corpus_root):
            raise CliError(f"corpus not found: {corpus_root}")

        start = time.time()
        documents = DataProcess().load_docs(corpus_root)
        if not documents:
            raise CliError(f"no indexable file under {corpus_root}")
        print(f"files read .....: {len(documents):,}")

        indexer = Index(documents)
        index = indexer.build_index(max_chunk_size)
        indexer.save_index(index, index_path)

        size = os.path.getsize(index_path) / 1e6
        print(f"chunks .........: {index['n_chunks']:,}")
        print(f"vocabulary .....: {len(index['postings']):,}")
        print(f"avgdl ..........: {index['avgdl']:.0f} tokens")
        print(f"elapsed ........: {time.time() - start:.1f} s (limit 300 s)")
        print(f"Ingestion complete! Indices saved under {index_path} "
              f"({size:.1f} MB)")

    def search(self,
               query: str,
               k: int = DEFAULT_K,
               index_path: str = INDEX_PATH,
               save_directory: str = "") -> None:
        """Show the k best sources for a question.

        Args:
            query: the question.
            k: number of sources to return.
            index_path: index to query.
            save_directory: if given, also writes a StudentSearchResults file.
        """
        k = as_int(k, "k")
        retriever = load_retriever(index_path)
        start = time.time()
        results = retriever.search(str(query), k)
        elapsed = (time.time() - start) * 1000

        if not results:
            print("no result (empty query, k <= 0 or no token matched)")
        for rank, (source, score) in enumerate(results, 1):
            print(f"{rank:>2}. {score:7.2f}  {source.file_path}  "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")
        print(f"({elapsed:.0f} ms)")

        if save_directory:
            question = UnansweredQuestion(question=str(query))
            payload = StudentSearchResults(
                search_results=[MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[s for s, _ in results])],
                k=k)
            path = write_json(payload, save_directory, "search_result.json")
            print(f"Saved student_search_results to {path}")

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = DEFAULT_K,
                       index_path: str = INDEX_PATH) -> None:
        """Search an entire dataset and save a StudentSearchResults file.

        Args:
            dataset_path: JSON with the questions.
            save_directory: output folder, with the dataset name.
            k: number of sources per question.
            index_path: index to query.
        """
        k = as_int(k, "k")
        dataset = load_dataset(dataset_path)
        retriever = load_retriever(index_path)
        print(f"Loaded {len(dataset.rag_questions)} questions")

        start = time.time()
        search_results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retriever.sources(question.question, k))
            for question in tqdm(dataset.rag_questions, desc="searching",
                                 unit="question")]
        elapsed = time.time() - start

        payload = StudentSearchResults(search_results=search_results, k=k)
        path = write_json(payload, save_directory,
                          os.path.basename(dataset_path))
        print(f"Retrieved {len(search_results)} questions in {elapsed:.1f} s "
              "(limit 90 s for 200 questions)")
        print(f"Saved student_search_results to {path}")

    def answer(self,
               query: str,
               k: int = DEFAULT_K,
               index_path: str = INDEX_PATH,
               model_name: str = MODEL_NAME,
               max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        """Answer a question using the retrieved sources.

        Args:
            query: the question.
            k: number of sources to retrieve before generating.
            index_path: index to query.
            model_name: model used for generation.
            max_new_tokens: maximum generated tokens.
        """
        k = as_int(k, "k")
        retriever = load_retriever(index_path)
        sources = retriever.sources(str(query), k)
        for source in sources:
            print(f"   {source.file_path}  "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")
        generator = Generator(model_name,
                              as_int(max_new_tokens, "max_new_tokens"))
        print("\nAnswer:")
        print(generator.answer(str(query), sources))

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str,
                       model_name: str = MODEL_NAME,
                       max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        """Generate answers for already existing search results.

        Args:
            student_search_results_path: output of search_dataset.
            save_directory: folder where the answer file is written.
            model_name: model used for generation.
            max_new_tokens: maximum generated tokens.
        """
        max_new_tokens = as_int(max_new_tokens, "max_new_tokens")
        results = load_search_results(student_search_results_path)
        total = len(results.search_results)
        print(f"Loaded {total} questions")

        generator = Generator(model_name, max_new_tokens)
        answers: list[MinimalAnswer] = []
        for result in tqdm(results.search_results, desc="answering",
                           unit="question"):
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generator.answer(result.question,
                                        result.retrieved_sources)))

        payload = StudentSearchResultsAndAnswer(search_results=answers,
                                                k=results.k)
        path = write_json(payload, save_directory,
                          os.path.basename(student_search_results_path))
        print(f"Processed {len(answers)} of {total} questions")
        print(f"Saved student_search_results_and_answer to {path}")

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str,
                 k: int = 0) -> None:
        """Recall@k of my results against the ground truth.

        Args:
            student_search_results_path: output of search_dataset.
            dataset_path: AnsweredQuestions reference dataset.
            k: if 0, reports recall@1, @3, @5, and @10.
        """
        k = as_int(k, "k")
        results = load_search_results(student_search_results_path)
        dataset = load_dataset(dataset_path)
        ks = (k,) if k > 0 else REPORT_KS

        line = []
        for value in ks:
            score, found, total = recall_at_k(results, dataset, value)
            if not total:
                raise CliError(f"{dataset_path} has no ground-truth source; "
                               "evaluate against an AnsweredQuestions file")
            line.append(f"Recall@{value}: {score:.3f} ({found}/{total})")
        print("Evaluation Results")
        print("=" * 50)
        print("  ".join(line))
