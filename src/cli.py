"""Interface de linha de comandos: um metodo por subcomando Fire."""

import json
import os
import time
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from src.config import (CORPUS_ROOT, DEFAULT_K, INDEX_PATH, MAX_CHUNK_SIZE,
                        MAX_NEW_TOKENS, MODEL_NAME)
from src.data_process import DataProcess
from src.evaluation import recall_at_k
from src.generation import Generator
from src.index import Index
from src.models import (MinimalAnswer, MinimalSearchResults, RagDataset,
                        StudentSearchResults, StudentSearchResultsAndAnswer,
                        UnansweredQuestion)
from src.retrieval import Retriever

REPORT_KS = (1, 3, 5, 10)


class CliError(Exception):
    """Erro previsto: a CLI mostra a mensagem em vez de um traceback."""


def as_int(value: Any, name: str) -> int:
    """Converte um argumento da CLI em inteiro, ou explica-se."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CliError(f"--{name} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError:
        raise CliError(f"--{name} must be an integer, got {value!r}")


def read_json(path: str) -> Any:
    """Le um JSON do disco, traduzindo as falhas previsiveis."""
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
    """Escreve um modelo pydantic num ficheiro dentro de save_directory."""
    try:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload.model_dump_json(indent=2))
    except OSError as err:
        raise CliError(f"cannot write into {save_directory}: {err}")
    return path


def load_dataset(path: str) -> RagDataset:
    """Valida um dataset de perguntas contra o modelo pydantic."""
    print("Here")
    try:
        return RagDataset.model_validate(read_json(path))
    except ValidationError as err:
        raise CliError(f"{path} is not a valid RagDataset:\n{err}")


def load_search_results(path: str) -> StudentSearchResults:
    """Valida um ficheiro de resultados de pesquisa."""
    try:
        return StudentSearchResults.model_validate(read_json(path))
    except ValidationError as err:
        raise CliError(f"{path} is not a valid StudentSearchResults:\n{err}")


def load_retriever(index_path: str) -> Retriever:
    """Carrega o indice do disco e devolve um retriever pronto a usar."""
    print("Here")
    try:
        return Retriever(Index.load_index(index_path))
    except FileNotFoundError:
        raise CliError(f"index not found: {index_path}\n"
                       "build it first: uv run python -m src index")
    except (json.JSONDecodeError, KeyError) as err:
        raise CliError(f"corrupt index in {index_path}: {err}\n"
                       "rebuild it: uv run python -m src index")


class Cli():
    """Comandos disponiveis em `uv run python -m src <command>`."""

    def index(self,
              max_chunk_size: int = MAX_CHUNK_SIZE,
              corpus_root: str = CORPUS_ROOT,
              index_path: str = INDEX_PATH) -> None:
        """Ingere o corpus e grava o indice.

        Args:
            max_chunk_size: tamanho maximo de um chunk, em caracteres.
            corpus_root: raiz dos ficheiros a indexar.
            index_path: ficheiro onde o indice e gravado.
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
        """Mostra as k melhores sources para uma pergunta.

        Args:
            query: a pergunta.
            k: numero de sources a devolver.
            index_path: indice a consultar.
            save_directory: se dado, grava tambem um StudentSearchResults.
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
        """Pesquisa um dataset inteiro e grava um StudentSearchResults.

        Args:
            dataset_path: JSON com as perguntas.
            save_directory: pasta de saida, com o nome do dataset.
            k: numero de sources por pergunta.
            index_path: indice a consultar.
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
        """Responde a uma pergunta a partir das sources recuperadas.

        Args:
            query: a pergunta.
            k: numero de sources a recuperar antes de gerar.
            index_path: indice a consultar.
            model_name: modelo usado na geracao.
            max_new_tokens: limite de tokens gerados.
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
        """Gera respostas para resultados de pesquisa ja existentes.

        Args:
            student_search_results_path: saida de search_dataset.
            save_directory: pasta onde grava o ficheiro com respostas.
            model_name: modelo usado na geracao.
            max_new_tokens: limite de tokens gerados.
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
        """Recall@k dos meus resultados contra o ground truth.

        Args:
            student_search_results_path: saida de search_dataset.
            dataset_path: dataset AnsweredQuestions de referencia.
            k: se 0, reporta recall@1, @3, @5 e @10.
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
        print("=" * 40)
        print("  ".join(line))
