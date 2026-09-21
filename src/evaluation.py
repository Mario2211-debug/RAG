"""Recall@k do aluno, para iterar sem depender da moulinette."""

from src.config import MIN_IOU
from src.models import (AnsweredQuestion, MinimalSearchResults, MinimalSource,
                        RagDataset, StudentSearchResults)


def iou(first: MinimalSource, second: MinimalSource) -> float:
    """Intersection over union de dois intervalos de caracteres."""
    a_start = first.first_character_index
    a_end = first.last_character_index
    b_start = second.first_character_index
    b_end = second.last_character_index
    inter = min(a_end, b_end) - max(a_start, b_start)
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union else 0.0


def is_found(expected: MinimalSource,
             retrieved: list[MinimalSource],
             k: int) -> bool:
    """Uma source conta se o top-k tocar no mesmo ficheiro e regiao."""
    for source in retrieved[:k]:
        if source.file_path != expected.file_path:
            continue
        if iou(source, expected) >= MIN_IOU:
            return True
    return False


def recall_at_k(results: StudentSearchResults,
                dataset: RagDataset,
                k: int) -> tuple[float, int, int]:
    """Fatia das sources de referencia que aparecem no top-k."""
    retrieved_by_id: dict[str, MinimalSearchResults] = {
        result.question_id: result for result in results.search_results}

    found = 0
    total = 0
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue
        result = retrieved_by_id.get(question.question_id)
        retrieved = result.retrieved_sources if result else []
        for expected in question.sources:
            total += 1
            if is_found(expected, retrieved, k):
                found += 1
    return (found / total if total else 0.0), found, total
