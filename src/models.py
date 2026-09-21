 """Modelos pydantic trocados entre as fases do pipeline.

Sao a fronteira do sistema: tudo o que entra de JSON e validado aqui,
tudo o que sai e serializado a partir daqui.
"""

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Uma localizacao no corpus: ficheiro + intervalo de caracteres."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Pergunta sem resposta de referencia."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Pergunta com resposta e sources de referencia (ground truth)."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset de perguntas lido de JSON."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Resultado de pesquisa para uma pergunta."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Resultado de pesquisa com a resposta gerada."""

    answer: str


class StudentSearchResults(BaseModel):
    """Ficheiro de saida do comando search_dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Ficheiro de saida do comando answer_dataset."""

    search_results: List[MinimalAnswer]
    k: int
