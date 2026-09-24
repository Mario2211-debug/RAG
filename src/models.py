"""Pydantic models exchanged across pipeline stages.

These are the boundary of the system: everything entering from JSON is
validated here, and everything leaving is serialized from here.
"""

import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A location in the corpus: file + character-range interval."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without a reference answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with an answer and ground-truth sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset of questions read from JSON."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search result for a question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output file of the search_dataset command."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output file of the answer_dataset command."""

    search_results: List[MinimalAnswer]
    k: int
