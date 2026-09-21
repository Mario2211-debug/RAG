import re
import uuid
from typing import List
from pydantic import BaseModel, Field


K1 = 1.5
B = 0.75
OVERLAP = 200
MIN_IOU = 0.05
MAX_CHUNK_SIZE = 2000
PATH_PREFIX = "data/raw/"
CORPUS_ROOT = "vllm-0.10.1"
TEXT_EXT = (".py", ".md", ".txt")
INDEX_PATH = "data/processed/index.json"
HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}
_TOKEN_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z]+")


class Augment(BaseModel):
    pass


class Generate(BaseModel):
    pass


class Indexing(BaseModel):
    pass


class Evaluate(BaseModel):
    pass


class DataProcess(BaseModel):
    pass


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
