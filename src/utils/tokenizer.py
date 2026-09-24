"""Lexical tokenization used on both sides of the search."""

from src.config import PART_RE, WORD_RE


def tokenizer(text: str) -> list[str]:
    """Split text into lowercase tokens.

    Keep the whole identifier and also its parts: a question may quote
    `get_num_layers` exactly or mention "num layers", and both sides must
    reach the same chunk.

    The same function runs on chunks during indexing and on the question during
    retrieval: both sides must speak the same dialect.
    """
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        word = match.group(0)
        tokens.append(word.lower())
        parts = PART_RE.findall(word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens
