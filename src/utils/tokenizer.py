"""Tokenizacao lexical usada dos dois lados da pesquisa."""

from src.config import PART_RE, WORD_RE


def tokenizer(text: str) -> list[str]:
    """Corta texto em tokens minusculos.

    Guarda o identificador inteiro e tambem as suas partes: uma pergunta
    pode citar `get_num_layers` tal e qual ou falar de "num layers", e os
    dois lados tem de chegar ao mesmo chunk.

    A mesma funcao corre sobre os chunks (na indexacao) e sobre a
    pergunta (na pesquisa): os dois lados tem de falar o mesmo dialeto.
    """
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        word = match.group(0)
        tokens.append(word.lower())
        parts = PART_RE.findall(word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens
