from models import _TOKEN_RE


def tokenizer(text: str) -> list[str]:
    """Corta texto em tokens minusculos.

    A mesma funcao corre sobre os chunks (na indexacao) e sobre a
    pergunta (na pesquisa): os dois lados tem de falar o mesmo dialeto.
    """
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
