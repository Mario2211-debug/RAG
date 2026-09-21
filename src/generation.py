"""Geracao da resposta com Qwen3-0.6B a partir das sources recuperadas."""

from typing import Any

from src.config import (MAX_CONTEXT_CHARS, MAX_CONTEXT_SOURCES,
                        MAX_NEW_TOKENS, MODEL_NAME)
from src.models import MinimalSource

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the vLLM "
    "codebase. Answer only with what the provided context supports. "
    "If the context does not contain the answer, say so plainly. "
    "Answer in at most four sentences, with no preamble."
)

USER_PROMPT = (
    "Context extracted from the codebase:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer using only the context above."
)


class Generator():
    """Envolve o modelo local: carrega-o uma vez e responde com contexto."""

    def __init__(self, model_name: str = MODEL_NAME,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Carrega o modelo a pedido.

        O import vive aqui de proposito: quem so indexa ou pesquisa nao
        paga os segundos que o torch demora a arrancar.
        """
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype="float32")
        self.model.eval()

    @staticmethod
    def read_source(source: MinimalSource) -> str:
        """Le do disco o troco de ficheiro que a source aponta."""
        try:
            with open(source.file_path, "r", encoding="utf-8",
                      errors="ignore") as handle:
                text = handle.read()
        except OSError:
            return ""
        return text[source.first_character_index:
                    source.last_character_index]

    def build_context(self, sources: list[MinimalSource]) -> str:
        """Junta as melhores sources num contexto com orcamento fixo.

        Truncar aqui e deliberado: o modelo tem 0.6B de parametros e
        afoga-se em contexto tanto como com contexto a menos.
        """
        blocks: list[str] = []
        budget = MAX_CONTEXT_CHARS
        for source in sources[:MAX_CONTEXT_SOURCES]:
            snippet = self.read_source(source).strip()
            if not snippet:
                continue
            snippet = snippet[:budget]
            budget -= len(snippet)
            blocks.append(f"--- {source.file_path} ---\n{snippet}")
            if budget <= 0:
                break
        return "\n\n".join(blocks)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Responde a pergunta a partir das sources; nunca levanta."""
        if not question.strip():
            return "No question was provided."
        context = self.build_context(sources)
        if not context:
            return "No relevant context was retrieved for this question."
        try:
            self.load()
            return self._generate(question, context)
        except Exception as err:  # noqa: BLE001 - a CLI nunca pode rebentar
            return f"Answer generation failed: {err}"

    def _generate(self, question: str, context: str) -> str:
        """Corre o modelo sobre o prompt ja montado."""
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                context=context, question=question)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._strip_thinking(answer).strip()

    @staticmethod
    def _strip_thinking(answer: str) -> str:
        """Qwen3 pode abrir um bloco <think>; so a resposta interessa."""
        if "</think>" in answer:
            answer = answer.split("</think>")[-1]
        return answer.replace("<think>", "")
