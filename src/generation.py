"""Generate answers with Qwen3-0.6B from the retrieved sources."""

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
    """Loads the local model once and answers using the provided context."""

    def __init__(self, model_name: str = MODEL_NAME,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the model on demand.

        The import lives here on purpose: anyone doing only indexing or search
        does not pay the seconds it takes for torch to start up.
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
        """Read from disk the snippet of file pointed to by the source."""
        try:
            with open(source.file_path, "r", encoding="utf-8",
                      errors="ignore") as handle:
                text = handle.read()
        except OSError:
            return ""
        return text[source.first_character_index:
                    source.last_character_index]

    def build_context(self, sources: list[MinimalSource]) -> str:
        """Combine the best sources into one context under a fixed budget.

        Truncating here is intentional: the model has 0.6B parameters and it
        drowns in too much context as much as it does with too little.
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
        """Answer a question from the retrieved sources without raising."""
        if not question.strip():
            return "No question was provided."
        context = self.build_context(sources)
        if not context:
            return "No relevant context was retrieved for this question."
        try:
            self.load()
            return self._generate(question, context)
        except Exception as err:  # noqa: BLE001 - the CLI must never crash
            return f"Answer generation failed: {err}"

    def _generate(self, question: str, context: str) -> str:
        """Run the model on the assembled prompt."""
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
        """Qwen3 may open a <think> block; only the final answer matters."""
        if "</think>" in answer:
            answer = answer.split("</think>")[-1]
        return answer.replace("<think>", "")
