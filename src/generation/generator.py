from anthropic import Anthropic

from retrieval.base import RetrievedChunk

from .base import GeneratedAnswer
from .prompts import SYSTEM_PROMPT, build_user_message

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnswerGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, client: Anthropic | None = None):
        self.model = model
        self.client = client or Anthropic()

    def generate(self, query: str, chunks: list[RetrievedChunk], max_tokens: int = 1024) -> GeneratedAnswer:
        user_message = build_user_message(query, chunks)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = "".join(block.text for block in response.content if block.type == "text")
        return GeneratedAnswer(answer=answer_text, query=query, sources=chunks)
