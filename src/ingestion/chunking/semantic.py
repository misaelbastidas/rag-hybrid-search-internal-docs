import re

import numpy as np
from openai import AzureOpenAI, OpenAI

from ingestion.loaders.base import LoadedDocument
from ingestion.openai_client import get_embedding_model, get_openai_client

from .base import BaseChunker, Chunk

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticChunker(BaseChunker):
    """Splits on topic boundaries: embed each sentence, break where similarity to the next sentence drops below threshold."""

    strategy_name = "semantic"

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1000,
        embedding_model: str | None = None,
        client: OpenAI | AzureOpenAI | None = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.embedding_model = embedding_model or get_embedding_model()
        self.client = client or get_openai_client()

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(doc.text) if s.strip()]
        if len(sentences) <= 1:
            return self._to_chunks([doc.text], doc)

        response = self.client.embeddings.create(model=self.embedding_model, input=sentences)
        embeddings = [np.array(e.embedding) for e in response.data]

        texts: list[str] = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            similarity = _cosine(embeddings[i - 1], embeddings[i])
            sentence = sentences[i]
            is_topic_break = similarity < self.similarity_threshold
            exceeds_size = current_len + len(sentence) > self.max_chunk_size

            if is_topic_break or exceeds_size:
                texts.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_len = len(sentence)
            else:
                current_sentences.append(sentence)
                current_len += len(sentence)

        if current_sentences:
            texts.append(" ".join(current_sentences))

        return self._to_chunks(texts, doc)
