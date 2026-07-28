from dataclasses import replace

from sentence_transformers import CrossEncoder

from .base import RetrievedChunk

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    """Cross-encoder: reads (query, chunk) together and scores fit directly, unlike dense/sparse which score independently."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        rescored = [
            replace(chunk, score=float(score), retrieval_method="reranked")
            for chunk, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]
