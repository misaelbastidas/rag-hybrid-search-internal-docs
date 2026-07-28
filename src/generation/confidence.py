import json
import math
import re
from dataclasses import dataclass

from anthropic import Anthropic

from .base import GeneratedAnswer
from .citation_verifier import CitationCheck

COMPLETENESS_MODEL = "claude-haiku-4-5-20251001"

COMPLETENESS_SYSTEM_PROMPT = """You judge how completely an answer addresses a question, given the answer was generated only from limited context.
Respond with ONLY a JSON object: {"completeness": 0.0 to 1.0, "reason": "short explanation"}.
A correct, honest "I don't know" for a question the context can't answer should score 1.0 (it's the completeness of that response, not the question)."""


@dataclass
class ConfidenceScore:
    retrieval_confidence: float
    citation_coverage: float
    completeness: float
    composite: float


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def score_retrieval_confidence(chunks_with_rerank_scores: list, dense_top_score: float | None = None) -> float:
    """Cross-encoder scores are unbounded logits; squash the top chunk's score into 0-1 via sigmoid.

    The reranker occasionally tanks a specific phrasing (e.g. full grammatical yes/no questions)
    even when dense retrieval found the right chunk with a strong cosine match. dense_top_score,
    when given, acts as a rescue signal: take whichever of the two stages is more confident, so one
    stage's blind spot doesn't veto a genuinely correct answer the other stage already found.
    """
    if not chunks_with_rerank_scores:
        return 0.0
    rerank_confidence = _sigmoid(chunks_with_rerank_scores[0].score)
    if dense_top_score is None:
        return rerank_confidence
    # Calibrated against observed scores: ~0.35 cosine is a weak-but-real match (-> ~0.5),
    # ~0.45+ is a strong match (-> ~0.85+), matching the range seen across real queries.
    dense_confidence = _sigmoid((dense_top_score - 0.35) * 12)
    return max(rerank_confidence, dense_confidence)


def score_citation_coverage(citation_checks: list[CitationCheck]) -> float:
    if not citation_checks:
        return 0.0
    supported = sum(1 for c in citation_checks if c.supported)
    return supported / len(citation_checks)


class ConfidenceScorer:
    def __init__(self, client: Anthropic | None = None, model: str = COMPLETENESS_MODEL):
        self.client = client or Anthropic()
        self.model = model

    def score(
        self,
        generated: GeneratedAnswer,
        citation_checks: list[CitationCheck],
        retrieval_weight: float = 0.4,
        citation_weight: float = 0.3,
        completeness_weight: float = 0.3,
        dense_top_score: float | None = None,
    ) -> ConfidenceScore:
        retrieval_confidence = score_retrieval_confidence(generated.sources, dense_top_score)
        citation_coverage = score_citation_coverage(citation_checks)
        completeness = self._judge_completeness(generated)

        composite = (
            retrieval_weight * retrieval_confidence
            + citation_weight * citation_coverage
            + completeness_weight * completeness
        )

        return ConfidenceScore(
            retrieval_confidence=retrieval_confidence,
            citation_coverage=citation_coverage,
            completeness=completeness,
            composite=composite,
        )

    def _judge_completeness(self, generated: GeneratedAnswer) -> float:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=COMPLETENESS_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {generated.query}\n\nAnswer: {generated.answer}",
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return 0.0
        return float(json.loads(json_match.group(0))["completeness"])
