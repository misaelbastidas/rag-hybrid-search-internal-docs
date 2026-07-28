from dataclasses import dataclass, field

from retrieval.base import RetrievedChunk
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker

from .citation_verifier import CitationCheck, CitationVerifier
from .confidence import ConfidenceScore, ConfidenceScorer, score_retrieval_confidence
from .generator import AnswerGenerator


@dataclass
class RAGResponse:
    query: str
    answered: bool
    sources: list[RetrievedChunk]
    answer: str | None = None
    confidence: ConfidenceScore | None = None
    citation_checks: list[CitationCheck] = field(default_factory=list)
    message: str | None = None


def _build_insufficient_message(chunks: list[RetrievedChunk], retrieval_confidence: float) -> str:
    if not chunks:
        return "No documents matched this question. Nothing found to check."

    lines = [
        f"Retrieval confidence too low ({retrieval_confidence:.2f}) to answer reliably - "
        "skipping generation to avoid a guess.",
        "Closest sections found, worth checking manually:",
    ]
    for c in chunks[:5]:
        location = c.section_heading or (f"page {c.page_num}" if c.page_num else "")
        lines.append(f"  - {c.source_path}" + (f" ({location})" if location else ""))
    return "\n".join(lines)


class RAGPipeline:
    """Full ask flow: retrieve -> rerank -> gate on retrieval confidence -> generate -> verify -> score.
    Generation is skipped entirely below the confidence threshold, so a low-confidence query costs no LLM call."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        generator: AnswerGenerator | None = None,
        verifier: CitationVerifier | None = None,
        scorer: ConfidenceScorer | None = None,
        retrieval_confidence_threshold: float = 0.2,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.generator = generator or AnswerGenerator()
        self.verifier = verifier or CitationVerifier()
        self.scorer = scorer or ConfidenceScorer()
        self.retrieval_confidence_threshold = retrieval_confidence_threshold

    def ask(self, query: str, candidate_k: int = 20, top_k: int = 8) -> RAGResponse:
        candidates = self.retriever.retrieve(query, top_k=candidate_k, candidate_k=candidate_k)
        if not candidates:
            return RAGResponse(
                query=query,
                answered=False,
                sources=[],
                message=_build_insufficient_message([], 0.0),
            )

        top_chunks = self.reranker.rerank(query, candidates, top_k=top_k)

        # Cheap secondary signal (one embedding call, no LLM): dense cosine match, used as a
        # rescue when the reranker tanks a specific phrasing dense retrieval was actually confident about.
        dense_top = self.retriever.dense.retrieve(query, top_k=1)
        dense_top_score = dense_top[0].score if dense_top else None

        retrieval_confidence = score_retrieval_confidence(top_chunks, dense_top_score)

        if retrieval_confidence < self.retrieval_confidence_threshold:
            return RAGResponse(
                query=query,
                answered=False,
                sources=top_chunks,
                message=_build_insufficient_message(top_chunks, retrieval_confidence),
            )

        generated = self.generator.generate(query, top_chunks)
        checks = self.verifier.verify(generated)
        confidence = self.scorer.score(generated, checks, dense_top_score=dense_top_score)

        return RAGResponse(
            query=query,
            answered=True,
            sources=top_chunks,
            answer=generated.answer,
            confidence=confidence,
            citation_checks=checks,
        )
