from dataclasses import replace

from .base import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """RRF: a chunk's fused score is 1/(k+rank) per list it appears in, weighted per list, then summed.
    Rank-based (not raw score) so dense's 0-1 similarity and sparse's unbounded BM25 score are comparable."""
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + dense_weight / (rrf_k + rank)
        chunks_by_id[chunk.id] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + sparse_weight / (rrf_k + rank)
        chunks_by_id.setdefault(chunk.id, chunk)

    fused = [
        replace(chunks_by_id[id_], score=score, retrieval_method="hybrid")
        for id_, score in fused_scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused
