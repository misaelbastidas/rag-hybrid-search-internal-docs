from .base import RetrievedChunk
from .dense import DenseRetriever
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .sparse import SparseRetriever


class HybridRetriever:
    def __init__(self, dense: DenseRetriever | None = None, sparse: SparseRetriever | None = None):
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 20,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[RetrievedChunk]:
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse.retrieve(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )
        return fused[:top_k]
