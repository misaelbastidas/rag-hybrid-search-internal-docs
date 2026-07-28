from ingestion.sparse_store import BM25Store
from ingestion.vector_store import VectorStore

from .base import RetrievedChunk


class SparseRetriever:
    """BM25 keyword search. Reuses ChromaDB to fetch text/metadata for the ids BM25 ranks, since both stores share the same ids."""

    def __init__(self, bm25_store: BM25Store | None = None, vector_store: VectorStore | None = None):
        self.bm25_store = bm25_store or BM25Store()
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        ranked = self.bm25_store.query(query, top_k=top_k)
        ranked = [(id_, score) for id_, score in ranked if score > 0]
        if not ranked:
            return []

        ids = [id_ for id_, _ in ranked]
        scores = dict(ranked)

        results = self.vector_store.collection.get(ids=ids, include=["documents", "metadatas"])
        assert results["documents"] is not None
        assert results["metadatas"] is not None

        chunks = []
        for id_, text, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            section_heading = str(meta["section_heading"]) if meta["section_heading"] else None
            page_num = int(meta["page_num"]) if meta["page_num"] != -1 else None  # type: ignore[arg-type]
            chunks.append(
                RetrievedChunk(
                    id=id_,
                    text=text,
                    source_path=str(meta["source_path"]),
                    score=scores[id_],
                    retrieval_method="sparse",
                    section_heading=section_heading,
                    page_num=page_num,
                    chunking_strategy=str(meta["chunking_strategy"]),
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
