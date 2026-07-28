from ingestion.embeddings import embed_texts
from ingestion.vector_store import VectorStore

from .base import RetrievedChunk


class DenseRetriever:
    """Embeds the query, finds nearest chunks in ChromaDB by cosine similarity."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        query_embedding = embed_texts([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        assert results["documents"] is not None
        assert results["metadatas"] is not None
        assert results["distances"] is not None

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        chunks = []
        for id_, text, meta, distance in zip(ids, documents, metadatas, distances):
            similarity = 1 - distance
            section_heading = str(meta["section_heading"]) if meta["section_heading"] else None
            page_num = int(meta["page_num"]) if meta["page_num"] != -1 else None  # type: ignore[arg-type]
            chunks.append(
                RetrievedChunk(
                    id=id_,
                    text=text,
                    source_path=str(meta["source_path"]),
                    score=similarity,
                    retrieval_method="dense",
                    section_heading=section_heading,
                    page_num=page_num,
                    chunking_strategy=str(meta["chunking_strategy"]),
                )
            )
        return chunks
