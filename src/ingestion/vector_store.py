import os

import chromadb

from ingestion.chunking.base import Chunk

DEFAULT_PERSIST_DIR = "data/processed/chroma"
DEFAULT_COLLECTION = "docs"


class VectorStore:
    """Embedded (local file) by default. If CHROMA_HOST is set (e.g. in the containerized
    setup, where Chroma runs as its own service), connects to that server over HTTP instead."""

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR, collection_name: str = DEFAULT_COLLECTION):
        chroma_host = os.environ.get("CHROMA_HOST")
        if chroma_host:
            chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def is_near_duplicate(self, embedding: list[float], threshold: float = 0.95) -> bool:
        if self.collection.count() == 0:
            return False
        result = self.collection.query(query_embeddings=[embedding], n_results=1)
        distances = result.get("distances") or [[]]
        if not distances[0]:
            return False
        cosine_distance = distances[0][0]
        similarity = 1 - cosine_distance
        return similarity > threshold

    def list_documents(self) -> list[dict]:
        """One row per distinct source file, with its chunk count and how it was chunked."""
        if self.collection.count() == 0:
            return []

        result = self.collection.get(include=["metadatas"])
        by_source: dict[str, dict] = {}
        for meta in result["metadatas"]:
            source_path = meta["source_path"]
            entry = by_source.setdefault(
                source_path,
                {
                    "source_path": source_path,
                    "doc_type": meta["doc_type"],
                    "chunking_strategy": meta["chunking_strategy"],
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
        return list(by_source.values())

    def upsert(self, chunk: Chunk, embedding: list[float]) -> None:
        self.collection.upsert(
            ids=[chunk.id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[
                {
                    "source_path": chunk.source_path,
                    "doc_type": chunk.doc_type,
                    "chunk_index": chunk.chunk_index,
                    "chunking_strategy": chunk.chunking_strategy,
                    "char_count": chunk.char_count,
                    "section_heading": chunk.section_heading or "",
                    "page_num": chunk.page_num if chunk.page_num is not None else -1,
                }
            ],
        )
