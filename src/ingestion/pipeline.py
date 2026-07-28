from pathlib import Path

from ingestion.chunking import get_chunker
from ingestion.embeddings import embed_texts
from ingestion.loaders import load_directory
from ingestion.sparse_store import BM25Store
from ingestion.vector_store import VectorStore


def ingest_directory(
    dir_path: str | Path,
    strategy: str = "recursive",
    chunker_kwargs: dict | None = None,
    dedup_threshold: float = 0.95,
    persist_dir: str = "data/processed/chroma",
    bm25_path: str = "data/processed/bm25_index.pkl",
) -> dict:
    """Loads every doc under dir_path, chunks it, embeds, dedupes, and upserts into ChromaDB + BM25 in lockstep."""
    docs = load_directory(Path(dir_path))
    chunker = get_chunker(strategy, **(chunker_kwargs or {}))

    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))

    if not chunks:
        return {"total": 0, "inserted": 0, "skipped_duplicates": 0}

    embeddings = embed_texts([c.text for c in chunks])
    vector_store = VectorStore(persist_dir=persist_dir)
    bm25_store = BM25Store(index_path=bm25_path)

    inserted, skipped = 0, 0
    for chunk, embedding in zip(chunks, embeddings):
        if vector_store.is_near_duplicate(embedding, threshold=dedup_threshold):
            skipped += 1
            continue
        vector_store.upsert(chunk, embedding)
        bm25_store.add(chunk)
        inserted += 1

    bm25_store.save()

    return {"total": len(chunks), "inserted": inserted, "skipped_duplicates": skipped}
