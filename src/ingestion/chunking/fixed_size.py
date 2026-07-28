from langchain_text_splitters import CharacterTextSplitter

from ingestion.loaders.base import LoadedDocument

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Baseline: fixed char window with overlap, no structure awareness."""

    strategy_name = "fixed_size"

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        texts = self.splitter.split_text(doc.text)
        return self._to_chunks(texts, doc)
