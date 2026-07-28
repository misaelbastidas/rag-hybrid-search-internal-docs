from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.loaders.base import LoadedDocument

from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """Structure-aware: tries section/paragraph/sentence breaks before falling back to hard char cuts."""

    strategy_name = "recursive"

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n#", "\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        texts = self.splitter.split_text(doc.text)
        return self._to_chunks(texts, doc)
