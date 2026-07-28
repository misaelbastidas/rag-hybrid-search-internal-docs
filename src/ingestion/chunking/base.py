import hashlib
from dataclasses import dataclass

from ingestion.loaders.base import LoadedDocument


@dataclass
class Chunk:
    """Embedding-sized piece of a LoadedDocument. chunking_strategy is tracked so eval can compare strategies later."""

    text: str
    source_path: str
    doc_type: str
    chunk_index: int
    chunking_strategy: str
    char_count: int
    section_heading: str | None = None
    page_num: int | None = None

    @property
    def id(self) -> str:
        """Stable id for upsert/dedup: same doc + section/page + strategy + index always maps to the same id."""
        raw = f"{self.source_path}|{self.section_heading}|{self.page_num}|{self.chunking_strategy}|{self.chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()


class BaseChunker:
    strategy_name: str = "unknown"

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        raise NotImplementedError

    def _to_chunks(self, texts: list[str], doc: LoadedDocument) -> list[Chunk]:
        return [
            Chunk(
                text=text,
                source_path=doc.source_path,
                doc_type=doc.doc_type,
                chunk_index=i,
                chunking_strategy=self.strategy_name,
                char_count=len(text),
                section_heading=doc.section_heading,
                page_num=doc.page_num,
            )
            for i, text in enumerate(texts)
            if text.strip()
        ]
