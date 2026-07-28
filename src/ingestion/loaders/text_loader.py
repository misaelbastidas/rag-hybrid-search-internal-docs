from pathlib import Path

from .base import BaseLoader, LoadedDocument


class TextLoader(BaseLoader):
    doc_type = "text"

    def load(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [LoadedDocument(text=text, source_path=str(path), doc_type=self.doc_type)]
