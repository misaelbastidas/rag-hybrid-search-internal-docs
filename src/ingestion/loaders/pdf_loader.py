from pathlib import Path

from pypdf import PdfReader

from .base import BaseLoader, LoadedDocument


class PdfLoader(BaseLoader):
    """One LoadedDocument per page; page_num is the citation anchor since PDFs rarely have clean headers."""

    doc_type = "pdf"

    def load(self, path: Path) -> list[LoadedDocument]:
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            docs.append(
                LoadedDocument(
                    text=text,
                    source_path=str(path),
                    doc_type=self.doc_type,
                    page_num=i + 1,
                )
            )
        return docs
