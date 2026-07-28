import re
from pathlib import Path

from .base import BaseLoader, LoadedDocument

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownLoader(BaseLoader):
    """Splits on ATX headers so each section keeps its own heading in metadata; downstream chunkers work per-section."""

    doc_type = "markdown"

    def load(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = list(HEADING_RE.finditer(text))

        if not matches:
            return [LoadedDocument(text=text, source_path=str(path), doc_type=self.doc_type)]

        docs = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            docs.append(LoadedDocument(text=preamble, source_path=str(path), doc_type=self.doc_type))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            heading = match.group(2).strip()
            docs.append(
                LoadedDocument(
                    text=section_text,
                    source_path=str(path),
                    doc_type=self.doc_type,
                    section_heading=heading,
                )
            )
        return docs
