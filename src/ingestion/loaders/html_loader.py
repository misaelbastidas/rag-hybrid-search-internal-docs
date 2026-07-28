from pathlib import Path

from bs4 import BeautifulSoup

from .base import BaseLoader, LoadedDocument

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


class HtmlLoader(BaseLoader):
    """Splits on heading tags (h1-h6), mirroring MarkdownLoader's section-per-heading approach."""

    doc_type = "html"

    def load(self, path: Path) -> list[LoadedDocument]:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup(["script", "style"]):
            tag.decompose()

        headings = soup.find_all(HEADING_TAGS)
        if not headings:
            text = soup.get_text(separator="\n", strip=True)
            return [LoadedDocument(text=text, source_path=str(path), doc_type=self.doc_type)]

        docs = []
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            section_parts = [heading_text]
            for sibling in heading.find_next_siblings():
                if sibling.name in HEADING_TAGS:
                    break
                section_parts.append(sibling.get_text(separator="\n", strip=True))
            section_text = "\n".join(p for p in section_parts if p)
            docs.append(
                LoadedDocument(
                    text=section_text,
                    source_path=str(path),
                    doc_type=self.doc_type,
                    section_heading=heading_text,
                )
            )
        return docs
