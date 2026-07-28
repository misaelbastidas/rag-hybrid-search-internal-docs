from pathlib import Path

from .base import BaseLoader, LoadedDocument
from .html_loader import HtmlLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PdfLoader
from .text_loader import TextLoader

_LOADERS: dict[str, BaseLoader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".txt": TextLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
    ".pdf": PdfLoader(),
}


def load_document(path: Path) -> list[LoadedDocument]:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader registered for extension: {path.suffix}")
    return loader.load(path)


def load_directory(dir_path: Path) -> list[LoadedDocument]:
    docs = []
    for path in sorted(Path(dir_path).rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            docs.extend(load_document(path))
    return docs
