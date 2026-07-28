from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedDocument:
    """Plaintext extracted from a source file plus metadata needed downstream for chunking/citation."""

    text: str
    source_path: str
    doc_type: str
    section_heading: str | None = None
    page_num: int | None = None
    extra: dict = field(default_factory=dict)


class BaseLoader:
    doc_type: str = "unknown"

    def load(self, path: Path) -> list[LoadedDocument]:
        raise NotImplementedError
