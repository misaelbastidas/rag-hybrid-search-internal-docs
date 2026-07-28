from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """One search result: a chunk plus the score that ranked it, tagged by which retrieval method found it."""

    id: str
    text: str
    source_path: str
    score: float
    retrieval_method: str
    section_heading: str | None = None
    page_num: int | None = None
    chunking_strategy: str | None = None
