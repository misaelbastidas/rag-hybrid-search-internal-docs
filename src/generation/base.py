from dataclasses import dataclass

from retrieval.base import RetrievedChunk


@dataclass
class GeneratedAnswer:
    """The model's answer plus the exact source chunks it was shown, so citations like [1] can be resolved back to a real chunk."""

    answer: str
    query: str
    sources: list[RetrievedChunk]
