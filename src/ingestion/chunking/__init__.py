from .base import BaseChunker, Chunk
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

_CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    chunker_cls = _CHUNKERS.get(strategy)
    if chunker_cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choices: {list(_CHUNKERS)}")
    return chunker_cls(**kwargs)
