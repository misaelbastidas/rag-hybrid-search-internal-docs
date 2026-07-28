import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingestion.chunking.base import Chunk

DEFAULT_INDEX_PATH = "data/processed/bm25_index.pkl"
TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Store:
    """Keeps ids aligned 1:1 with VectorStore so both indexes reference the same chunks."""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.tokenized_corpus = data["tokenized_corpus"]

    def add(self, chunk: Chunk) -> None:
        tokens = tokenize(chunk.text)
        if chunk.id in self.ids:
            self.tokenized_corpus[self.ids.index(chunk.id)] = tokens
        else:
            self.ids.append(chunk.id)
            self.tokenized_corpus.append(tokens)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"ids": self.ids, "tokenized_corpus": self.tokenized_corpus}, f)

    def build_bm25(self) -> BM25Okapi:
        return BM25Okapi(self.tokenized_corpus)

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        bm25 = self.build_bm25()
        scores = bm25.get_scores(tokenize(text))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
