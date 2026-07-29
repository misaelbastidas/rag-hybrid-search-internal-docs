from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    candidate_k: int = 20
    top_k: int = 8


class SourceOut(BaseModel):
    id: str
    text: str
    source_path: str
    score: float
    section_heading: str | None = None
    page_num: int | None = None
    chunking_strategy: str | None = None


class ConfidenceOut(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    completeness: float
    composite: float


class AskResponse(BaseModel):
    query: str
    answered: bool
    answer: str | None = None
    sources: list[SourceOut] = []
    confidence: ConfidenceOut | None = None
    message: str | None = None


class DocumentInfo(BaseModel):
    source_path: str
    doc_type: str
    chunking_strategy: str
    chunk_count: int


class IngestRequest(BaseModel):
    dir: str = "data/raw"
    strategy: str = "recursive"
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    dedup_threshold: float = 0.95


class IngestResponse(BaseModel):
    total: int
    inserted: int
    skipped_duplicates: int
