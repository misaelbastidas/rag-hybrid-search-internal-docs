import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from generation.rag_pipeline import RAGPipeline
from ingestion.pipeline import ingest_directory
from ingestion.vector_store import VectorStore

from .auth import require_admin_key
from .rate_limit import rate_limiter
from .schemas import (
    AskRequest,
    AskResponse,
    ConfidenceOut,
    DocumentInfo,
    IngestRequest,
    IngestResponse,
    SourceOut,
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store = VectorStore()
    state["vector_store"] = vector_store

    if os.environ.get("AUTO_SEED", "false").lower() == "true" and vector_store.collection.count() == 0:
        seed_dir = os.environ.get("AUTO_SEED_DIR", "data/raw")
        print(f"Empty index detected, auto-seeding from {seed_dir}...")
        result = ingest_directory(seed_dir, strategy="recursive")
        print(f"Auto-seed complete: {result}")

    state["pipeline"] = RAGPipeline()
    yield
    state.clear()


app = FastAPI(
    title="RAG Pipeline with Hybrid Search",
    description="Hybrid (dense + BM25) retrieval, reranking, grounded generation with citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every /v1/ask call fires several paid LLM/embedding calls, so a public deployment
# needs per-IP limits to avoid one bot or over-enthusiastic visitor running up a real bill.


@app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(rate_limiter(10, 60))])
def ask(payload: AskRequest) -> AskResponse:
    pipeline: RAGPipeline = state["pipeline"]
    result = pipeline.ask(payload.question, candidate_k=payload.candidate_k, top_k=payload.top_k)

    return AskResponse(
        query=result.query,
        answered=result.answered,
        answer=result.answer,
        message=result.message,
        sources=[
            SourceOut(
                id=s.id,
                text=s.text,
                source_path=s.source_path,
                score=s.score,
                section_heading=s.section_heading,
                page_num=s.page_num,
                chunking_strategy=s.chunking_strategy,
            )
            for s in result.sources
        ],
        confidence=(
            ConfidenceOut(
                retrieval_confidence=result.confidence.retrieval_confidence,
                citation_coverage=result.confidence.citation_coverage,
                completeness=result.confidence.completeness,
                composite=result.confidence.composite,
            )
            if result.confidence
            else None
        ),
    )


@app.get("/v1/documents", response_model=list[DocumentInfo], dependencies=[Depends(rate_limiter(30, 60))])
def list_documents() -> list[DocumentInfo]:
    vector_store: VectorStore = state["vector_store"]
    return [DocumentInfo(**doc) for doc in vector_store.list_documents()]


@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_admin_key), Depends(rate_limiter(2, 3600))],
)
def ingest(payload: IngestRequest) -> IngestResponse:
    chunker_kwargs = {}
    if payload.chunk_size is not None:
        chunker_kwargs["chunk_size"] = payload.chunk_size
    if payload.chunk_overlap is not None:
        chunker_kwargs["chunk_overlap"] = payload.chunk_overlap

    try:
        result = ingest_directory(payload.dir, strategy=payload.strategy, chunker_kwargs=chunker_kwargs or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(**result)
