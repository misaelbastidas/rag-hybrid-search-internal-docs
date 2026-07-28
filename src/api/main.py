import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from generation.rag_pipeline import RAGPipeline
from ingestion.pipeline import ingest_directory
from ingestion.vector_store import VectorStore

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


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    pipeline: RAGPipeline = state["pipeline"]
    result = pipeline.ask(request.question, candidate_k=request.candidate_k, top_k=request.top_k)

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


@app.get("/v1/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    vector_store: VectorStore = state["vector_store"]
    return [DocumentInfo(**doc) for doc in vector_store.list_documents()]


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    chunker_kwargs = {}
    if request.chunk_size is not None:
        chunker_kwargs["chunk_size"] = request.chunk_size
    if request.chunk_overlap is not None:
        chunker_kwargs["chunk_overlap"] = request.chunk_overlap

    try:
        result = ingest_directory(request.dir, strategy=request.strategy, chunker_kwargs=chunker_kwargs or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(**result)
