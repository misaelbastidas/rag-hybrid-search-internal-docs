# CLAUDE.md — RAG Pipeline with Hybrid Search Over Internal Docs

## What We're Building
Production-grade Retrieval-Augmented Generation system. Ingests company internal docs, indexes with **both** dense vector + sparse keyword search, retrieves best context per question, generates grounded answers with inline source citations.

Differentiator vs toy demos: hybrid retrieval, switchable chunking strategies, citation verification, confidence scoring, eval framework.

## Tech Stack
| Component | Tool | Reason |
|-----------|------|--------|
| Language | Python 3.11+ | Ecosystem standard |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap, high quality |
| Vector Store | ChromaDB (start file-based) | Simple local, swap to Qdrant later |
| Sparse Search | BM25 via `rank_bm25` | Exact keyword match |
| LLM | GPT-4o **or** Claude Sonnet | Strong grounding + citation |
| Chunking | LangChain text splitters | Configurable overlap/size |
| API | FastAPI | Async-native, production |
| Frontend | Streamlit | Fast query dashboard |
| Container | Docker + docker-compose | Reproducible deploy |

## Current Status
**Phase 0 — Empty repo.** Nothing built yet. Next step: Phase 1.

Build order = follow phases below step by step with Sonnet. Mark each item `[x]` when done + validated (compiles, tests pass, evidence).

## Build Plan

### Phase 1 — Ingestion + Chunking (Day 1–3)
- [ ] Multi-format loader: markdown, text, HTML, PDF → clean plaintext + metadata (source file, section heading, page num). Keep raw docs beside processed for re-index without re-upload.
- [ ] Configurable chunking, 3 switchable strategies: (a) fixed-size + overlap (baseline), (b) recursive char split by section headers (structure-aware), (c) semantic split on topic boundaries via embedding similarity. Track strategy per chunk.
- [ ] Generate + store embeddings via `text-embedding-3-small`. ChromaDB metadata: source doc, chunk index, section heading, chunking strategy, char count. Build BM25 index in parallel over same chunks. **Both indexes stay in sync.**
- [ ] Deduplication: before insert, check near-dupe (cosine > 0.95 vs existing). Flag + skip. Saves context window slots.

### Phase 2 — Hybrid Retrieval Engine (Day 3–6)
- [ ] Dense retrieval: embed query, top-k by cosine. Start k=10.
- [ ] Sparse retrieval: same query through BM25, top-k by score. Catches function names, config keys, error codes.
- [ ] Fusion layer: Reciprocal Rank Fusion (RRF) merge dense + sparse. Configurable weight (e.g. 0.7 dense / 0.3 sparse).
- [ ] Reranker: top 20 candidates → cross-encoder (or LLM-as-judge) → keep top 5. Big precision boost.

### Phase 3 — Generation + Citation (Day 6–9)
- [ ] Grounded generation prompt: answer only from context, cite chunks with `[1]`,`[2]`, state when context insufficient. Retrieved chunks as numbered blocks.
- [ ] Citation verification: parse citations, verify each claim→citation via LLM-as-judge. Flag unsupported.
- [ ] Confidence scorer: composite of retrieval confidence + citation coverage + answer completeness.
- [ ] "I don't know" handling: if retrieval confidence < threshold, no hallucination. Structured response: what found, what missing, which docs to check manually.

### Phase 4 — Evaluation Framework (Day 9–11)
- [ ] Golden Q&A dataset: 50+ hand-written pairs tied to corpus sections. Mix: straight lookups, multi-hop, no-answer, ambiguous.
- [ ] Auto eval metrics per case: answer correctness (LLM-judge vs golden), faithfulness, retrieval relevance, citation accuracy. Run full suite on every pipeline change.
- [ ] Chunking strategy comparison: run eval across all 3 strategies → comparison report (which wins which metric). Drives architecture decisions + interview numbers.

### Phase 5 — API + Dashboard (Day 11–13)
- [ ] FastAPI: `POST /v1/ask` (question → answer + citations + confidence + source metadata), `GET /v1/documents` (list indexed), `POST /v1/ingest` (add docs). OpenAPI docs.
- [ ] Streamlit dashboard: ask questions, see answer with clickable citations, retrieved chunks ranked, confidence breakdown, toggle hybrid vs dense-only side by side.
- [ ] Containerize: docker-compose with API + ChromaDB + frontend. Seed script indexes sample corpus for instant reviewer spin-up.

### Phase 6 — Portfolio Polish (Day 13–14)
- [ ] Demo walkthrough <4 min: ingest docs, ask varying difficulty, citation verification catching hallucination, hybrid vs dense-only comparison.
- [ ] Case study: lead with numbers ("X% faithfulness, Y% citation accuracy on 50-question eval"). Explain why hybrid beats dense-only for technical docs. Show chunking comparison data.

## Working Rules
- Surgical edits. Change only what needed. No unrequested abstractions.
- Validate before "done": compile, run tests, show evidence.
- Keep both indexes (vector + BM25) in sync — core invariant.
- Config-driven where plan says "configurable" (chunking, RRF weights, k, thresholds).
- Secrets (OpenAI key) via env var, never committed.

## Proposed Structure (create as we go)
```
src/
  ingestion/    # loaders, chunkers, embeddings, dedup
  retrieval/    # dense, sparse, fusion, rerank
  generation/   # prompt, citation verify, confidence
  eval/         # golden dataset, metrics, comparison
  api/          # FastAPI app
dashboard/      # Streamlit
data/
  raw/          # uploaded docs
  processed/    # normalized plaintext + metadata
tests/
docker-compose.yml
requirements.txt
.env.example
```
