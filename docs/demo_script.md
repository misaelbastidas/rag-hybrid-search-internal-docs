# Demo Walkthrough Script (~4 min)

Record with the app already running (`docker compose up` or local dev), corpus already ingested. Talk over screen recording of the chat UI at `http://localhost:5173`.

## 0:00–0:25 — Intro (25s)

> "This is a hybrid-search RAG pipeline over a set of internal company documents — 100 real policy PDFs from a fictional restaurant chain, in Spanish. Dense embeddings plus BM25 keyword search, reciprocal rank fusion, cross-encoder reranking, and grounded generation with citation verification and confidence scoring."

Show the running app briefly — chat UI, and `docker compose up` output or the architecture diagram in the README.

## 0:25–1:00 — Ingestion (35s)

> "Ingestion supports markdown, text, HTML, and PDF. Each document is chunked, embedded, and indexed into both ChromaDB and a BM25 index in parallel — those two stay in sync by design. Near-duplicate chunks get deduplicated automatically."

Show `python scripts/cli.py ingest data/raw` output (or the auto-seed log from `docker compose up`):
```
{'total': 515, 'inserted': 493, 'skipped_duplicates': 22}
```

## 1:00–2:15 — Ask varying difficulty (75s)

**Simple lookup**:
> "How many vacation days do I get with 1 year of tenure in Mexico?"

Point out: cited answer (`[2]`), confidence badges, expandable sources.

**Multi-hop** (English, cross-lingual):
> "There is a loyalty program in the company?"

Point out: answers in English even though the source documents are Spanish — the model reads the retrieved Spanish text and answers in the query's language automatically.

**Ambiguous**:
> "How long does Naguara58 have to resolve a customer complaint?"

Point out: the answer correctly explains the SLA *depends on complaint category* (4h/24h/48h) rather than picking one number arbitrarily — the golden eval dataset specifically tests this pattern.

## 2:15–3:00 — Refusing to hallucinate (45s)

> "One of the harder things to get right: knowing when *not* to answer."

Ask something plausible-sounding but not in the corpus:
> "What is Naguara58's approval procedure for mergers and acquisitions?"

Point out: retrieval finds a *topically related* document (corporate governance), high confidence — but the answer explicitly states the specific fact isn't covered, rather than fabricating a plausible-sounding procedure. This is verified automatically in the eval suite via an LLM-judge that checks whether declines are "honest" vs "fabricated."

## 3:00–3:40 — Hybrid vs. dense-only, with real numbers (40s)

> "Is hybrid retrieval actually worth the extra complexity over dense-only? I measured it — ran the full 70-question golden eval suite both ways, same corpus, same everything else."

Show the comparison table (see `docs/case_study.md`):
- Composite: hybrid **0.955** vs dense-only **0.946**
- Citation accuracy: hybrid **0.937** vs dense-only **0.910** — the biggest gap

> "Modest but consistent — biggest win is citation accuracy, where BM25's exact keyword matching helps ground specific claims sparse retrieval or dense wouldn't reliably surface alone."

## 3:40–4:00 — Close (20s)

> "Full eval framework compares 3 chunking strategies, generates confidence scores per answer, and everything's containerized — FastAPI backend, React frontend, standalone ChromaDB, one `docker compose up`. Code and case study on GitHub."
