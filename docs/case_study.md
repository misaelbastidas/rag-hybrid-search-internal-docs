# Case Study: Hybrid RAG Over Internal Documents

## Headline numbers

Measured on a 70-question golden evaluation set (mixed lookup / multi-hop / ambiguous / no-answer questions, independently fact-checked against source documents before use), run end-to-end through the real pipeline — no mocking, real embedding calls, real LLM generation and judging.

| Metric | Score |
|---|---|
| Composite | **95.5%** |
| Correctness (vs. golden reference) | **93.6%** |
| Faithfulness (no unsupported claims) | **98.3%** |
| Citation accuracy | **93.7%** |
| Retrieval relevance | **90.0%** |

## The corpus

100 real internal policy documents (PDF, Spanish) for a fictional quick-service restaurant chain operating in Mexico and the US — HR, finance, legal, operations, food safety, IT security, marketing, and sustainability policies. ~428K characters, chunked into 515 chunks after near-duplicate detection (recursive chunking, 22 duplicates skipped from repeated boilerplate headers).

## Does hybrid retrieval actually beat dense-only?

It's a common claim in RAG tutorials, rarely measured. I ran the identical 70-question suite twice — once through the full hybrid pipeline (dense + BM25 + RRF fusion + reranking), once with sparse retrieval and fusion disabled entirely (dense embeddings + reranking only), same corpus, same chunking, same everything else.

| | Hybrid | Dense-only | Δ |
|---|---|---|---|
| Composite | **0.955** | 0.946 | +0.009 |
| Correctness | 0.936 | 0.930 | +0.006 |
| Faithfulness | 0.983 | 0.973 | +0.010 |
| Citation accuracy | **0.937** | 0.910 | **+0.027** |
| Retrieval relevance | 0.900 | 0.900 | +0.000 |

**Honest finding: the gain is real but modest, not dramatic.** `text-embedding-3-small` is a strong enough dense model that it finds the correct source document about as often as hybrid does (`retrieval_relevance` is identical). Hybrid's advantage shows up more in *answer quality* than in *document retrieval* — the biggest gap is citation accuracy, where BM25's exact keyword matching seems to help ground specific claims in the right passage. This corpus is prose-heavy policy documents, not dense with error codes, part numbers, or exact technical identifiers — the domain where BM25 typically provides its largest lift over dense embeddings. A corpus with more of that character would likely show a bigger hybrid advantage.

## Chunking strategy comparison

All three strategies indexed the same 100 documents separately, evaluated under the identical current pipeline for a fair comparison.

| Strategy | Composite | Correctness | Faithfulness | Citation | Chunks produced |
|---|---|---|---|---|---|
| **Recursive** | 0.955 | 0.936 | 0.983 | 0.937 | 515 |
| **Fixed-size** | 0.956 | 0.936 | 0.979 | 0.940 | 515 |
| Semantic | 0.886 | 0.785 | 0.984 | 0.881 | 2,756 |

Recursive and fixed-size are effectively tied. **Semantic chunking clearly underperforms** — most visible on ambiguous questions (composite 0.749 vs. ~0.96 for the other two, correctness dropping to 0.48). Root cause, confirmed directly: semantic chunking's sentence-level topic-boundary detection triggers far more aggressively than expected on this corpus — 2,756 chunks vs. 515 for the other strategies, even after equalizing the max chunk size across all three. Short, fact-dense policy sentences register as "topic shifts" to the similarity threshold much more often than in the long-form narrative text this technique is usually designed for. More fragmentation hurts exactly the questions that need multiple facts assembled together (ambiguous, multi-hop) — which is exactly what the data shows.

**Production choice: recursive chunking.** Same quality as fixed-size, but structurally sounder — it tries paragraph/section breaks before falling back to hard character cuts, which matters for auditability even when the eval numbers come out close.

## Things the evaluation framework actually caught

Building the eval suite wasn't just a formality — it surfaced and helped fix three real bugs during development, each validated with before/after numbers on the same 70-question suite:

1. **Confidence-gate fragility (multi-hop retrieval, chunk size).** An early smoke test showed a wrongly-declined question despite the correct source being retrieved. Root causes: default chunking (`chunk_size=500`) was splitting a data table across chunk boundaries mid-row, and multi-hop questions needing 2 source documents often only had 1 survive the reranker's top-k cut. Fixing both (`chunk_size` → 1500, `top_k` → 8) took multi-hop correctness from 48% to 79% in one pass.
2. **Cross-lingual usability.** The English-trained reranker gave strongly negative relevance scores to correct Spanish passages when queried in English, even though dense retrieval had found the right document with a confident cosine match. Swapping to a multilingual cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) fixed English usability *and* improved Spanish performance simultaneously (composite 0.894 → 0.926) — not a tradeoff, a straight upgrade.
3. **Reranker phrasing sensitivity.** Even after the multilingual swap, some grammatically complete yes/no questions ("is there a loyalty program?") scored catastrophically low from the reranker alone, despite dense retrieval being confident (0.47 cosine) about the same, correct document. Fix: the confidence gate now takes the max of reranker and dense confidence, not reranker alone — a cheap rescue signal (one extra embedding call, no LLM) that took composite from 0.926 to 0.955.

Each fix was validated by re-running the full suite, not by anecdote — the discipline the eval framework was built to enforce.
