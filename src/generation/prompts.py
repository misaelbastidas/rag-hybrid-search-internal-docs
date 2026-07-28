from retrieval.base import RetrievedChunk

SYSTEM_PROMPT = """You are a careful assistant answering questions using ONLY the numbered context blocks provided.

Rules:
- Answer only using information found in the context blocks below. Do not use outside knowledge.
- Every factual claim must be followed by a citation like [1] or [2] referencing the context block it came from.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        location = chunk.section_heading or (f"page {chunk.page_num}" if chunk.page_num else "")
        header = f"[{i}] Source: {chunk.source_path}" + (f" ({location})" if location else "")
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"
