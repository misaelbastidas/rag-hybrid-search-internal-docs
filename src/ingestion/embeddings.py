from openai import AzureOpenAI, OpenAI

from ingestion.openai_client import get_embedding_model, get_openai_client

DEFAULT_BATCH_SIZE = 100


def embed_texts(
    texts: list[str],
    model: str | None = None,
    client: OpenAI | AzureOpenAI | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Batched to stay under the embeddings API's per-request size/token limits on large corpora."""
    if not texts:
        return []
    client = client or get_openai_client()
    model = model or get_embedding_model()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings
