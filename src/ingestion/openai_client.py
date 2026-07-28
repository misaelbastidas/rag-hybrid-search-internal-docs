import os

from openai import AzureOpenAI, OpenAI


def get_openai_client() -> OpenAI | AzureOpenAI:
    """Azure if AZURE_OPENAI_ENDPOINT is set, else standard OpenAI."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return OpenAI()


def get_embedding_model() -> str:
    """Azure deployment name if configured, else the OpenAI model id."""
    return os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
