# src/embeddings.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
) -> list[float]:
    """
    Returns a dense embedding vector for the given text.
    """
    text = text or ""
    resp = _client.embeddings.create(
        model=model,
        input=text
    )
    return resp.data[0].embedding
