# backend/services/embedder.py
#
# Responsible for one thing: turning a string of text into a vector (embedding).
#
# Why a separate file? Because the embedding model (nomic-embed-text) and the
# generation model (llama3) are completely different things. Keeping them
# in separate modules makes it easy to swap one without touching the other.

import requests
import os
from dotenv import load_dotenv

load_dotenv()

# The embedding endpoint is on the same Ollama instance, different path.
# Example: http://ollama:11434/api/embeddings  (inside Docker network)
#          http://localhost:11434/api/embeddings  (outside Docker)
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")

# Derive the embeddings URL from the base URL.
# We strip /api/generate and add /api/embeddings instead.
# This way there's only one place to configure the Ollama host.
EMBED_URL = OLLAMA_BASE_URL.replace("/api/generate", "/api/embeddings")

# nomic-embed-text is a small, fast model trained specifically to produce
# high-quality embeddings. It outputs 768-dimensional vectors.
# It is NOT a text generator — you can't ask it questions.
# Think of it as a "meaning compressor": text in → numbers out.
EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    """
    Send a piece of text to Ollama's embedding endpoint.
    Returns a list of ~768 floats representing the semantic meaning of the text.

    These numbers have no human-readable meaning on their own — they only make
    sense when compared to other vectors. Two vectors that are "close" (high
    cosine similarity) mean the texts have similar meaning.

    Args:
        text: Any string. Works best when it's a few sentences to a paragraph.
              Very short strings (1-2 words) give less accurate embeddings.

    Returns:
        A list of floats, e.g. [0.023, -0.91, 0.44, ...] (~768 values)

    Raises:
        requests.HTTPError: if Ollama is unreachable or the model isn't pulled.
    """
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,  # Note: Ollama uses "prompt" not "input" here
    }

    response = requests.post(EMBED_URL, json=payload)

    # Raise an exception immediately if the HTTP status is 4xx or 5xx.
    # Better to fail loudly than to silently return bad data.
    response.raise_for_status()

    data = response.json()

    # Ollama returns: { "embedding": [0.023, -0.91, ...] }
    # We just return the list directly — callers don't need the wrapper dict.
    return data["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts, one at a time.

    Ollama doesn't support true batch embedding (sending many texts in one
    request), so we loop. For a 100-page PDF with ~100 chunks, this takes
    roughly 10-30 seconds depending on your hardware. That's acceptable —
    it only happens once per upload, not on every question.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of vectors, in the same order as the input texts.
    """
    embeddings = []

    for i, text in enumerate(texts):
        print(f"[embedder] embedding chunk {i + 1}/{len(texts)}")
        vector = embed(text)
        embeddings.append(vector)

    return embeddings
