def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks that fit inside an LLM context window.

    chunk_size: max characters per chunk (~500 tokens, well within LLaMA's 8192 limit)
    overlap:    how many characters the next chunk repeats from the previous one
                preserves meaning at boundaries

    Returns a list of strings. A 100-page PDF (~200,000 chars) produces ~105 chunks.
    """
    if not text:
        return []

    # If the whole text fits in one chunk, no splitting needed
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap   # how far to advance the window each iteration
    i = 0

    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += step

    return chunks


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate. LLMs tokenize by subword units, but
    1 token ≈ 4 characters is a reliable rule of thumb for English.
    Useful for sanity-checking chunk sizes before sending to Ollama.
    """
    return len(text) // 4
