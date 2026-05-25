from backend.services.chunker import chunk_text
from backend.services.ollama_client import ask


def summarize_short(text: str) -> str:
    """
    For short texts that fit in one chunk — just ask directly.
    No map-reduce needed.
    """
    return ask(
        prompt=text,
        system="You are a document summarization assistant. Summarize the key points clearly and concisely in 3-5 sentences."
    )


def summarize_long(text: str) -> str:
    """
    Map-reduce summarization for documents too long for one context window.

    Map phase:   summarize each chunk independently
    Reduce phase: combine all mini-summaries into one final summary

    This handles documents of any length.
    """
    chunks = chunk_text(text)

    # MAP: one mini-summary per chunk
    print(f"[summarizer] {len(chunks)} chunks — starting map phase")
    mini_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[summarizer] summarizing chunk {i+1}/{len(chunks)}")
        mini = ask(
            prompt=chunk,
            system="You are summarizing one section of a longer document. Write 2-3 sentences capturing the key points of this section only."
        )
        mini_summaries.append(mini)

    # REDUCE: combine all mini-summaries
    print(f"[summarizer] reduce phase — combining {len(mini_summaries)} summaries")
    combined = "\n\n".join(
        f"Section {i+1}: {s}" for i, s in enumerate(mini_summaries)
    )
    final = ask(
        prompt=combined,
        system="You are given summaries of individual sections of a document. Write a coherent 4-6 sentence summary of the whole document based on these section summaries."
    )
    return final


def summarize(text: str, chunk_size: int = 2000) -> str:
    """
    Smart entry point: use the simple path for short texts,
    map-reduce for long ones. Callers don't need to think about this.
    """
    if len(text) <= chunk_size:
        return summarize_short(text)
    return summarize_long(text)
