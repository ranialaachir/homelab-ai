# backend/services/rag.py
#
# RAG = Retrieval-Augmented Generation.
#
# This module owns the "generation" half of RAG.
# The "retrieval" half lives in vector_store.py.
#
# Its one job: given a question and a list of relevant chunks,
# build a well-structured prompt and call LLaMA 3.
#
# Why does prompt structure matter?
# LLaMA 3 is very sensitive to how you frame the context. If you just dump
# the chunks in without framing, the model may ignore them or mix them up
# with its training knowledge. The system prompt below tells it explicitly:
# "only use what I give you, don't invent anything."

from backend.services.ollama_client import ask


def answer_from_chunks(question: str, chunks: list[str]) -> str:
    """
    Generate an answer to a question, grounded only in the provided chunks.

    This is the final step of the RAG pipeline. The chunks were already
    retrieved by vector_store.query_chunks() — they are the most relevant
    pieces of the PDF for this question.

    We format them into a numbered context block, then send everything
    to LLaMA 3 with a strict system prompt that prevents hallucination.

    Args:
        question: The user's question, as a plain string.
        chunks:   List of relevant text chunks from the PDF (most relevant first).

    Returns:
        LLaMA's answer as a string.
    """

    # Build the context block.
    # Numbering the sections helps the model reference them and keeps
    # it from blending chunks together into one confused mass of text.
    context_block = "\n\n---\n\n".join(
        f"[Section {i + 1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    # The system prompt is the most important part of RAG prompt engineering.
    # "Only use the context below" is the key instruction — without it,
    # the model will mix document content with its training knowledge,
    # which leads to confident-sounding wrong answers (hallucination).
    system_prompt = """You are a precise document assistant.
You are given sections extracted from a PDF document and a question about it.

Rules you must follow:
1. Answer ONLY using information from the provided sections.
2. If the answer is not in the sections, say: "I couldn't find that in the document."
3. Do not use your own training knowledge to fill gaps.
4. Quote the relevant section when it helps the user trust your answer.
5. Be concise and direct."""

    # The user-facing prompt combines the context and the question.
    # We put context first — LLMs pay more attention to content that appears
    # early in the prompt.
    user_prompt = f"""Here are the relevant sections from the document:

{context_block}

---

Question: {question}"""

    return ask(prompt=user_prompt, system=system_prompt)


def build_chat_prompt(question: str, chunks: list[str], history: list[dict]) -> str:
    """
    Build a prompt for multi-turn chat with conversation history.

    For the /chat endpoint (Phase 4c extension). The history is a list of
    past turns, e.g.:
        [
          {"role": "user",      "content": "What is this document about?"},
          {"role": "assistant", "content": "It is a report on climate change."},
        ]

    We prepend recent history so the model understands the conversation context.
    We cap history at the last 6 turns to avoid blowing up the context window.

    Args:
        question: The new question from the user.
        chunks:   Relevant chunks retrieved for this specific question.
        history:  Previous conversation turns (list of role/content dicts).

    Returns:
        A formatted prompt string including history and context.
    """
    # Only keep the last 6 turns (3 user + 3 assistant).
    # Older turns are less relevant and cost precious context window space.
    recent_history = history[-6:]

    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in recent_history
    )

    context_block = "\n\n---\n\n".join(
        f"[Section {i + 1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    return f"""Previous conversation:
{history_text}

Relevant sections from the document:
{context_block}

---

New question: {question}"""
