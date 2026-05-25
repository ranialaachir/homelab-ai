# backend/services/vector_store.py
#
# Responsible for storing and querying text chunks in ChromaDB.
#
# ChromaDB concepts you need to know:
#
#   Client     — your connection to the ChromaDB server
#   Collection — like a table. We create one per PDF (named by pdf_id).
#   Document   — the raw text of a chunk
#   Embedding  — the vector representation of that chunk
#   ID         — a unique string identifying each chunk within a collection
#   Metadata   — any extra info you want to store alongside a chunk
#                (we store chunk_index so we can sort results later)
#
# ChromaDB stores all three together: (id, document, embedding, metadata).
# When you query, you give it a vector and it returns the closest matches.

import chromadb
import os

# CHROMA_HOST is set in docker-compose.yml as an environment variable.
# Inside the Docker network, the ChromaDB container is reachable by its
# service name "chromadb" on port 8000.
# Outside Docker (running locally), it would be "localhost".
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))


def _get_client() -> chromadb.HttpClient:
    """
    Create and return a ChromaDB HTTP client.

    We use HttpClient (not the in-memory Client) because ChromaDB runs as
    a separate Docker container. HttpClient connects over the network.

    The underscore prefix (_) signals this is an internal helper — callers
    outside this module should use the public functions below, not this directly.
    """
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _collection_name(pdf_id: str) -> str:
    """
    Build a consistent collection name from a pdf_id.

    ChromaDB collection names must be 3-63 characters, start and end with
    alphanumeric characters, and contain only alphanumerics, underscores,
    or hyphens. UUIDs (with hyphens) satisfy all these rules.

    Example: "pdf_3f2a1b4c-..." → "pdf_3f2a1b4c-..."
    We prefix with "pdf_" to be explicit and avoid any future naming collisions.
    """
    return f"pdf_{pdf_id}"


def store_chunks(pdf_id: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    """
    Store text chunks and their embeddings in ChromaDB.

    This is called once per PDF upload, after chunking and embedding.
    Each chunk gets:
      - a unique ID (e.g. "chunk_0", "chunk_1", ...)
      - its raw text stored as the "document"
      - its vector stored as the "embedding"
      - its index stored as metadata (useful for ordering results)

    Args:
        pdf_id:     The UUID assigned to this PDF by pdf_store.py
        chunks:     The list of text chunks from chunker.py
        embeddings: The list of vectors from embedder.py, same order as chunks
    """
    client = _get_client()

    # get_or_create_collection: safe to call even if the collection already
    # exists (e.g. if the server restarted). It just returns the existing one.
    collection = client.get_or_create_collection(name=_collection_name(pdf_id))

    # ChromaDB's add() expects parallel lists of equal length:
    #   ids[i], documents[i], embeddings[i], metadatas[i] all describe chunk i
    ids        = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas  = [{"chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,       # raw text — ChromaDB stores this for retrieval
        embeddings=embeddings,  # vectors — ChromaDB uses these for similarity search
        metadatas=metadatas,    # extra info we can filter on later
    )

    print(f"[vector_store] stored {len(chunks)} chunks for pdf_id={pdf_id}")


def query_chunks(pdf_id: str, question_embedding: list[float], n_results: int = 5) -> list[str]:
    """
    Find the most semantically relevant chunks for a given question.

    This is the heart of RAG. We give ChromaDB the question's vector and ask:
    "which of the stored chunks are most similar to this?"

    ChromaDB computes cosine similarity between the question vector and every
    stored chunk vector, then returns the top n_results matches.

    Cosine similarity: a measure of how "aligned" two vectors are.
    Score of 1.0 = identical meaning. Score of 0.0 = completely unrelated.
    We don't need to think about the math — ChromaDB handles it.

    Args:
        pdf_id:             The UUID of the PDF to search within
        question_embedding: The vector of the user's question (from embedder.py)
        n_results:          How many chunks to return (5 is a good default —
                            enough context without overloading the LLM prompt)

    Returns:
        A list of raw text chunks, ordered by relevance (most relevant first).
        Returns an empty list if the collection doesn't exist yet.
    """
    client = _get_client()

    try:
        collection = client.get_collection(name=_collection_name(pdf_id))
    except Exception:
        # Collection doesn't exist — PDF was never indexed, or was deleted.
        # Return empty rather than crashing so the router can give a clean error.
        print(f"[vector_store] no collection found for pdf_id={pdf_id}")
        return []

    results = collection.query(
        query_embeddings=[question_embedding],  # list of vectors (we send one)
        n_results=n_results,
        include=["documents"],  # we only need the text, not the embeddings back
    )

    # ChromaDB returns: { "documents": [[chunk1, chunk2, ...]], ... }
    # The outer list is one entry per query_embedding we sent (we sent one).
    # So results["documents"][0] is the list of matching chunks.
    chunks = results["documents"][0]

    print(f"[vector_store] retrieved {len(chunks)} relevant chunks for query")
    return chunks


def delete_collection(pdf_id: str) -> None:
    """
    Delete the ChromaDB collection for a given PDF.

    Called when a PDF is deleted from pdf_store — we clean up the vectors too.
    If the collection doesn't exist (already deleted, or never indexed),
    we do nothing rather than raising an error.

    Args:
        pdf_id: The UUID of the PDF whose collection should be deleted.
    """
    client = _get_client()

    try:
        client.delete_collection(name=_collection_name(pdf_id))
        print(f"[vector_store] deleted collection for pdf_id={pdf_id}")
    except Exception:
        # Collection didn't exist — that's fine, nothing to clean up.
        pass
