# backend/routers/pdf.py
#
# HTTP layer for all PDF-related operations.
#
# Endpoints:
#   POST /upload-pdf        — upload a PDF, extract text, summarize, and index it
#   POST /pdf/{id}/ask      — ask a single question about a stored PDF (RAG)
#   POST /pdf/{id}/chat     — multi-turn conversation about a PDF (RAG + history)
#
# This file only handles HTTP concerns: parsing requests, returning responses,
# and translating service-layer errors into HTTP status codes.
# All business logic (chunking, embedding, RAG) lives in the services/ layer.

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.services.pdf_store import upload as store_pdf, get as get_pdf
from backend.services.summarizer import summarize
from backend.services.ocr import extract_text
from backend.services.chunker import chunk_text
from backend.services.embedder import embed, embed_batch
from backend.services.vector_store import store_chunks, query_chunks
from backend.services.rag import answer_from_chunks, build_chat_prompt
from backend.services.ollama_client import ask

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# Pydantic models give us automatic validation, serialization, and OpenAPI docs.
# ---------------------------------------------------------------------------

class PDFSummaryResponse(BaseModel):
    pdf_id: str
    filename: str
    char_count: int
    chunk_count: int    # new: tells the user how many chunks were indexed
    summary: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    pdf_id: str
    question: str
    answer: str
    chunks_used: int    # how many chunks were retrieved — useful for debugging


class ChatTurn(BaseModel):
    """A single turn in a conversation: who said it and what they said."""
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []   # defaults to empty list for the first turn


class ChatResponse(BaseModel):
    pdf_id: str
    answer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload-pdf", response_model=PDFSummaryResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = "default",
    keep: bool = False,
):
    """
    Upload a PDF file. This endpoint does four things:
      1. Saves the file to disk and records metadata in SQLite
      2. Extracts text (with OCR fallback for scanned PDFs)
      3. Generates a summary using map-reduce summarization
      4. Chunks the text and indexes it in ChromaDB for future /ask queries

    The indexing step (4) is what makes /ask possible. Without it, there's
    nothing in ChromaDB to search against.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()

    # --- Step 1: persist to disk + SQLite ---
    pdf_id = store_pdf(file_bytes, file.filename, session_id, keep)
    print(f"[pdf] uploaded: {file.filename} → pdf_id={pdf_id}")

    # --- Step 2: extract text ---
    try:
        text = extract_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")

    if not text:
        raise HTTPException(status_code=422, detail="PDF appears to be empty or unreadable.")

    # --- Step 3: summarize the whole document ---
    # summarize() handles both short and long documents automatically (map-reduce)
    summary = summarize(text)

    # --- Step 4: chunk + embed + index for RAG ---
    # We re-chunk here with the same settings as the summarizer uses internally.
    # These chunks go into ChromaDB so /ask can retrieve them later.
    print(f"[pdf] indexing {len(text)} chars for RAG...")
    chunks = chunk_text(text)

    # embed_batch sends each chunk to nomic-embed-text one at a time.
    # For a 50-page PDF (~100 chunks) this might take 20-60 seconds.
    # In a production system you'd do this in a background task — for now,
    # we wait synchronously. The user sees a longer upload time, but it
    # means /ask works immediately after upload.
    embeddings = embed_batch(chunks)

    # Store everything in ChromaDB, tagged with this pdf_id.
    store_chunks(pdf_id, chunks, embeddings)
    print(f"[pdf] RAG index ready: {len(chunks)} chunks stored")

    return PDFSummaryResponse(
        pdf_id=pdf_id,
        filename=file.filename,
        char_count=len(text),
        chunk_count=len(chunks),
        summary=summary,
    )


@router.post("/pdf/{pdf_id}/ask", response_model=AskResponse)
def ask_pdf(pdf_id: str, request: AskRequest):
    """
    Ask a single question about a previously uploaded PDF.

    This is the core RAG endpoint. The pipeline is:
      1. Verify the PDF exists in our database
      2. Embed the question (convert it to a vector)
      3. Query ChromaDB for the most relevant chunks
      4. Send chunks + question to LLaMA 3
      5. Return the answer

    The LLM never sees the whole PDF — only the chunks that are semantically
    close to the question. This keeps the prompt short and the answer focused.
    """
    # Verify the PDF exists before doing any expensive operations.
    pdf_meta = get_pdf(pdf_id)
    if not pdf_meta:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found with id={pdf_id}. It may have been deleted."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # --- Step 1: embed the question ---
    # The question vector must be produced by the SAME model as the chunks
    # (nomic-embed-text). Mixing embedding models breaks similarity search —
    # the vectors live in different "spaces" and distances become meaningless.
    print(f"[pdf/ask] embedding question for pdf_id={pdf_id}")
    question_vector = embed(request.question)

    # --- Step 2: retrieve relevant chunks from ChromaDB ---
    # n_results=5 is a good balance: enough context, small enough prompt.
    relevant_chunks = query_chunks(pdf_id, question_vector, n_results=5)

    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="This PDF has not been indexed yet, or its index was lost. "
                   "Please re-upload the file."
        )

    # --- Step 3: generate the answer ---
    answer = answer_from_chunks(request.question, relevant_chunks)

    return AskResponse(
        pdf_id=pdf_id,
        question=request.question,
        answer=answer,
        chunks_used=len(relevant_chunks),
    )


@router.post("/pdf/{pdf_id}/chat", response_model=ChatResponse)
def chat_pdf(pdf_id: str, request: ChatRequest):
    """
    Multi-turn conversation about a PDF.

    Same as /ask, but the caller sends along the conversation history so the
    model can understand follow-up questions like "tell me more about that"
    or "what did it say earlier about X?"

    The caller is responsible for maintaining the history list and sending
    it back on each request. This is stateless on the server side — we don't
    store conversation history. It's simpler, and it means no cleanup needed.

    Example request body:
        {
          "question": "Can you expand on the budget section?",
          "history": [
            {"role": "user",      "content": "What is this document about?"},
            {"role": "assistant", "content": "It is a Q3 financial report."}
          ]
        }
    """
    pdf_meta = get_pdf(pdf_id)
    if not pdf_meta:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found with id={pdf_id}."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Embed the new question and retrieve relevant chunks, same as /ask.
    question_vector = embed(request.question)
    relevant_chunks = query_chunks(pdf_id, question_vector, n_results=5)

    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="PDF index not found. Please re-upload.")

    # Convert Pydantic ChatTurn models → plain dicts for build_chat_prompt.
    history_dicts = [{"role": t.role, "content": t.content} for t in request.history]

    # Build the full prompt including history and context.
    full_prompt = build_chat_prompt(request.question, relevant_chunks, history_dicts)

    system_prompt = """You are a helpful document assistant having a conversation about a PDF.
Use only the provided document sections to answer. If continuity with prior conversation is relevant, use it.
If the answer is not in the document, say so clearly. Do not invent information."""

    answer = ask(prompt=full_prompt, system=system_prompt)

    return ChatResponse(pdf_id=pdf_id, answer=answer)
