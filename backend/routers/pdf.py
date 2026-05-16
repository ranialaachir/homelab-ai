from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz
import io
from backend.services.ollama_client import ask

router = APIRouter()

class PDFSummaryResponse(BaseModel):
    filename: str
    char_count: int
    summary: str


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Opens a PDF from raw bytes and extracts all text content.
    fitz.open() can open from a stream (bytes) instead of a file path.
    """
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text.strip()


def chunk_text(text: str, max_chars: int = 3000) -> str:
    """
    LLaMA 3 has a context window of 8192 tokens (~6000 words / ~32000 chars).
    To be safe, we only send the first 3000 characters for now.
    In Phase 5 (RAG), we'll handle full documents properly by splitting
    them into chunks and processing each one.
    """
    return text[:max_chars] if len(text) > max_chars else text


@router.post("/upload-pdf", response_model=PDFSummaryResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, extracts its text, and returns a summary.
    The 'async' here is important — file I/O is awaited so the server
    doesn't block while reading the upload.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read the uploaded file into memory as raw bytes
    file_bytes = await file.read()

    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")

    if not text:
        raise HTTPException(status_code=422, detail="PDF appears to be empty or scanned (no extractable text).")

    # Trim to safe context window size
    trimmed_text = chunk_text(text)

    summary = ask(
        prompt=trimmed_text,
        system="You are a document summarization assistant. Summarize the key points of this document clearly and concisely in 3-5 sentences."
    )

    return PDFSummaryResponse(
        filename=file.filename,
        char_count=len(text),
        summary=summary
    )
