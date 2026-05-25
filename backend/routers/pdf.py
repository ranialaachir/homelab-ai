from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz
from backend.services.ollama_client import ask
from backend.services.pdf_store import upload as store_pdf, get as get_pdf

router = APIRouter()

class PDFSummaryResponse(BaseModel):
    pdf_id: str          # ← NEW: returned so frontend can use it for /ask later
    filename: str
    char_count: int
    summary: str


def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text.strip()


def chunk_text(text: str, max_chars: int = 3000) -> str:
    # Still the same workaround — we'll replace this in Step 2
    return text[:max_chars] if len(text) > max_chars else text


@router.post("/upload-pdf", response_model=PDFSummaryResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = "default",   # ← later the frontend will pass a real session UUID
    keep: bool = False,
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()

    # Save to disk and get a UUID back
    pdf_id = store_pdf(file_bytes, file.filename, session_id, keep)

    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")

    if not text:
        raise HTTPException(status_code=422, detail="PDF appears to be empty or scanned.")

    trimmed_text = chunk_text(text)
    summary = ask(
        prompt=trimmed_text,
        system="You are a document summarization assistant. Summarize the key points clearly and concisely in 3-5 sentences."
    )

    return PDFSummaryResponse(
        pdf_id=pdf_id,
        filename=file.filename,
        char_count=len(text),
        summary=summary
    )
