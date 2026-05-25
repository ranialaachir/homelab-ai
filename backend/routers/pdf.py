from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.services.pdf_store import upload as store_pdf
from backend.services.summarizer import summarize
from backend.services.ocr import extract_text   # ← replaces the old fitz-only function

router = APIRouter()

class PDFSummaryResponse(BaseModel):
    pdf_id: str
    filename: str
    char_count: int
    summary: str


@router.post("/upload-pdf", response_model=PDFSummaryResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = "default",
    keep: bool = False,
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    pdf_id = store_pdf(file_bytes, file.filename, session_id, keep)

    try:
        text = extract_text(file_bytes)   # ← now handles both text and scanned PDFs
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")

    if not text:
        raise HTTPException(status_code=422, detail="PDF appears to be empty.")

    summary = summarize(text)

    return PDFSummaryResponse(
        pdf_id=pdf_id,
        filename=file.filename,
        char_count=len(text),
        summary=summary
    )
