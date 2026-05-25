import fitz                  # PyMuPDF — already installed
import pytesseract           # OCR wrapper
from PIL import Image        # image handling
import io

def extract_text_fitz(file_bytes: bytes) -> str:
    """Fast path — works for PDFs with real embedded text."""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text.strip()


def extract_text_ocr(file_bytes: bytes) -> str:
    """
    Slow path — for scanned PDFs (images of pages).
    
    How it works:
    1. fitz renders each PDF page as a pixel image (a "pixmap")
    2. We convert that to a PIL Image (format pytesseract understands)
    3. pytesseract runs Tesseract on it and returns the text
    """
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in pdf:
        # Render page at 2x zoom — higher resolution = better OCR accuracy
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert pixmap bytes → PIL Image
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run OCR — lang="fra+eng" means try French and English
        # Change this to whatever languages your PDFs use
        page_text = pytesseract.image_to_string(img, lang="fra+eng")
        text += page_text + "\n"
    
    return text.strip()


def extract_text(file_bytes: bytes) -> str:
    """
    Smart entry point: try fitz first (fast), fall back to OCR if needed.
    The threshold of 100 chars catches both empty PDFs and scanned ones
    that might have a tiny bit of embedded metadata text.
    """
    text = extract_text_fitz(file_bytes)
    
    if len(text) < 100:
        print("[ocr] fitz returned little/no text — switching to OCR")
        text = extract_text_ocr(file_bytes)
    else:
        print(f"[ocr] fitz extracted {len(text)} chars — no OCR needed")
    
    return text
