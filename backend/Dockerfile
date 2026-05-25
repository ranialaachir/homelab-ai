FROM python:3.11-slim
WORKDIR /app

# Install system dependencies:
# - tesseract-ocr: the OCR engine itself
# - tesseract-ocr-fra: French language pack
# - poppler-utils: needed by pdf2image to convert PDF pages to images
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
