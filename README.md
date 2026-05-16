# 🧠 Homelab AI Assistant

A self-hosted AI assistant running entirely on a personal Linux server — no cloud APIs, no subscriptions, no data leaving the network.

Built as a hands-on ML/AI learning project. Designed to be useful to the whole household.

---

## What it does

| Feature | Endpoint | Description |
|---|---|---|
| Text summarization | `POST /summarize` | Summarize any text in three styles: concise, bullet points, or ELI5 |
| PDF summarization | `POST /upload-pdf` | Upload a PDF, get back a summary — works with any language |
| Health check | `GET /health` | Service liveness check for monitoring and Docker |

---

## Stack

| Layer | Technology |
|---|---|
| LLM runtime | [Ollama](https://ollama.com) + LLaMA 3 8B (local inference) |
| API framework | FastAPI + Uvicorn |
| PDF parsing | PyMuPDF (fitz) |
| Config | python-dotenv |
| Infrastructure | Ubuntu Linux homelab, i7 CPU |
| Version control | Git + GitHub |

---

## Architecture

```
Client (browser / curl / any device on LAN)
        │
        ▼
  FastAPI (port 9000)
  ├── POST /summarize      ← routers/summarize.py
  ├── POST /upload-pdf     ← routers/pdf.py
  └── GET  /health
        │
        ▼
  services/ollama_client.py
        │
        ▼
  Ollama HTTP API (port 11434)
        │
        ▼
  LLaMA 3 8B — running locally on bare metal
```

The API follows a **router / service separation**: routers handle HTTP concerns (validation, request/response), services handle business logic (LLM calls, PDF parsing). This keeps each layer independently testable.

---

## Setup

### Prerequisites

- Ubuntu Linux server (or any Linux machine)
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Install Ollama and pull the model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

### Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/homelab-ai.git
cd homelab-ai

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your values
```

### Environment variables (`.env`)

```env
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3
PORT=9000
```

### Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 9000
```

API is now accessible at `http://YOUR_SERVER_IP:9000`

Interactive docs (Swagger UI): `http://YOUR_SERVER_IP:9000/docs`

---

## Usage examples

### Summarize text

```bash
curl -X POST http://localhost:9000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here.", "style": "bullet_points"}'
```

Supported styles: `concise` (default) · `bullet_points` · `eli5`

### Summarize a PDF

```bash
curl -X POST http://localhost:9000/upload-pdf \
  -F "file=@/path/to/document.pdf"
```

Works with any language — tested with French and English documents.

---

## Project structure

```
homelab-ai/
├── .env.example
├── requirements.txt
└── backend/
    ├── main.py               ← FastAPI app + router registration
    ├── routers/
    │   ├── summarize.py      ← POST /summarize
    │   └── pdf.py            ← POST /upload-pdf
    └── services/
        └── ollama_client.py  ← Ollama HTTP client
```

---

## Roadmap

- [x] Phase 1 — Local LLM setup (Ollama + LLaMA 3) + Python client
- [x] Phase 2 — FastAPI backend with text and PDF summarization
- [ ] Phase 3 — Project scaffolding generator (natural language → boilerplate code + zip download)
- [ ] Phase 4 — Docker Compose deployment + Nginx reverse proxy + web UI
- [ ] Phase 5 — RAG pipeline (ChromaDB vector search over document collections), voice interface, fine-tuning experiments

---

## Key concepts demonstrated

- **Local LLM inference** — running a quantized 8B parameter model on consumer hardware via Ollama
- **REST API design** — FastAPI with Pydantic models for automatic validation and OpenAPI docs
- **PDF processing** — text extraction from arbitrary PDFs using PyMuPDF
- **Prompt engineering** — system prompts for consistent, structured model output
- **Separation of concerns** — router / service layering for maintainability
- **Secure config** — environment variable management with dotenv, secrets never committed
- **RAG foundations** — context window management and document chunking strategy

---

## Why this project

I built this to learn ML/AI engineering hands-on — not through tutorials alone, but by shipping something real that runs on my own hardware and is useful to people I know. Every line of code is something I understand and can explain.

The goal is to keep extending it through the roadmap: containerization, a proper RAG pipeline, and eventually a voice interface.

---

*Built with Python · FastAPI · Ollama · LLaMA 3 · Running on a homelab*
