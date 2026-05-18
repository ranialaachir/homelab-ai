from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.routers import summarize, pdf, scaffold

app = FastAPI(
    title="Homelab AI Assistant",
    description="Self-hosted AI API powered by Ollama + LLaMA 3",
    version="0.1.0",
)

app.include_router(summarize.router)
app.include_router(pdf.router)
app.include_router(scaffold.router)

# Serve static frontend files from /frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/health")
def health():
    return {"status": "ok"}

# Catch-all: serve index.html for the root
@app.get("/")
def root():
    return FileResponse("frontend/index.html")
