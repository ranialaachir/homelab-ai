from fastapi import FastAPI
from backend.routers import summarize, pdf, scaffold

app = FastAPI(
    title="Homelab AI Assistant",
    description="Self-hosted AI API powered by Ollama + LLaMA 3",
    version="0.1.0",
)

app.include_router(summarize.router)
app.include_router(pdf.router)
app.include_router(scaffold.router)

@app.get("/health")
def health():
    return {"status": "ok"}
