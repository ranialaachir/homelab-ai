from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.routers import summarize, pdf, scaffold
from backend.jobs.cleanup import delete_expired_pdfs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    scheduler = BackgroundScheduler()
    scheduler.add_job(delete_expired_pdfs, "interval", hours=1)
    scheduler.start()
    print("[scheduler] Cleanup job started — runs every hour.")
    
    yield  # app runs here
    
    # --- SHUTDOWN ---
    scheduler.shutdown()
    print("[scheduler] Cleanup job stopped.")

app = FastAPI(
    title="Homelab AI Assistant",
    description="Self-hosted AI API powered by Ollama + LLaMA 3",
    version="0.1.0",
    lifespan=lifespan,   # ← hook in the lifespan manager
)

app.include_router(summarize.router)
app.include_router(pdf.router)
app.include_router(scaffold.router)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return FileResponse("frontend/index.html")
