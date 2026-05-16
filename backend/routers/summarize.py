from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.ollama_client import ask

router = APIRouter()

# Shame of the JSON body in the request
class SummarizeRequest(BaseModel):
    text: str
    style: str="concise" # optional

class SummarizeResponse(BaseModel):
    summary: str
    style: str

@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    """
    Accepts a block of text and returns a summary.
    Optional 'style' field: concise (default), bullet_points, eli5
    """
    system_prompts={
        "concise": "You are a summarization assistant. Summarize the given text in 2-3 concise sentences.",
        "bullet_points": "You are a summarization assistant. Summarize the given text as 4-6 bullet points.",
        "eli5": "You are a summarization assistant. Explain the given text simply, as if to a 12-year-old.",
    }
    system=system_prompts.get(request.style, system_prompts["concise"])
    summary=ask(prompt=request.text, system=system)
    return SummarizeResponse(summary=summary, style=request.style)

