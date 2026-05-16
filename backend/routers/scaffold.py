from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
from backend.services.scaffolder import generate_scaffold, build_zip

router = APIRouter()

SUPPORTED_TYPES = ["python-api", "node-backend", "fullstack-react"]


class ScaffoldRequest(BaseModel):
    description: str
    project_type: str = "python-api"


@router.post("/scaffold")
def scaffold(request: ScaffoldRequest):
    """
    Accepts a project description and returns a zip file containing
    a complete generated project scaffold.
    """
    if request.project_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported project type. Choose from: {SUPPORTED_TYPES}"
        )

    try:
        scaffold_data = generate_scaffold(
            description=request.description,
            project_type=request.project_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    zip_bytes = build_zip(scaffold_data)
    project_name = scaffold_data.get("project_name", "project")

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={project_name}.zip"}
    )


@router.post("/scaffold/debug")
def scaffold_debug(request: ScaffoldRequest):
    """Temporary — returns raw LLM output for inspection."""
    from backend.services.ollama_client import ask
    system = """You are a software architect assistant.
Respond with ONLY valid JSON, no explanation, no markdown, no preamble.
Structure:
{
  "project_name": "slug-style-name",
  "files": [{"path": "relative/path", "content": "file content"}],
  "readme": "README.md content in markdown"
}
Rules: relative paths only, real working content, include Dockerfile, requirements.txt, .gitignore."""

    raw = ask(
        prompt=f"Project type: {request.project_type}\nDescription: {request.description}",
        system=system
    )
    return {"raw": raw}
