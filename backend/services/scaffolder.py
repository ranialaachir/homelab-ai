import json
import zipfile
import io
import re
from backend.services.ollama_client import ask


def clean_json_response(raw: str) -> str:
    """
    LLMs produce nearly-valid JSON with two consistent issues:
    1. Literal newlines inside string values (illegal in JSON)
    2. Markdown code fences wrapping the JSON

    Fix 1 — strip markdown fences:
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw)
    raw = raw.strip()

    """
    Fix 2 — replace literal newlines inside JSON strings.

    The problem: LLMs write readme content like this:
        "readme": "# Title
        Some text"          ← actual newline, illegal in JSON

    When it should be:
        "readme": "# Title\nSome text"   ← escaped \n, valid JSON

    Strategy: walk through the string character by character.
    When we're inside a JSON string value (between unescaped quotes),
    replace any literal newline with \n and literal tab with \t.
    """
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '\\' and in_string:
            # escaped character — copy both chars as-is, don't toggle in_string
            result.append(ch)
            i += 1
            if i < len(raw):
                result.append(raw[i])
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
        elif ch == '\n' and in_string:
            result.append('\\n')   # replace literal newline with escaped \n
        elif ch == '\t' and in_string:
            result.append('\\t')   # replace literal tab with escaped \t
        elif ch == '\r' and in_string:
            pass                   # strip carriage returns entirely
        else:
            result.append(ch)
        i += 1

    return ''.join(result)


def generate_scaffold(description: str, project_type: str) -> dict:
    system = """You are a software architect assistant.
When given a project description, you respond with ONLY valid JSON — no explanation, no markdown, no preamble.

The JSON must follow this exact structure:
{
  "project_name": "slug-style-name",
  "files": [
    {"path": "relative/path/to/file.py", "content": "full file content here"}
  ],
  "readme": "full README.md content in markdown"
}

Rules:
- file paths are relative (never start with /)
- every file must have real, working content — not placeholders
- always include a Dockerfile and requirements.txt
- always include a .gitignore
- respond with JSON only, nothing else"""

    prompt = f"Project type: {project_type}\nDescription: {description}"
    raw = ask(prompt=prompt, system=system)
    cleaned = clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw response:\n{raw}")


def build_zip(scaffold: dict) -> bytes:
    buffer = io.BytesIO()
    project_name = scaffold.get("project_name", "project")

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in scaffold.get("files", []):
            zip_path = f"{project_name}/{file['path']}"
            zf.writestr(zip_path, file["content"])

        if "readme" in scaffold:
            zf.writestr(f"{project_name}/README.md", scaffold["readme"])

    buffer.seek(0)
    return buffer.read()
