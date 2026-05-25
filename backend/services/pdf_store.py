import sqlite3
import uuid
import os
from pathlib import Path
from datetime import datetime, timedelta

# Where files live on disk. Path() is cleaner than string concatenation.
# __file__ = this file's path. .parent = the directory it's in.
# We go up two levels to reach the project root, then into data/uploads/.
BASE_DIR = Path(__file__).parent.parent.parent  # project root
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DB_PATH    = BASE_DIR / "data" / "pdf_store.db"

def _init_db():
    """
    Create the SQLite database and table if they don't exist yet.
    The underscore prefix (_) is a Python convention meaning 
    "internal function, not meant to be called from outside this module."
    
    'IF NOT EXISTS' makes this safe to call every time the app starts —
    it won't destroy data if the table already exists.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id           TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            uploaded_at  TEXT NOT NULL,
            keep         INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

# Call it once when the module is imported.
_init_db()


def upload(file_bytes: bytes, filename: str, session_id: str, keep: bool = False) -> str:
    """
    Save a PDF to disk and record its metadata in SQLite.
    Returns the UUID assigned to this file.
    """
    pdf_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{pdf_id}.pdf"

    # Write raw bytes to disk
    save_path.write_bytes(file_bytes)

    # Insert metadata row
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO pdfs (id, filename, session_id, uploaded_at, keep) VALUES (?, ?, ?, ?, ?)",
        (pdf_id, filename, session_id, datetime.utcnow().isoformat(), int(keep))
    )
    conn.commit()
    conn.close()

    return pdf_id


def get(pdf_id: str) -> dict | None:
    """
    Return metadata + file path for a given PDF id.
    Returns None if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, filename, session_id, uploaded_at, keep FROM pdfs WHERE id = ?",
        (pdf_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id":          row[0],
        "filename":    row[1],
        "session_id":  row[2],
        "uploaded_at": row[3],
        "keep":        bool(row[4]),
        "path":        UPLOAD_DIR / f"{row[0]}.pdf",
    }


def delete(pdf_id: str) -> bool:
    """
    Delete a PDF from disk and remove its metadata from SQLite.
    Returns True if it existed, False if not found.
    """
    meta = get(pdf_id)
    if not meta:
        return False

    path = meta["path"]
    if path.exists():
        path.unlink()  # unlink = delete a file (Path API)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
    conn.commit()
    conn.close()
    return True


def list_expired(max_age_hours: int = 1) -> list[str]:
    """
    Return IDs of all PDFs that:
    - have keep=False (temporary)
    - were uploaded more than max_age_hours ago
    
    Used by the cleanup job.
    """
    cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id FROM pdfs WHERE keep = 0 AND uploaded_at < ?",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]
