"""Chunked upload router — bypasses Cloudflare's per-request body limit.

Flow:
  1. POST /api/upload/init             → {upload_id, chunk_size_hint}
  2. POST /api/upload/{id}/chunk       → {ok, received, total}   (repeat per chunk)
  3. POST /api/upload/{id}/complete    → {ok, upload_id}

The assembled file lives at UPLOAD_DIR/_assembled_{upload_id}{ext} until consumed
by one of the upload-accepting routes.  Uploads older than 2 h are pruned lazily
on each init call.

Helper for routes.py:
  get_staged_file(upload_id, user_id) → meta dict | None
  release_staged_dir(upload_id)       → removes chunk dir (assembled file stays;
                                        process_*_async deletes it in its finally)
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path as FSPath

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.database import get_db  # noqa: F401 – kept for future use
from app.utils import require_authenticated_user

router = APIRouter()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_UPLOAD_DIR = os.path.join(_BASE_DIR, "static", "uploads")
_CHUNKS_DIR = os.path.join(_UPLOAD_DIR, "_chunks")

CHUNK_SIZE_HINT = 50 * 1024 * 1024   # 50 MB — safe under Cloudflare free (100 MB)
_MAX_AGE_SECS   = 7_200              # prune sessions older than 2 h


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _meta_path(upload_id: str) -> FSPath:
    return FSPath(_CHUNKS_DIR) / upload_id / "meta.json"


def _chunk_path(upload_id: str, index: int) -> FSPath:
    return FSPath(_CHUNKS_DIR) / upload_id / f"{index:05d}"


def _assembled_path(upload_id: str, ext: str) -> FSPath:
    return FSPath(_UPLOAD_DIR) / f"_assembled_{upload_id}{ext}"


def _load_meta(upload_id: str) -> dict | None:
    try:
        return json.loads(_meta_path(upload_id).read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_meta(upload_id: str, meta: dict) -> None:
    p = _meta_path(upload_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta), encoding="utf-8")


def _prune_old() -> None:
    """Remove chunk dirs and assembled files older than _MAX_AGE_SECS."""
    chunks_dir = FSPath(_CHUNKS_DIR)
    if not chunks_dir.exists():
        return
    cutoff = time.time() - _MAX_AGE_SECS
    for child in chunks_dir.iterdir():
        meta_p = child / "meta.json"
        try:
            if not meta_p.exists():
                shutil.rmtree(child, ignore_errors=True)
                continue
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("created_at", 0) < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                ap = _assembled_path(meta.get("upload_id", ""), meta.get("ext", ""))
                ap.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public helpers (imported by routes.py)
# ---------------------------------------------------------------------------

def get_staged_file(upload_id: str, user_id: int) -> dict | None:
    """Return meta dict if the upload is assembled and owned by user_id, else None."""
    meta = _load_meta(upload_id)
    if not meta:
        return None
    if meta.get("user_id") != user_id:
        return None
    if not meta.get("assembled"):
        return None
    if not FSPath(meta.get("assembled_path", "")).exists():
        return None
    return meta


def release_staged_dir(upload_id: str) -> None:
    """Remove the chunk directory (NOT the assembled file — caller owns that)."""
    shutil.rmtree(FSPath(_CHUNKS_DIR) / upload_id, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/upload/init")
async def upload_init(
    filename: str = Form(...),
    content_type: str = Form(...),
    total_chunks: int = Form(...),
    user=Depends(require_authenticated_user),
):
    """Start a chunked upload session.  Returns upload_id and suggested chunk size."""
    if not 1 <= total_chunks <= 1000:
        raise HTTPException(400, "total_chunks must be 1–1000")

    _prune_old()

    ext = FSPath(filename).suffix.lower() or ""
    upload_id = uuid.uuid4().hex
    _save_meta(upload_id, {
        "upload_id":    upload_id,
        "user_id":      user.id,
        "filename":     filename,
        "content_type": content_type,
        "ext":          ext,
        "total_chunks": total_chunks,
        "received":     [],
        "assembled":    False,
        "created_at":   time.time(),
    })
    return {"upload_id": upload_id, "chunk_size_hint": CHUNK_SIZE_HINT}


@router.post("/api/upload/{upload_id}/chunk")
async def upload_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    file: UploadFile = File(...),
    user=Depends(require_authenticated_user),
):
    """Store one chunk.  Chunks may arrive in any order."""
    meta = _load_meta(upload_id)
    if not meta:
        raise HTTPException(404, "Upload session not found or expired")
    if meta["user_id"] != user.id:
        raise HTTPException(403, "Not your upload")
    if meta.get("assembled"):
        raise HTTPException(400, "Upload already assembled")
    if not 0 <= chunk_index < meta["total_chunks"]:
        raise HTTPException(400, f"chunk_index out of range (0–{meta['total_chunks'] - 1})")

    dest = _chunk_path(upload_id, chunk_index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(await file.read())

    if chunk_index not in meta["received"]:
        meta["received"].append(chunk_index)
    _save_meta(upload_id, meta)

    return {"ok": True, "received": len(meta["received"]), "total": meta["total_chunks"]}


@router.post("/api/upload/{upload_id}/complete")
async def upload_complete(
    upload_id: str,
    user=Depends(require_authenticated_user),
):
    """Assemble all chunks into a single file.  Idempotent if already assembled."""
    meta = _load_meta(upload_id)
    if not meta:
        raise HTTPException(404, "Upload session not found or expired")
    if meta["user_id"] != user.id:
        raise HTTPException(403, "Not your upload")

    if meta.get("assembled"):
        return {"ok": True, "upload_id": upload_id}

    total    = meta["total_chunks"]
    received = set(meta["received"])
    missing  = [i for i in range(total) if i not in received]
    if missing:
        raise HTTPException(400, {"error": "missing_chunks", "missing": missing[:20]})

    assembled = _assembled_path(upload_id, meta["ext"])
    try:
        with open(assembled, "wb") as out:
            for i in range(total):
                with open(_chunk_path(upload_id, i), "rb") as c:
                    shutil.copyfileobj(c, out)
    except Exception as exc:
        assembled.unlink(missing_ok=True)
        raise HTTPException(500, f"Assembly failed: {exc}")

    # Remove individual chunk files to free space; keep meta for get_staged_file
    for i in range(total):
        _chunk_path(upload_id, i).unlink(missing_ok=True)

    meta["assembled"]      = True
    meta["assembled_path"] = str(assembled)
    _save_meta(upload_id, meta)

    return {"ok": True, "upload_id": upload_id}
