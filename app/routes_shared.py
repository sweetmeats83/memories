from pathlib import Path as FSPath
from fastapi.templating import Jinja2Templates

from .media_pipeline import MediaPipeline, UserBucketsStrategy

BASE_DIR = FSPath(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory="templates")
PIPELINE = MediaPipeline(static_root=STATIC_DIR, path_strategy=UserBucketsStrategy())

__all__ = ["BASE_DIR", "STATIC_DIR", "templates", "PIPELINE"]
