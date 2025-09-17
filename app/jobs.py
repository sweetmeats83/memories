# app/jobs.py
from dataclasses import dataclass, field
from typing import Optional, Dict
import uuid, time, threading

@dataclass
class Job:
    id: str
    status: str = "queued"       # queued|running|done|error
    progress: float = 0.0
    error: Optional[str] = None
    segment_id: Optional[int] = None
    transcript_preview: Optional[str] = None

class JobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def new(self) -> Job:
        j = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[j.id] = j
        return j

    def get(self, jid: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(jid)

    def set(self, jid: str, **kw):
        with self._lock:
            j = self._jobs.get(jid)
            if not j: return
            for k,v in kw.items(): setattr(j, k, v)

JOBS = JobStore()
