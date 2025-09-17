from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# =========================
# USER SCHEMAS
# =========================
class UserBase(BaseModel):
    email: EmailStr
    username: str
    is_superuser: bool = False
    must_change_password: bool = False
    relationship: Optional[str] = None
    goals: Optional[str] = None

class UserRead(UserBase):
    id: int

    class Config:
        from_attributes = True

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str]
    is_superuser: Optional[bool]
    must_change_password: Optional[bool]
    relationship: Optional[str]
    goals: Optional[str]


# =========================
# PROMPT MEDIA SCHEMAS
# =========================
class PromptMediaRead(BaseModel):
    id: int
    file_path: str
    media_type: Optional[str]
    thumbnail_url: Optional[str] = None
    mime_type: Optional[str] = None
    duration_sec: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None

    class Config:
        from_attributes = True


# =========================
# PROMPT SCHEMAS
# =========================
class PromptBase(BaseModel):
    text: str
    chapter: Optional[str] = None

class PromptRead(PromptBase):
    id: int
    created_at: datetime
    media: List[PromptMediaRead] = []

    class Config:
        from_attributes = True

class PromptCreate(PromptBase):
    pass


# =========================
# RESPONSE SCHEMAS
# =========================
class ResponseBase(BaseModel):
    text: Optional[str] = None
    transcription: Optional[str] = None

class ResponseRead(ResponseBase):
    id: int
    prompt_id: int
    user_id: int
    created_at: datetime

    # primary media standard fields
    primary_media_url: Optional[str] = None
    primary_thumbnail_path: Optional[str] = None
    primary_mime_type: Optional[str] = None
    primary_duration_sec: Optional[int] = None
    primary_sample_rate: Optional[int] = None
    primary_channels: Optional[int] = None
    primary_width: Optional[int] = None
    primary_height: Optional[int] = None
    primary_size_bytes: Optional[int] = None

    class Config:
        from_attributes = True


class SupportingMediaRead(BaseModel):
    id: int
    file_path: str
    media_type: Optional[str]
    thumbnail_url: Optional[str] = None
    mime_type: Optional[str] = None
    duration_sec: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None

    class Config:
        from_attributes = True

class ResponseCreate(ResponseBase):
    prompt_id: int
    primary_media: Optional[str] = None
    supporting_media: Optional[List[str]] = None


# =========================
# INVITE SCHEMAS
# =========================
class InviteBase(BaseModel):
    email: EmailStr
    invited_by: Optional[int] = None

class InviteRead(InviteBase):
    id: int
    token: str
    accepted: bool
    created_at: datetime

    class Config:
        from_attributes = True

class InviteCreate(InviteBase):
    pass

class UserProfileRead(BaseModel):
    user_id: int
    display_name: str | None = None
    birth_year: int | None = None
    location: str | None = None
    relation_roles: list[str] | None = None
    interests: list[str] | None = None
    accessibility_prefs: dict | None = None
    consent_flags: dict | None = None
    bio: str | None = None
    class Config: from_attributes = True

class UserProfileUpdate(BaseModel):
    display_name: str | None = None
    birth_year: int | None = None
    location: str | None = None
    relation_roles: list[str] | None = None
    interests: list[str] | None = None
    accessibility_prefs: dict | None = None
    consent_flags: dict | None = None
    bio: str | None = None

class PromptSuggestionRead(BaseModel):
    id: int
    user_id: int
    prompt_id: int | None
    source: str
    title: str | None
    text: str
    tags: list[str] | None
    status: str
    rationale_json: dict | None
    created_at: datetime
    class Config: from_attributes = True
# --- NEW: Response Segment Schemas ---------------------------------------
class ResponseSegmentRead(BaseModel):
    id: int
    order_index: int
    media_path: Optional[str] = None
    media_mime: Optional[str] = None
    transcript: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReorderSegmentsRequest(BaseModel):
    order: List[int]

# --- ADD: Chapter compile DTOs ---
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GapQuestion(BaseModel):
    question: str
    why: Optional[str] = None
    tags: Optional[List[str]] = None

class UsedBlock(BaseModel):
    prompt_id: int
    response_id: int
    title: Optional[str] = None
    used_excerpt: Optional[str] = None

class ChapterCompilationDTO(BaseModel):
    id: Optional[int] = None
    user_id: int
    chapter: str
    version: int
    status: str
    compiled_markdown: str
    gap_questions: List[GapQuestion] = []
    used_blocks: List[UsedBlock] = []
    model_name: Optional[str] = None
    token_stats: Dict[str, Optional[int]] = {}
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ChapterStatusDTO(BaseModel):
    chapter: str
    display_name: str
    ready: bool
    missing_prompts: List[Dict[str, Any]] = []  # [{id, text}]
    latest_compilation: Optional[ChapterCompilationDTO] = None
