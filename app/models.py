from sqlalchemy import (
    Column, Integer, String, Boolean, ForeignKey, Text, DateTime, func,
    Table, UniqueConstraint, Index, JSON, Float
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
import sqlalchemy as sa
from .database import Base
from datetime import datetime, timedelta
from sqlalchemy.dialects.postgresql import JSONB as JSON
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy import Enum as SAEnum, UniqueConstraint
import enum


class WeeklyState(str, enum.Enum):
    not_sent = "not_sent"
    queued = "queued"
    sent = "sent"
    opened = "opened"
    clicked = "clicked"
    used = "used"
    recorded = "recorded"
    responded = "responded"
    skipped = "skipped"
    expired = "expired"

class WeeklyTokenStatus(str, enum.Enum):
    active = "active"
    opened = "opened"
    used = "used"
    expired = "expired"
# ---------------------------
# USER MODEL
# ---------------------------
class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, index=True)
    hashed_password = Column(String, nullable=False)
    weekly_prompts = relationship(
        "UserWeeklyPrompt",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    weekly_skips = relationship(
        "UserWeeklySkip",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    assigned_prompts = relationship(
        "UserPrompt",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    super_admin = Column(Boolean, default=False)
    must_change_password = Column(Boolean, default=False)
    notify_new_responses = Column(Boolean, default=False, nullable=False)
    relationship_status = Column(String, nullable=True)  # ✅ renamed
    goals = Column(String, nullable=True)
    # ---- Weekly pointers & state ----
    weekly_current_prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="SET NULL"), nullable=True)
    weekly_on_deck_prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="SET NULL"), nullable=True)

    weekly_state = Column(SAEnum(WeeklyState), default=WeeklyState.not_sent, nullable=False)

    # timeline fields (nullable):
    weekly_queued_at = Column(DateTime, nullable=True)
    weekly_sent_at = Column(DateTime, nullable=True)
    weekly_opened_at = Column(DateTime, nullable=True)
    weekly_clicked_at = Column(DateTime, nullable=True)
    weekly_used_at = Column(DateTime, nullable=True)
    weekly_completed_at = Column(DateTime, nullable=True)
    weekly_skipped_at = Column(DateTime, nullable=True)
    weekly_expires_at = Column(DateTime, nullable=True)

    weekly_email_provider_id = Column(String, nullable=True)  # optional for dedupe/debug

    weekly_current_prompt = relationship("Prompt", foreign_keys=[weekly_current_prompt_id])
    weekly_on_deck_prompt = relationship("Prompt", foreign_keys=[weekly_on_deck_prompt_id])

    responses = relationship("Response", back_populates="user", cascade="all, delete-orphan")
    notification_watchers = relationship(
        "ResponseNotificationTarget",
        foreign_keys="ResponseNotificationTarget.owner_user_id",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    notification_subscriptions = relationship(
        "ResponseNotificationTarget",
        foreign_keys="ResponseNotificationTarget.watcher_user_id",
        lazy="selectin",
    )

# ---------------------------
# TAGS
# ---------------------------
class Tag(Base):
    __tablename__ = "tag"

    id = Column(Integer, primary_key=True, index=True)  # index on PK is harmless
    name = Column(String(64), nullable=False)           # <- no unique=True here
    slug = Column(String(64), unique=True, nullable=False)  # <- unique only (no index=True)
    color = Column(String(16), nullable=True)

    def __repr__(self):
        return f"<Tag {self.name}>"


# Association tables
prompt_tags = Table(
    "prompt_tags",
    Base.metadata,
    Column("prompt_id", ForeignKey("prompt.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tag.id", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("prompt_id", "tag_id", name="uq_prompt_tag")
)

response_tags = Table(
    "response_tags",
    Base.metadata,
    Column("response_id", ForeignKey("response.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tag.id", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("response_id", "tag_id", name="uq_response_tag")
)




# ---------------------------
# PROMPTS
# ---------------------------
class Prompt(Base):
    __tablename__ = "prompt"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    chapter = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    media = relationship("PromptMedia", back_populates="prompt", cascade="all, delete-orphan")

    # NEW: many-to-many tags
    tags = relationship("Tag", secondary=prompt_tags, lazy="joined")


prompt_media_assignee = Table(
    "prompt_media_assignee",
    Base.metadata,
    Column("prompt_media_id", ForeignKey("prompt_media.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", ForeignKey("user.id", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("prompt_media_id", "user_id", name="uq_media_user_once"),
)


class PromptMedia(Base):
    __tablename__ = "prompt_media"

    id = Column(Integer, primary_key=True, index=True)
    prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="CASCADE"))
    assignee_user_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    file_path = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image, video, audio
    thumbnail_url = Column(String, nullable=True)
    mime_type = Column(String, nullable=True)
    duration_sec = Column(Integer, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    codec_audio = Column(String, nullable=True)
    codec_video = Column(String, nullable=True)
    wav_path = Column(String, nullable=True)  # relative to static/
    prompt = relationship("Prompt", back_populates="media")
    assignee_user = relationship("User")
    assignees = relationship("User", secondary=prompt_media_assignee, lazy="joined")

class UserPrompt(Base):
    __tablename__ = "user_prompts"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    prompt_id   = Column(Integer, ForeignKey("prompt.id", ondelete="CASCADE"), index=True, nullable=False)
    status      = Column(String, default="queued")     # queued|active|answered|skipped
    score       = Column(Float, default=0.0)
    assigned_at = Column(DateTime, default=func.now())
    last_sent_at= Column(DateTime)
    times_sent  = Column(Integer, default=0)

    # (optional but helpful)
    user   = relationship("User", back_populates="assigned_prompts", lazy="joined")
    prompt = relationship("Prompt", lazy="joined")

    __table_args__ = (
        UniqueConstraint("user_id","prompt_id", name="uq_user_prompt"),
    )
# ---------------------------
# RESPONSES
# ---------------------------
class Response(Base):
    __tablename__ = "response"

    id = Column(Integer, primary_key=True, index=True)
    prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    response_text = Column(Text, nullable=True)
    primary_media_url = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    transcription = Column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    user = relationship("User", back_populates="responses")
    prompt = relationship("Prompt")
    supporting_media = relationship("SupportingMedia", back_populates="response", cascade="all, delete-orphan")
    ai_polished = Column(Text, nullable=True)
    ai_polished_at = Column(DateTime(timezone=True), nullable=True)
    # NEW: many-to-many tags
    tags = relationship("Tag", secondary=response_tags, lazy="joined")
    primary_thumbnail_path = Column(String, nullable=True)
    primary_mime_type = Column(String, nullable=True)
    primary_duration_sec = Column(Integer, nullable=True)
    primary_sample_rate = Column(Integer, nullable=True)
    primary_channels = Column(Integer, nullable=True)
    primary_width = Column(Integer, nullable=True)
    primary_height = Column(Integer, nullable=True)
    primary_size_bytes = Column(Integer, nullable=True)
    primary_codec_audio = Column(String, nullable=True)
    primary_codec_video = Column(String, nullable=True)
    primary_wav_path = Column(String, nullable=True)  # relative to static/
    segments = relationship(
        "ResponseSegment",
        order_by="ResponseSegment.order_index.asc()",
        cascade="all, delete-orphan",
        back_populates="response",
        lazy="selectin",
    )


class ResponseNotificationTarget(Base):
    __tablename__ = "response_notification_target"

    id = Column(Integer, primary_key=True, index=True)
    owner_user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    watcher_user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", foreign_keys=[owner_user_id], back_populates="notification_watchers")
    watcher = relationship("User", foreign_keys=[watcher_user_id], back_populates="notification_subscriptions")

    __table_args__ = (
        UniqueConstraint("owner_user_id", "watcher_user_id", name="uq_notification_owner_watcher"),
    )
class SupportingMedia(Base):
    __tablename__ = "supporting_media"

    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(Integer, ForeignKey("response.id", ondelete="CASCADE"))
    file_path = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image, video, audio
    thumbnail_url = Column(String, nullable=True)
    mime_type = Column(String, nullable=True)
    duration_sec = Column(Integer, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    codec_audio = Column(String, nullable=True)
    codec_video = Column(String, nullable=True)
    wav_path = Column(String, nullable=True)
    response = relationship("Response", back_populates="supporting_media")

# ---------------------------
# VERSIONING / AUDIT
# ---------------------------
class ResponseVersion(Base):
    __tablename__ = "response_version"

    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("response.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    title = Column(Text, nullable=True)
    transcription = Column(Text, nullable=True)
    tags_json = Column(JSON, nullable=True)  # e.g., {"tags": ["life:adult", ...]}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    edited_by_admin_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)

class AdminEditLog(Base):
    __tablename__ = "admin_edit_log"

    id = Column(Integer, primary_key=True)
    admin_user_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True)
    target_user_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True)
    response_id = Column(Integer, ForeignKey("response.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(64), nullable=False)  # e.g., "edit_text", "set_tags", "add_media", "delete_media"
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ---------------------------
# PUBLIC SHARE TOKENS
# ---------------------------
class ResponseShare(Base):
    __tablename__ = "response_share"

    id = Column(Integer, primary_key=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    response_id = Column(Integer, ForeignKey("response.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    permanent = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked = Column(Boolean, default=False, nullable=False)


# ---------------------------
# INVITATIONS
# ---------------------------
class Invite(Base):
    __tablename__ = "invite"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    last_sent = Column(DateTime(timezone=True))
    sent_count = Column(Integer, default=1)

    # NEW fields required by your migration / flow
    token = Column(String(64), nullable=False, unique=True)           # one-time token
    expires_at = Column(DateTime(timezone=True), nullable=False)      # link TTL
    used_at = Column(DateTime(timezone=True), nullable=True)          # when consumed
    invited_by_user_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    invited_by = relationship("User", lazy="joined", foreign_keys=[invited_by_user_id])

class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), unique=True, index=True)
    display_name = Column(String(128), nullable=True)
    birth_year = Column(Integer, nullable=True)
    location = Column(String(128), nullable=True)
    relation_roles = Column(JSON, nullable=True)        # list[str]
    interests = Column(JSON, nullable=True)             # list[str] (tag slugs or names)
    accessibility_prefs = Column(JSON, nullable=True)   # { fontScale, highContrast, ... }
    consent_flags = Column(JSON, nullable=True)         # { researchOk, ... }
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    tag_weights   = Column(MutableDict.as_mutable(JSON), default=dict, nullable=False)
    privacy_prefs = Column(MutableDict.as_mutable(JSON), default=dict, nullable=False)
    
class PromptSuggestion(Base):
    __tablename__ = "prompt_suggestion"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="SET NULL"), nullable=True)
    source = Column(String(16), nullable=False)         # 'tag_match' | 'llm'
    title = Column(String(200), nullable=True)
    text = Column(Text, nullable=False)
    tags = Column(JSON, nullable=True)                  # list[str]
    status = Column(String(16), default="pending")      # pending|approved|rejected
    rationale_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ChapterMeta(Base):
    __tablename__ = "chapter_meta"

    id = Column(Integer, primary_key=True, index=True)
    # exact key used in Prompt.chapter
    name = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)

    # ordering & tint for UI
    order = Column(Integer, default=0)          # lower = earlier
    tint  = Column(String, nullable=True)       # hex, e.g. "#e5e7eb"

    # LLM-useful fields (optional but helpful):
    description   = Column(Text, nullable=True) # plain text summary of what this chapter is about
    keywords      = Column(Text, nullable=True) # comma- or JSON-like string (keep it simple)
    llm_guidance  = Column(Text, nullable=True) # “how to write prompts / themes”
# ---------------------------
# Weekly prompts
# ---------------------------


class WeeklyToken(Base):
    __tablename__ = "weekly_token"

    id = Column(Integer, primary_key=True)
    token = Column(String(64), unique=True, nullable=False, index=True)  # url-safe
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="CASCADE"), index=True, nullable=False)
    status = Column(SAEnum(WeeklyTokenStatus), default=WeeklyTokenStatus.active, nullable=False)

    sent_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    clicked_at = Column(DateTime, nullable=True)
    used_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # one active token per (user, prompt)
    __table_args__ = (UniqueConstraint("user_id", "prompt_id", name="uq_weeklytoken_user_prompt"),)

    user = relationship("User")
    prompt = relationship("Prompt")

class UserWeeklyPrompt(Base):
    __tablename__ = "user_weekly_prompts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    year = Column(Integer, index=True)
    week = Column(Integer, index=True)
    prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(16), default="active")  # active|answered|skipped
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="weekly_prompts")
    prompt = relationship("Prompt")  # optional but handy for eager loads

    __table_args__ = (UniqueConstraint("user_id", "year", "week", name="uq_user_week"),)

class UserWeeklySkip(Base):
    __tablename__ = "user_weekly_skips"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    year = Column(Integer, index=True, nullable=False)
    week = Column(Integer, index=True, nullable=False)
    prompt_id = Column(Integer, ForeignKey("prompt.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="weekly_skips")
    prompt = relationship("Prompt")
    __table_args__ = (
        UniqueConstraint("user_id", "year", "week", "prompt_id", name="uq_user_week_prompt_skip"),
    )


class ResponseSegment(Base):
    __tablename__ = "response_segments"

    id = Column(Integer, primary_key=True)
    response_id = Column(ForeignKey("response.id", ondelete="CASCADE"), index=True, nullable=False)
    order_index = Column(Integer, nullable=False, default=0)

    # media is stored via MediaPipeline under users/<user>/responses/<response_id>/supporting/<segment_id>/
    media_path = Column(String, nullable=True)    # relative to 'uploads/...'
    media_mime = Column(String, nullable=True)

    transcript = Column(Text, nullable=False, default="")  # filled after transcription finishes
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    response = relationship("Response", back_populates="segments")

# --- People & Relationships ---

class Person(Base):
    __tablename__ = "person"
    id = Column(Integer, primary_key=True)
    owner_user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    display_name = Column(String(128), nullable=False)
    given_name = Column(String(64))
    family_name = Column(String(64))
    birth_year = Column(Integer)
    death_year = Column(Integer)
    notes = Column(Text)
    photo_url = Column(String(256))
    meta = Column(JSON)  # consider JSONB on Postgres

    # Visibility: 'private' (default) or 'groups' (shared via PersonShare)
    visibility = Column(String(16), default="private")  # "private" | "groups"
    consent_source = Column(String(32), default="owner")  # owner|admin|inherited

class PersonAlias(Base):
    __tablename__ = "person_alias"
    id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.id", ondelete="CASCADE"), index=True, nullable=False)
    alias = Column(String(128), index=True, nullable=False)

class RelationshipEdge(Base):
    __tablename__ = "relationship_edge"
    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    src_id = Column(ForeignKey("person.id", ondelete="CASCADE"), index=True, nullable=False)
    dst_id = Column(ForeignKey("person.id", ondelete="CASCADE"), index=True, nullable=False)
    rel_type = Column(String(48), index=True, nullable=False)  # parent-of, spouse-of, sibling-of, friend-of, mentor-of ...
    confidence = Column(Float, default=0.8)
    notes = Column(Text)
    meta  = Column(JSON)  # [{"response_id":..,"span":"..."}]
    __table_args__ = (UniqueConstraint("user_id", "src_id", "dst_id", "rel_type", name="uq_rel_once"),)

class ResponsePerson(Base):
    __tablename__ = "response_person"
    id = Column(Integer, primary_key=True)
    response_id = Column(ForeignKey("response.id", ondelete="CASCADE"), index=True, nullable=False)
    person_id   = Column(ForeignKey("person.id", ondelete="CASCADE"), index=True, nullable=False)
    alias_used  = Column(String(128))
    start_char  = Column(Integer)
    end_char    = Column(Integer)
    confidence  = Column(Float, default=0.7)
    role_hint   = Column(String(48))  # grandmother, uncle, coach, neighbor ...

# --- Generic family/circle groups (multi-group support) ---

class KinGroup(Base):
    __tablename__ = "kin_group"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)          # e.g., "The Gomez Family", "Maternal Side"
    kind = Column(String(32), default="family")         # family|household|circle (optional)
    join_code = Column(String(16), unique=True)         # invite/join code
    created_by = Column(ForeignKey("user.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=datetime.utcnow)

class KinMembership(Base):
    __tablename__ = "kin_membership"
    id = Column(Integer, primary_key=True)
    group_id = Column(ForeignKey("kin_group.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id  = Column(ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    role     = Column(String(32), default="member")     # member|admin
    __table_args__ = (UniqueConstraint("group_id","user_id", name="uq_kin_group_user"),)

class PersonShare(Base):
    """
    Many-to-many visibility from Person -> KinGroup.
    If a Person has visibility='groups', members of any linked group can view them.
    """
    __tablename__ = "person_share"
    id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.id", ondelete="CASCADE"), index=True, nullable=False)
    group_id  = Column(ForeignKey("kin_group.id", ondelete="CASCADE"), index=True, nullable=False)
    shared_by_user_id = Column(ForeignKey("user.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("person_id","group_id", name="uq_person_group_once"),)

# --- ADD: chapter compilation model (below ChapterMeta) ---
class ChapterCompilation(Base):
    __tablename__ = "chapter_compilation"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    # chapter key must match Prompt.chapter (string key) and ChapterMeta.name
    chapter = Column(String, nullable=False, index=True)

    version = Column(Integer, nullable=False, default=1)              # monotonic per (user, chapter)
    status  = Column(String(16), nullable=False, default="draft")      # draft|published

    compiled_markdown = Column(Text, nullable=False)                   # the chapter body (markdown)
    gap_questions     = Column(JSON, nullable=True)                    # list[{"question": str, "why": str, "tags": list[str]}]
    used_blocks       = Column(JSON, nullable=True)                    # ordered mapping for traceability

    model_name = Column(String(64), nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "chapter", "version", name="uq_compilation_user_chapter_version"),
        Index("ix_compilation_user_chapter_created", "user_id", "chapter", "created_at"),
    )
