# app/settings/config.py  (Pydantic v2)
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Settings(BaseSettings):
    # ---------- App feature switches ----------
    # "off" = no sharing, "group" = allow sharing via KinGroups/PersonShare
    SHARE_PEOPLE_SCOPE: Literal["off", "group"] = Field(
        default="group",
        env=["SHARE_PEOPLE_SCOPE", "MEMORIES_SHARE_PEOPLE_SCOPE"],
    )
    # If true, new People entities auto-share to all of the owner’s groups
    SHARE_DEFAULT_OPT_IN: bool = Field(
        default=False,
        env=["SHARE_DEFAULT_OPT_IN", "MEMORIES_SHARE_DEFAULT_OPT_IN"],
    )

    # ---------- Whisper / transcription ----------
    WHISPER_MODEL: str = Field(default="small", env=["WHISPER_MODEL"])
    WHISPER_DEVICE: str = Field(default="auto", env=["WHISPER_DEVICE"])
    WHISPER_COMPUTE: str = Field(default="auto", env=["WHISPER_COMPUTE"])

    # ---------- LLM / API keys (add others as needed) ----------
    OPENAI_API_KEY: Optional[str] = Field(default=None, env=["OPENAI_API_KEY"])

    # ---------- pydantic-settings config ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # allow lower/upper env names
        extra="ignore",
    )

    # ---------- Email / SMTP ----------
    EMAIL_TRANSPORT: Literal["smtp", "dummy"] = Field(
        default="smtp",
        env=["EMAIL_TRANSPORT"],
    )
    SMTP_HOST: str = Field(default="smtp.gmail.com", env=["SMTP_HOST"])
    SMTP_PORT: int = Field(default=587, env=["SMTP_PORT"])
    SMTP_USERNAME: Optional[str] = Field(default=None, env=["SMTP_USERNAME"])
    SMTP_PASSWORD: Optional[str] = Field(default=None, env=["SMTP_PASSWORD"])
    SMTP_FROM: Optional[str] = Field(default=None, env=["SMTP_FROM"])
    SMTP_USE_TLS: bool = Field(default=True, env=["SMTP_USE_TLS"])
    SMTP_USE_SSL: bool = Field(default=False, env=["SMTP_USE_SSL"])

settings = Settings()
