"""Typed domain models shared by the pipeline, API, CLI, UI and workers.

Everything that crosses a process boundary (HTTP body, database row, JSON log)
is represented here so that validation happens in exactly one place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class StrEnum(str, Enum):  # noqa: UP042 - custom StrEnum ensures str() returns the bare value
    """String enum whose ``str()`` is the bare value (not ``Class.MEMBER``)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class Category(StrEnum):
    IMPORTANT = "Important"
    WORK = "Work"
    PERSONAL = "Personal"
    NEWSLETTER = "Newsletter"
    PROMOTION = "Promotion"
    SPAM = "Spam"

    @classmethod
    def coerce(cls, value: Any) -> Category:
        """Map arbitrary model output onto a valid category.

        Falls back to fuzzy matching on the raw string so that a model
        answering ``"work related"`` or ``"IMPORTANT!"`` still lands on a
        supported bucket instead of being dropped.
        """
        if isinstance(value, Category):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if text == member.value.lower():
                return member
        for member in cls:
            if member.value.lower() in text:
                return member
        aliases = {
            "urgent": cls.IMPORTANT,
            "critical": cls.IMPORTANT,
            "important work": cls.IMPORTANT,
            "personal mail": cls.PERSONAL,
            "social": cls.PERSONAL,
            "marketing": cls.PROMOTION,
            "ads": cls.PROMOTION,
            "offer": cls.PROMOTION,
            "bulk": cls.NEWSLETTER,
            "newsletter mail": cls.NEWSLETTER,
            "junk": cls.SPAM,
            "phishing": cls.SPAM,
        }
        for key, member in aliases.items():
            if key in text:
                return member
        return cls.WORK


class Tone(StrEnum):
    PROFESSIONAL = "Professional"
    FRIENDLY = "Friendly"
    CONCISE = "Concise"
    FORMAL = "Formal"
    APOLOGETIC = "Apologetic"

    @classmethod
    def coerce(cls, value: Any) -> Tone:
        if isinstance(value, Tone):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if text == member.value.lower():
                return member
        for member in cls:
            if member.value.lower() in text:
                return member
        return cls.PROFESSIONAL


class RiskLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Email(BaseModel):
    """A normalised email, independent of where it came from."""

    id: int | None = None
    created_at: str = Field(default_factory=utcnow)
    sender: str = ""
    sender_email: str = ""
    to: str = ""
    subject: str = ""
    body: str = ""
    date: str = ""
    message_id: str = ""
    thread_id: str = ""
    in_reply_to: str = ""
    references: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    attachment_meta: list[dict[str, Any]] = Field(default_factory=list)
    snippet: str = ""
    source: str = "unknown"

    @field_validator("snippet", mode="before")
    @classmethod
    def _snippet_default(cls, v: Any) -> Any:
        return v or ""

    def short(self, limit: int = 120) -> str:
        text = (self.snippet or self.body or "").strip().replace("\n", " ")
        return text[:limit]


class Analysis(BaseModel):
    """Structured triage output for a single email."""

    category: Category = Category.WORK
    priority: int = 3
    priority_reason: str = ""
    summary: str = ""
    action_items: list[str] = Field(default_factory=list)
    needs_reply: bool = True
    sentiment: str = "neutral"
    language: str = "en"
    security_flags: list[str] = Field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    latency_ms: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    email_id: int | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["category"] = str(self.category)
        return data


class Draft(BaseModel):
    """A generated reply awaiting human review."""

    id: int | None = None
    email_id: int | None = None
    body: str = ""
    tone: Tone = Tone.PROFESSIONAL
    instructions: str = ""
    warnings: list[str] = Field(default_factory=list)
    model: str = ""
    approved: bool = False
    sent_at: str | None = None
    created_at: str = ""


class Feedback(BaseModel):
    """Human rating of a draft; the raw material of the learning loop."""

    id: int | None = None
    draft_id: int | None = None
    email_id: int | None = None
    rating: int = 0
    correction: str = ""
    notes: str = ""
    created_at: str = ""


class TriageResult(BaseModel):
    """Everything the caller needs about one triaged email."""

    email: Email
    analysis: Analysis
    draft: Draft | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "email": self.email.model_dump(),
            "analysis": self.analysis.to_dict(),
            "draft": self.draft.model_dump() if self.draft else None,
        }


class Digest(BaseModel):
    """A prioritised summary of an inbox over a time window."""

    generated_at: str = ""
    email_count: int = 0
    urgent_count: int = 0
    needs_reply_count: int = 0
    top_emails: list[dict[str, Any]] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    by_category: dict[str, int] = Field(default_factory=dict)
    text: str = ""


class HealthStatus(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    database: str = "ok"
    llm_provider: str = "none"
    llm_model: str = ""
    mailbox: str = "not_configured"
    checked_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
