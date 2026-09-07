from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator


def _require_aware(v: datetime) -> datetime:
    if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
        raise ValueError("timestamps must be timezone-aware")
    return v.astimezone(timezone.utc)


class RawEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)
    subject_id: str
    source: str
    source_id: str
    ts: datetime
    payload: dict[str, Any]
    ingested_at: datetime

    _aware = field_validator("ts", "ingested_at")(_require_aware)

    def key(self) -> tuple[str, str, str]:
        return (self.subject_id, self.source, self.source_id)


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)
    subject_id: str
    source: str
    source_id: str
    thread_id: str
    ts: datetime
    author_id: str
    is_subject: bool
    text: str
    assisted: bool          # CC1 — no default, by design
    corpus_version: str | None = None

    _aware = field_validator("ts")(_require_aware)
