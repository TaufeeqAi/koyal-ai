from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptTurn(BaseModel):
    """
    Normalized transcript event shared by backend broadcaster and frontend UI.

    speaker:
        "caller" | "agent"
    text:
        Human-readable utterance text.
    language:
        BCP-47 language code, e.g. "hi-IN", "en-IN", "hi-IN+en-IN".
    timestamp:
        ISO-8601 timestamp string in UTC.
    confidence:
        Optional STT confidence for caller turns.
    is_escalation:
        True if the turn is part of an escalation flow.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    speaker: Literal["caller", "agent"]
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(min_length=2, max_length=32)
    timestamp: str = Field(min_length=1, max_length=64)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_escalation: bool = False