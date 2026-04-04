from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── OpenAI response shapes ─────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None   # "stop", "length", "content_filter", etc.


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class IntentShift(BaseModel):
    from_intent: Optional[str] = None
    to_intent: Optional[str] = None
    description: Optional[str] = None


class ProgressionSummary(BaseModel):
    trend: Optional[str] = None          # "escalating", "stable", "declining"
    dominant_flag: Optional[str] = None  # "SPIKE", "TREND", "SUSTAINED", etc.
    summary: Optional[str] = None


class TCAResult(BaseModel):
    """TCA analysis scores for a single conversation turn."""

    risk_level: Optional[int] = Field(default=None, ge=0, le=10)
    threat_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    obfuscation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    toxicity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    topic_shift_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    historical_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    intent_shift: Optional[IntentShift] = None
    progression_summary: Optional[ProgressionSummary] = None


# ── Combined response ──────────────────────────────────────────

class Response(BaseModel):
    """Full response: the model's completion plus TCA analysis."""

    completion: str
    analysis: Optional[TCAResult] = None