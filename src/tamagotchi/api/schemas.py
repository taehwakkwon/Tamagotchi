"""Request/Response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    xp_gained: int = 0
    new_preferences: int = 0
    leveled_up: bool = False


class StatusResponse(BaseModel):
    level: int
    level_name: str
    xp: int
    xp_to_next: int | None
    total_conversations: int
    total_preferences: int
    recommendations_accepted: int
    recommendations_rejected: int
    acceptance_rate: float | None


class PreferenceItem(BaseModel):
    category: str
    key: str
    value: str
    confidence: float
    source: str


class ProfileResponse(BaseModel):
    preferences: list[PreferenceItem]
    categories: list[str]
    total_count: int


class ForgetRequest(BaseModel):
    category: str
    key: str | None = None


class ForgetResponse(BaseModel):
    deleted_count: int


class FeedbackRequest(BaseModel):
    category: str
    key: str
    accepted: bool


class FeedbackResponse(BaseModel):
    new_confidence: float | None
    xp_gained: int


class HistoryItem(BaseModel):
    id: int
    summary: str
    created_at: str
    message_count: int


class HistoryResponse(BaseModel):
    episodes: list[HistoryItem]
