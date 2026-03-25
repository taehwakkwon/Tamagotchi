"""FastAPI routes for the Tamagotchi API."""

from __future__ import annotations

import uuid
from typing import Any

import anthropic
from fastapi import APIRouter, HTTPException

from tamagotchi.agent.prompts import build_system_prompt
from tamagotchi.config import get_chat_model
from tamagotchi.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForgetRequest,
    ForgetResponse,
    HistoryItem,
    HistoryResponse,
    PreferenceItem,
    ProfileResponse,
    StatusResponse,
)
from tamagotchi.growth.state import LEVELS, GrowthManager
from tamagotchi.learning.extractor import extract_preferences
from tamagotchi.learning.feedback import FeedbackTracker
from tamagotchi.memory.profile import ProfileManager
from tamagotchi.memory.store import MemoryStore

router = APIRouter(prefix="/api")

# In-memory session storage (maps session_id → message history)
_sessions: dict[str, list[dict[str, str]]] = {}


def _get_deps() -> tuple[MemoryStore, ProfileManager, GrowthManager]:
    store = MemoryStore()
    return store, ProfileManager(store), GrowthManager(store)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    store, profile_mgr, growth_mgr = _get_deps()
    try:
        session_id = req.session_id or str(uuid.uuid4())
        messages = _sessions.setdefault(session_id, [])
        messages.append({"role": "user", "content": req.message})

        profile = profile_mgr.load()
        system_prompt = build_system_prompt(profile, growth_mgr)

        client = anthropic.Anthropic()
        chat_model = get_chat_model(req.model)
        response = client.messages.create(
            model=chat_model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )

        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})

        # Extract preferences
        new_prefs = 0
        try:
            extracted = extract_preferences(client, messages)
            for pref in extracted:
                profile_mgr.add_preference(
                    category=pref["category"],
                    key=pref["key"],
                    value=pref["value"],
                    confidence=pref.get("confidence", 0.8),
                    source="implicit",
                )
            new_prefs = len(extracted)
        except Exception:
            pass

        # Update growth
        xp_gained = growth_mgr.add_xp_for_conversation(new_preferences=new_prefs)
        leveled_up = growth_mgr.check_level_up()

        # Save episode
        try:
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            summary = user_msgs[0][:80] if user_msgs else "대화"
            store.save_episode(summary=summary, messages=messages)
        except Exception:
            pass

        return ChatResponse(
            reply=reply,
            session_id=session_id,
            xp_gained=xp_gained,
            new_preferences=new_prefs,
            leveled_up=leveled_up,
        )
    finally:
        store.close()


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    store, _, growth_mgr = _get_deps()
    try:
        state = growth_mgr.get_state()
        level_info = LEVELS.get(state["level"], LEVELS[1])
        xp_to_next = growth_mgr.xp_to_next_level()
        total = state["total_recommendations_accepted"] + state["total_recommendations_rejected"]
        acceptance_rate = state["total_recommendations_accepted"] / total if total > 0 else None

        return StatusResponse(
            level=state["level"],
            level_name=level_info["name"],
            xp=state["xp"],
            xp_to_next=xp_to_next,
            total_conversations=state["total_conversations"],
            total_preferences=state["total_preferences"],
            recommendations_accepted=state["total_recommendations_accepted"],
            recommendations_rejected=state["total_recommendations_rejected"],
            acceptance_rate=acceptance_rate,
        )
    finally:
        store.close()


@router.get("/profile", response_model=ProfileResponse)
def profile() -> ProfileResponse:
    store, profile_mgr, _ = _get_deps()
    try:
        p = profile_mgr.load()
        return ProfileResponse(
            preferences=[
                PreferenceItem(
                    category=pref.category,
                    key=pref.key,
                    value=pref.value,
                    confidence=pref.confidence,
                    source=pref.source,
                )
                for pref in p.preferences
            ],
            categories=p.get_categories(),
            total_count=len(p.preferences),
        )
    finally:
        store.close()


@router.post("/profile/forget", response_model=ForgetResponse)
def forget(req: ForgetRequest) -> ForgetResponse:
    store, profile_mgr, _ = _get_deps()
    try:
        count = profile_mgr.forget(req.category, req.key)
        return ForgetResponse(deleted_count=count)
    finally:
        store.close()


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    store, _, growth_mgr = _get_deps()
    try:
        tracker = FeedbackTracker(store, growth_mgr)
        new_conf = tracker.record_feedback(req.category, req.key, req.accepted)
        xp = 50 if req.accepted else 5
        return FeedbackResponse(new_confidence=new_conf, xp_gained=xp)
    finally:
        store.close()


@router.get("/history", response_model=HistoryResponse)
def history(limit: int = 10) -> HistoryResponse:
    store = MemoryStore()
    try:
        episodes = store.get_recent_episodes(limit)
        return HistoryResponse(
            episodes=[
                HistoryItem(
                    id=ep["id"],
                    summary=ep["summary"],
                    created_at=ep["created_at"],
                    message_count=len(ep["messages"]),
                )
                for ep in episodes
            ]
        )
    finally:
        store.close()


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
