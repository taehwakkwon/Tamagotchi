"""Data export/import for backup and migration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tamagotchi.growth.personality import PersonalityManager
from tamagotchi.growth.state import GrowthManager
from tamagotchi.memory.profile import ProfileManager
from tamagotchi.memory.store import MemoryStore


def export_data(store: MemoryStore) -> dict[str, Any]:
    """Export all Tamagotchi data as a JSON-serializable dict."""
    profile_mgr = ProfileManager(store)
    growth_mgr = GrowthManager(store)
    personality_mgr = PersonalityManager(store)

    profile = profile_mgr.load()
    growth_state = growth_mgr.get_state()
    personality = personality_mgr.load()
    episodes = store.get_recent_episodes(limit=9999)

    # Tasks
    tasks = [dict(r) for r in store._conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()]

    # Calendar events
    events = [dict(r) for r in store._conn.execute("SELECT * FROM calendar_events ORDER BY id").fetchall()]

    return {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "growth_state": {
            "xp": growth_state["xp"],
            "level": growth_state["level"],
            "total_conversations": growth_state["total_conversations"],
            "total_preferences": growth_state["total_preferences"],
            "total_recommendations_accepted": growth_state["total_recommendations_accepted"],
            "total_recommendations_rejected": growth_state["total_recommendations_rejected"],
        },
        "personality": {
            "warmth": personality.warmth,
            "humor": personality.humor,
            "curiosity": personality.curiosity,
            "formality": personality.formality,
            "energy": personality.energy,
            "empathy": personality.empathy,
        },
        "preferences": [
            {
                "category": p.category,
                "key": p.key,
                "value": p.value,
                "confidence": p.confidence,
                "source": p.source,
            }
            for p in profile.preferences
        ],
        "episodes": [
            {
                "summary": ep["summary"],
                "messages": ep["messages"],
                "preferences_extracted": ep.get("preferences_extracted"),
                "created_at": ep["created_at"],
            }
            for ep in episodes
        ],
        "tasks": tasks,
        "calendar_events": events,
    }


def import_data(store: MemoryStore, data: dict[str, Any]) -> dict[str, int]:
    """Import Tamagotchi data from a previously exported dict.

    Returns a summary of imported items.
    """
    counts = {
        "preferences": 0,
        "episodes": 0,
        "tasks": 0,
        "calendar_events": 0,
    }

    # Import growth state
    gs = data.get("growth_state", {})
    if gs:
        store.update_growth_state(
            xp=gs.get("xp", 0),
            level=gs.get("level", 1),
            total_conversations=gs.get("total_conversations", 0),
            total_preferences=gs.get("total_preferences", 0),
            total_recommendations_accepted=gs.get("total_recommendations_accepted", 0),
            total_recommendations_rejected=gs.get("total_recommendations_rejected", 0),
        )

    # Import personality
    pers = data.get("personality", {})
    if pers:
        personality_mgr = PersonalityManager(store)
        personality_mgr.load()  # ensure row exists
        from tamagotchi.growth.personality import PersonalityTraits
        traits = PersonalityTraits(**pers)
        personality_mgr._save(traits)

    # Import preferences
    for p in data.get("preferences", []):
        store.upsert_preference(
            category=p["category"],
            key=p["key"],
            value=p["value"],
            confidence=p.get("confidence", 1.0),
            source=p.get("source", "imported"),
        )
        counts["preferences"] += 1

    # Import episodes
    for ep in data.get("episodes", []):
        store.save_episode(
            summary=ep["summary"],
            messages=ep["messages"],
            preferences_extracted=ep.get("preferences_extracted"),
        )
        counts["episodes"] += 1

    # Import tasks
    now = datetime.now(timezone.utc).isoformat()
    for task in data.get("tasks", []):
        store._conn.execute(
            "INSERT INTO tasks (title, description, status, priority, due_date, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task["title"], task.get("description"), task.get("status", "pending"),
             task.get("priority", "medium"), task.get("due_date"), task.get("created_at", now), now),
        )
        counts["tasks"] += 1

    # Import calendar events
    for event in data.get("calendar_events", []):
        store._conn.execute(
            "INSERT INTO calendar_events (title, description, event_date, event_time, recurrence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event["title"], event.get("description"), event["event_date"],
             event.get("event_time"), event.get("recurrence"), event.get("created_at", now)),
        )
        counts["calendar_events"] += 1

    store._conn.commit()
    return counts


def export_to_file(store: MemoryStore, path: Path) -> None:
    """Export all data to a JSON file."""
    data = export_data(store)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def import_from_file(store: MemoryStore, path: Path) -> dict[str, int]:
    """Import data from a JSON file."""
    data = json.loads(path.read_text())
    return import_data(store, data)
