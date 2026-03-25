"""Local calendar/event manager backed by SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tamagotchi.memory.store import MemoryStore


class CalendarManager:
    """CRUD operations for calendar events."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def add_event(
        self,
        title: str,
        event_date: str,
        event_time: str | None = None,
        description: str | None = None,
        recurrence: str | None = None,
    ) -> dict[str, Any]:
        now = _now_iso()
        cur = self.store._conn.execute(
            """
            INSERT INTO calendar_events (title, description, event_date, event_time, recurrence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, description, event_date, event_time, recurrence, now),
        )
        self.store._conn.commit()
        return self.get_event(cur.lastrowid)  # type: ignore[arg-type]

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        row = self.store._conn.execute(
            "SELECT * FROM calendar_events WHERE id = ?", (event_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_events(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM calendar_events WHERE 1=1"
        params: list[Any] = []
        if from_date:
            query += " AND event_date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND event_date <= ?"
            params.append(to_date)
        query += " ORDER BY event_date, event_time"
        return [dict(r) for r in self.store._conn.execute(query, params).fetchall()]

    def upcoming_events(self, limit: int = 10) -> list[dict[str, Any]]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self.store._conn.execute(
            "SELECT * FROM calendar_events WHERE event_date >= ? ORDER BY event_date, event_time LIMIT ?",
            (today, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_event(self, event_id: int) -> bool:
        cur = self.store._conn.execute(
            "DELETE FROM calendar_events WHERE id = ?", (event_id,)
        )
        self.store._conn.commit()
        return cur.rowcount > 0

    def update_event(self, event_id: int, **kwargs: Any) -> dict[str, Any] | None:
        event = self.get_event(event_id)
        if not event:
            return None
        allowed = {"title", "description", "event_date", "event_time", "recurrence"}
        updates = []
        values = []
        for k, v in kwargs.items():
            if k in allowed:
                updates.append(f"{k} = ?")
                values.append(v)
        if not updates:
            return event
        values.append(event_id)
        self.store._conn.execute(
            f"UPDATE calendar_events SET {', '.join(updates)} WHERE id = ?", values
        )
        self.store._conn.commit()
        return self.get_event(event_id)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
