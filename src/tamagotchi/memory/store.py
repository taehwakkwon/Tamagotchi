"""SQLite-based persistent storage for Tamagotchi memory system."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_DIR = Path.home() / ".tamagotchi"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "memory.db"


class MemoryStore:
    """SQLite backend for preferences, episodes, and growth state."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(category, key)
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                preferences_extracted TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                priority TEXT NOT NULL DEFAULT 'medium',
                due_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                event_date TEXT NOT NULL,
                event_time TEXT,
                recurrence TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS growth_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                xp INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 1,
                total_conversations INTEGER NOT NULL DEFAULT 0,
                total_preferences INTEGER NOT NULL DEFAULT 0,
                total_recommendations_accepted INTEGER NOT NULL DEFAULT 0,
                total_recommendations_rejected INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS personality (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                warmth REAL NOT NULL DEFAULT 0.5,
                humor REAL NOT NULL DEFAULT 0.5,
                curiosity REAL NOT NULL DEFAULT 0.5,
                formality REAL NOT NULL DEFAULT 0.5,
                energy REAL NOT NULL DEFAULT 0.5,
                empathy REAL NOT NULL DEFAULT 0.5,
                updated_at TEXT NOT NULL DEFAULT ''
            );
        """)
        self._conn.commit()

    # ── Preferences ──

    def upsert_preference(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> None:
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO preferences (category, key, value, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (category, key, value, confidence, source, now, now),
        )
        self._conn.commit()

    def get_preferences(self, category: str | None = None) -> list[dict[str, Any]]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM preferences WHERE category = ? ORDER BY confidence DESC",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM preferences ORDER BY category, confidence DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_preference(self, category: str, key: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM preferences WHERE category = ? AND key = ?", (category, key)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_preferences_by_category(self, category: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM preferences WHERE category = ?", (category,)
        )
        self._conn.commit()
        return cur.rowcount

    def count_preferences(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM preferences").fetchone()
        return row[0]

    # ── Episodes ──

    def save_episode(
        self,
        summary: str,
        messages: list[dict[str, str]],
        preferences_extracted: list[dict] | None = None,
    ) -> int:
        now = _now_iso()
        cur = self._conn.execute(
            """
            INSERT INTO episodes (summary, messages_json, preferences_extracted, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                summary,
                json.dumps(messages, ensure_ascii=False),
                json.dumps(preferences_extracted, ensure_ascii=False) if preferences_extracted else None,
                now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_recent_episodes(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["messages"] = json.loads(d.pop("messages_json"))
            if d["preferences_extracted"]:
                d["preferences_extracted"] = json.loads(d["preferences_extracted"])
            results.append(d)
        return results

    # ── Growth State ──

    def get_growth_state(self) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM growth_state WHERE id = 1").fetchone()
        if row is None:
            now = _now_iso()
            self._conn.execute(
                """
                INSERT INTO growth_state (id, xp, level, total_conversations, total_preferences,
                    total_recommendations_accepted, total_recommendations_rejected, created_at, updated_at)
                VALUES (1, 0, 1, 0, 0, 0, 0, ?, ?)
                """,
                (now, now),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM growth_state WHERE id = 1").fetchone()
        return dict(row)  # type: ignore[arg-type]

    def update_growth_state(self, **kwargs: Any) -> dict[str, Any]:
        state = self.get_growth_state()
        updates = []
        values = []
        for k, v in kwargs.items():
            if k in state and k not in ("id", "created_at"):
                updates.append(f"{k} = ?")
                values.append(v)
        if updates:
            updates.append("updated_at = ?")
            values.append(_now_iso())
            values.append(1)
            self._conn.execute(
                f"UPDATE growth_state SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            self._conn.commit()
        return self.get_growth_state()

    def close(self) -> None:
        self._conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
