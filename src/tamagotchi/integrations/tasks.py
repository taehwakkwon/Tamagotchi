"""Local task/todo manager backed by SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tamagotchi.memory.store import MemoryStore


class TaskManager:
    """CRUD operations for user tasks/todos."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def add_task(
        self,
        title: str,
        description: str | None = None,
        priority: str = "medium",
        due_date: str | None = None,
    ) -> dict[str, Any]:
        now = _now_iso()
        cur = self.store._conn.execute(
            """
            INSERT INTO tasks (title, description, status, priority, due_date, created_at, updated_at)
            VALUES (?, ?, 'pending', ?, ?, ?, ?)
            """,
            (title, description, priority, due_date, now, now),
        )
        self.store._conn.commit()
        return self.get_task(cur.lastrowid)  # type: ignore[arg-type]

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        row = self.store._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_tasks(
        self,
        status: str | None = None,
        priority: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if priority:
            query += " AND priority = ?"
            params.append(priority)
        query += " ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 END, created_at DESC"
        return [dict(r) for r in self.store._conn.execute(query, params).fetchall()]

    def complete_task(self, task_id: int) -> bool:
        return self._update_status(task_id, "completed")

    def delete_task(self, task_id: int) -> bool:
        cur = self.store._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.store._conn.commit()
        return cur.rowcount > 0

    def update_task(self, task_id: int, **kwargs: Any) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if not task:
            return None
        allowed = {"title", "description", "priority", "due_date", "status"}
        updates = []
        values = []
        for k, v in kwargs.items():
            if k in allowed:
                updates.append(f"{k} = ?")
                values.append(v)
        if not updates:
            return task
        updates.append("updated_at = ?")
        values.append(_now_iso())
        values.append(task_id)
        self.store._conn.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", values
        )
        self.store._conn.commit()
        return self.get_task(task_id)

    def _update_status(self, task_id: int, status: str) -> bool:
        cur = self.store._conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now_iso(), task_id),
        )
        self.store._conn.commit()
        return cur.rowcount > 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
