"""Tests for task manager, calendar manager, and tool executor."""

import json

import pytest

from tamagotchi.integrations.tasks import TaskManager
from tamagotchi.integrations.calendar import CalendarManager
from tamagotchi.agent.tools import ToolExecutor, TOOL_DEFINITIONS
from tamagotchi.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def task_mgr(store):
    return TaskManager(store)


@pytest.fixture
def cal_mgr(store):
    return CalendarManager(store)


class TestTaskManager:
    def test_add_task(self, task_mgr):
        task = task_mgr.add_task("장보기", description="우유, 빵", priority="high")
        assert task["title"] == "장보기"
        assert task["status"] == "pending"
        assert task["priority"] == "high"

    def test_list_tasks(self, task_mgr):
        task_mgr.add_task("할일 1", priority="high")
        task_mgr.add_task("할일 2", priority="low")
        task_mgr.add_task("할일 3", priority="medium")

        all_tasks = task_mgr.list_tasks()
        assert len(all_tasks) == 3
        # Should be ordered by priority: high, medium, low
        assert all_tasks[0]["priority"] == "high"
        assert all_tasks[1]["priority"] == "medium"
        assert all_tasks[2]["priority"] == "low"

    def test_list_tasks_filtered(self, task_mgr):
        task_mgr.add_task("할일 1")
        t2 = task_mgr.add_task("할일 2")
        task_mgr.complete_task(t2["id"])

        pending = task_mgr.list_tasks(status="pending")
        assert len(pending) == 1
        completed = task_mgr.list_tasks(status="completed")
        assert len(completed) == 1

    def test_complete_task(self, task_mgr):
        task = task_mgr.add_task("테스트")
        assert task_mgr.complete_task(task["id"]) is True
        updated = task_mgr.get_task(task["id"])
        assert updated["status"] == "completed"

    def test_delete_task(self, task_mgr):
        task = task_mgr.add_task("삭제 테스트")
        assert task_mgr.delete_task(task["id"]) is True
        assert task_mgr.get_task(task["id"]) is None

    def test_update_task(self, task_mgr):
        task = task_mgr.add_task("원래 제목")
        updated = task_mgr.update_task(task["id"], title="변경된 제목", priority="high")
        assert updated["title"] == "변경된 제목"
        assert updated["priority"] == "high"

    def test_add_with_due_date(self, task_mgr):
        task = task_mgr.add_task("마감 있는 할일", due_date="2026-04-01")
        assert task["due_date"] == "2026-04-01"


class TestCalendarManager:
    def test_add_event(self, cal_mgr):
        event = cal_mgr.add_event("회의", "2026-04-01", event_time="14:00")
        assert event["title"] == "회의"
        assert event["event_date"] == "2026-04-01"
        assert event["event_time"] == "14:00"

    def test_list_events(self, cal_mgr):
        cal_mgr.add_event("이벤트 1", "2026-04-01")
        cal_mgr.add_event("이벤트 2", "2026-04-05")
        cal_mgr.add_event("이벤트 3", "2026-03-20")

        all_events = cal_mgr.list_events()
        assert len(all_events) == 3

        # Filter by date range
        filtered = cal_mgr.list_events(from_date="2026-04-01", to_date="2026-04-05")
        assert len(filtered) == 2

    def test_upcoming_events(self, cal_mgr):
        cal_mgr.add_event("과거 이벤트", "2020-01-01")
        cal_mgr.add_event("미래 이벤트", "2099-12-31")

        upcoming = cal_mgr.upcoming_events()
        assert len(upcoming) == 1
        assert upcoming[0]["title"] == "미래 이벤트"

    def test_delete_event(self, cal_mgr):
        event = cal_mgr.add_event("삭제 테스트", "2026-04-01")
        assert cal_mgr.delete_event(event["id"]) is True
        assert cal_mgr.get_event(event["id"]) is None

    def test_update_event(self, cal_mgr):
        event = cal_mgr.add_event("원래 제목", "2026-04-01")
        updated = cal_mgr.update_event(event["id"], title="변경된 제목", event_time="15:00")
        assert updated["title"] == "변경된 제목"
        assert updated["event_time"] == "15:00"


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_names_unique(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))


class TestToolExecutor:
    """Test tool executor with local tools (no API calls)."""

    def test_task_add_via_executor(self, store):
        executor = ToolExecutor(store, client=None)  # type: ignore
        result = executor.execute("manage_tasks", {
            "action": "add",
            "title": "테스트 할일",
            "priority": "high",
        })
        data = json.loads(result)
        assert data["title"] == "테스트 할일"

    def test_task_list_via_executor(self, store):
        executor = ToolExecutor(store, client=None)  # type: ignore
        executor.execute("manage_tasks", {"action": "add", "title": "할일 1"})
        executor.execute("manage_tasks", {"action": "add", "title": "할일 2"})

        result = executor.execute("manage_tasks", {"action": "list"})
        data = json.loads(result)
        assert len(data) == 2

    def test_calendar_add_via_executor(self, store):
        executor = ToolExecutor(store, client=None)  # type: ignore
        result = executor.execute("manage_calendar", {
            "action": "add",
            "title": "미팅",
            "event_date": "2026-04-01",
            "event_time": "10:00",
        })
        data = json.loads(result)
        assert data["title"] == "미팅"

    def test_remember_preference_via_executor(self, store):
        executor = ToolExecutor(store, client=None)  # type: ignore
        result = executor.execute("remember_preference", {
            "category": "food",
            "key": "커피",
            "value": "좋아함",
        })
        data = json.loads(result)
        assert data["status"] == "saved"

        # Verify it was actually saved
        prefs = store.get_preferences("food")
        assert len(prefs) == 1
        assert prefs[0]["key"] == "커피"
        assert prefs[0]["confidence"] == 1.0

    def test_search_memories_preference_match(self, store):
        store.upsert_preference("food", "커피", "좋아함")
        executor = ToolExecutor(store, client=None)  # type: ignore

        result = executor.execute("search_memories", {"query": "커피"})
        data = json.loads(result)
        pref_results = [r for r in data if r.get("type") == "preference"]
        assert len(pref_results) == 1

    def test_unknown_tool(self, store):
        executor = ToolExecutor(store, client=None)  # type: ignore
        result = executor.execute("nonexistent_tool", {})
        data = json.loads(result)
        assert "error" in data
