"""Tool definitions and execution for Claude tool use."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from tamagotchi.integrations.calendar import CalendarManager
from tamagotchi.integrations.content import ContentRecommender
from tamagotchi.integrations.shopping import ShoppingAssistant
from tamagotchi.integrations.tasks import TaskManager
from tamagotchi.integrations.web import WebSearcher
from tamagotchi.memory.profile import ProfileManager
from tamagotchi.memory.semantic import SemanticMemory
from tamagotchi.memory.store import MemoryStore

# Claude tool definitions (JSON Schema format)
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "manage_tasks",
        "description": "할일/태스크를 관리합니다. 추가, 조회, 완료, 삭제 가능.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "complete", "delete"],
                    "description": "수행할 작업",
                },
                "title": {
                    "type": "string",
                    "description": "태스크 제목 (add 시 필수)",
                },
                "description": {
                    "type": "string",
                    "description": "태스크 설명 (선택)",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "우선순위 (기본: medium)",
                },
                "due_date": {
                    "type": "string",
                    "description": "마감일 (YYYY-MM-DD 형식)",
                },
                "task_id": {
                    "type": "integer",
                    "description": "태스크 ID (complete, delete 시 필수)",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["pending", "completed"],
                    "description": "목록 필터 (list 시 선택)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "manage_calendar",
        "description": "일정/이벤트를 관리합니다. 추가, 조회, 삭제 가능.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "upcoming", "delete"],
                    "description": "수행할 작업",
                },
                "title": {
                    "type": "string",
                    "description": "이벤트 제목 (add 시 필수)",
                },
                "event_date": {
                    "type": "string",
                    "description": "이벤트 날짜 (YYYY-MM-DD, add 시 필수)",
                },
                "event_time": {
                    "type": "string",
                    "description": "이벤트 시간 (HH:MM, 선택)",
                },
                "description": {
                    "type": "string",
                    "description": "이벤트 설명 (선택)",
                },
                "event_id": {
                    "type": "integer",
                    "description": "이벤트 ID (delete 시 필수)",
                },
                "from_date": {
                    "type": "string",
                    "description": "조회 시작일 (list 시 선택)",
                },
                "to_date": {
                    "type": "string",
                    "description": "조회 종료일 (list 시 선택)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "recommend_content",
        "description": "사용자 취향에 맞는 컨텐츠(영화, 음악, 기사, 유튜브 등)를 추천합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "추천 요청 내용 (예: '오늘 볼 영화', '새로운 음악', '재미있는 유튜브')",
                },
            },
            "required": ["request"],
        },
    },
    {
        "name": "search_web",
        "description": "웹에서 정보를 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리",
                },
                "max_results": {
                    "type": "integer",
                    "description": "최대 결과 수 (기본: 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "shopping_recommend",
        "description": "사용자 취향에 맞는 쇼핑 상품을 추천하거나 가격을 비교합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["recommend", "price_compare"],
                    "description": "추천 또는 가격 비교",
                },
                "query": {
                    "type": "string",
                    "description": "쇼핑 요청 또는 상품명",
                },
            },
            "required": ["action", "query"],
        },
    },
    {
        "name": "remember_preference",
        "description": "사용자가 명시적으로 말한 선호도를 저장합니다. 사용자가 좋아하거나 싫어하는 것을 언급할 때 사용하세요.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "카테고리 (food, shopping, entertainment, lifestyle, content, schedule 등)",
                },
                "key": {
                    "type": "string",
                    "description": "구체적 항목 (예: 커피, 나이키, SF_영화)",
                },
                "value": {
                    "type": "string",
                    "description": "선호도 설명 (예: 좋아함, 싫어함, 매주 이용)",
                },
            },
            "required": ["category", "key", "value"],
        },
    },
    {
        "name": "search_memories",
        "description": "과거 대화나 선호도에서 관련 기억을 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리",
                },
            },
            "required": ["query"],
        },
    },
]


class ToolExecutor:
    """Executes tool calls from Claude API responses."""

    def __init__(
        self,
        store: MemoryStore,
        client: anthropic.Anthropic,
        semantic: SemanticMemory | None = None,
    ):
        self.store = store
        self.client = client
        self.task_mgr = TaskManager(store)
        self.calendar_mgr = CalendarManager(store)
        self.content_rec = ContentRecommender(store, client)
        self.web_searcher = WebSearcher(client)
        self.shopping = ShoppingAssistant(store, client)
        self.profile_mgr = ProfileManager(store)
        self.semantic = semantic

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        handlers = {
            "manage_tasks": self._handle_tasks,
            "manage_calendar": self._handle_calendar,
            "recommend_content": self._handle_content,
            "search_web": self._handle_web_search,
            "shopping_recommend": self._handle_shopping,
            "remember_preference": self._handle_remember,
            "search_memories": self._handle_search_memories,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)

        try:
            result = handler(tool_input)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _handle_tasks(self, inp: dict) -> Any:
        action = inp["action"]
        if action == "add":
            return self.task_mgr.add_task(
                title=inp["title"],
                description=inp.get("description"),
                priority=inp.get("priority", "medium"),
                due_date=inp.get("due_date"),
            )
        elif action == "list":
            return self.task_mgr.list_tasks(
                status=inp.get("status_filter"),
                priority=inp.get("priority"),
            )
        elif action == "complete":
            success = self.task_mgr.complete_task(inp["task_id"])
            return {"success": success}
        elif action == "delete":
            success = self.task_mgr.delete_task(inp["task_id"])
            return {"success": success}
        return {"error": f"Unknown action: {action}"}

    def _handle_calendar(self, inp: dict) -> Any:
        action = inp["action"]
        if action == "add":
            return self.calendar_mgr.add_event(
                title=inp["title"],
                event_date=inp["event_date"],
                event_time=inp.get("event_time"),
                description=inp.get("description"),
            )
        elif action == "list":
            return self.calendar_mgr.list_events(
                from_date=inp.get("from_date"),
                to_date=inp.get("to_date"),
            )
        elif action == "upcoming":
            return self.calendar_mgr.upcoming_events()
        elif action == "delete":
            success = self.calendar_mgr.delete_event(inp["event_id"])
            return {"success": success}
        return {"error": f"Unknown action: {action}"}

    def _handle_content(self, inp: dict) -> Any:
        return self.content_rec.recommend(inp["request"])

    def _handle_web_search(self, inp: dict) -> Any:
        return self.web_searcher.search(
            inp["query"],
            max_results=inp.get("max_results", 5),
        )

    def _handle_shopping(self, inp: dict) -> Any:
        if inp["action"] == "recommend":
            return self.shopping.recommend(inp["query"])
        elif inp["action"] == "price_compare":
            return self.shopping.price_compare(inp["query"])
        return {"error": f"Unknown action: {inp['action']}"}

    def _handle_remember(self, inp: dict) -> Any:
        self.profile_mgr.add_preference(
            category=inp["category"],
            key=inp["key"],
            value=inp["value"],
            confidence=1.0,
            source="explicit",
        )
        return {"status": "saved", "category": inp["category"], "key": inp["key"]}

    def _handle_search_memories(self, inp: dict) -> Any:
        results = []
        if self.semantic:
            episodes = self.semantic.search_episodes(inp["query"], n_results=5)
            for ep in episodes:
                summary = ep["document"].split("\n")[0] if ep["document"] else ""
                results.append({
                    "type": "episode",
                    "summary": summary,
                    "relevance": 1.0 - (ep["distance"] or 0.0),
                })

        prefs = self.store.get_preferences()
        query_lower = inp["query"].lower()
        for p in prefs:
            if query_lower in p["key"].lower() or query_lower in p["value"].lower() or query_lower in p["category"].lower():
                results.append({
                    "type": "preference",
                    "category": p["category"],
                    "key": p["key"],
                    "value": p["value"],
                    "confidence": p["confidence"],
                })

        return results if results else [{"message": "관련 기억을 찾지 못했습니다."}]
