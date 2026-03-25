"""Web search integration — provides search capability to the agent."""

from __future__ import annotations

import json
from typing import Any

import anthropic


class WebSearcher:
    """Simulates web search by using Claude's knowledge for information retrieval.

    In a production environment, this would integrate with a real search API
    (e.g., Google, Bing, Tavily). For now, Claude provides knowledge-based answers.
    """

    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search for information using Claude's knowledge."""
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""다음 검색 쿼리에 대해 {max_results}개의 관련 정보를 JSON 배열로 제공하세요.

검색 쿼리: {query}

각 항목:
- title: 제목
- snippet: 요약 (2-3문장)
- category: 카테고리

JSON 배열만 반환하세요.""",
            }],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return [{"title": "검색 결과", "snippet": response.content[0].text, "category": "general"}]
