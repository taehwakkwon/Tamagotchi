"""Shopping recommendation integration."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from tamagotchi.memory.profile import ProfileManager
from tamagotchi.memory.store import MemoryStore


class ShoppingAssistant:
    """Provides shopping recommendations based on user preferences.

    In a production environment, this would integrate with real shopping APIs
    (e.g., Coupang, Naver Shopping). For now, uses Claude for recommendation generation.
    """

    def __init__(self, store: MemoryStore, client: anthropic.Anthropic | None = None):
        self.store = store
        self.profile_mgr = ProfileManager(store)
        self.client = client or anthropic.Anthropic()

    def recommend(self, query: str) -> list[dict[str, Any]]:
        """Generate shopping recommendations based on query and user profile."""
        profile = self.profile_mgr.load()
        profile_text = profile.to_prompt()

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""사용자 프로필을 참고하여 쇼핑 추천을 JSON 배열로 제공하세요.

## 사용자 프로필
{profile_text}

## 쇼핑 요청
{query}

각 항목 (3~5개):
- name: 상품명
- category: 카테고리
- price_range: 예상 가격대
- reason: 추천 이유 (사용자 취향 기반)

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

        return []

    def price_compare(self, product: str) -> list[dict[str, Any]]:
        """Generate price comparison information for a product."""
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""다음 상품의 가격 비교 정보를 JSON 배열로 제공하세요.

상품: {product}

각 항목:
- store: 판매처
- price: 가격 (원)
- note: 비고 (배송, 할인 등)

JSON 배열만 반환하세요. 실제 가격 정보가 아닌 예상치입니다.""",
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

        return []
