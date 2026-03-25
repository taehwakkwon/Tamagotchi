"""Content recommendation engine — suggests content based on learned preferences."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from tamagotchi.memory.profile import ProfileManager
from tamagotchi.memory.store import MemoryStore

RECOMMENDATION_PROMPT = """\
사용자의 프로필을 기반으로 오늘 추천할 컨텐츠를 생성하세요.

## 사용자 프로필
{profile_text}

## 추천 요청
{request}

다음 JSON 배열로 응답하세요. 각 추천 항목:
- type: 컨텐츠 유형 (movie, music, article, youtube, podcast, book, game 등)
- title: 제목
- reason: 추천 이유 (사용자 프로필 기반, 1-2문장)
- category: 관련 선호도 카테고리

3~5개 추천하세요. 사용자의 취향에 맞는 구체적이고 실제 존재하는 컨텐츠를 추천하세요.
"""


class ContentRecommender:
    """Generates personalized content recommendations based on user profile."""

    def __init__(self, store: MemoryStore, client: anthropic.Anthropic | None = None):
        self.store = store
        self.profile_mgr = ProfileManager(store)
        self.client = client or anthropic.Anthropic()

    def recommend(self, request: str = "오늘의 추천 컨텐츠") -> list[dict[str, Any]]:
        """Generate content recommendations based on user profile."""
        profile = self.profile_mgr.load()
        profile_text = profile.to_prompt()

        if not profile.preferences:
            return [{
                "type": "tip",
                "title": "아직 취향을 파악하지 못했어요",
                "reason": "대화를 더 나누면 맞춤 추천이 가능해요!",
                "category": "general",
            }]

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": RECOMMENDATION_PROMPT.format(
                    profile_text=profile_text,
                    request=request,
                ),
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

    def recommend_for_prompt(self, max_items: int = 3) -> str:
        """Generate a prompt-ready recommendation block for proactive suggestions."""
        profile = self.profile_mgr.load()
        if not profile.preferences:
            return ""

        # Build a quick summary of top interests for proactive recommendation
        categories = profile.get_categories()
        top_prefs = sorted(profile.preferences, key=lambda p: -p.confidence)[:10]

        lines = ["## 선제적 추천을 위한 사용자 관심사"]
        for p in top_prefs:
            lines.append(f"- {p.category}/{p.key}: {p.value}")

        lines.append("")
        lines.append("위 관심사를 바탕으로 대화 중 자연스럽게 관련 컨텐츠, 정보, 상품을 추천하세요.")

        return "\n".join(lines)
