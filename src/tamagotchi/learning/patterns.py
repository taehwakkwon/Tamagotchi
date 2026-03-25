"""Behavior pattern recognition — detects recurring habits from episode history."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from tamagotchi.memory.store import MemoryStore


class PatternAnalyzer:
    """Analyzes conversation and preference history to detect behavioral patterns."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def analyze_all(self) -> list[dict[str, Any]]:
        """Run all pattern analyses and return detected patterns."""
        patterns = []
        patterns.extend(self._analyze_time_patterns())
        patterns.extend(self._analyze_category_trends())
        patterns.extend(self._analyze_preference_shifts())
        return patterns

    def _analyze_time_patterns(self) -> list[dict[str, Any]]:
        """Detect when the user is most active (time of day, day of week)."""
        episodes = self.store.get_recent_episodes(limit=100)
        if len(episodes) < 5:
            return []

        hour_counts: Counter[int] = Counter()
        weekday_counts: Counter[int] = Counter()

        for ep in episodes:
            try:
                dt = datetime.fromisoformat(ep["created_at"])
                hour_counts[dt.hour] += 1
                weekday_counts[dt.weekday()] += 1
            except (ValueError, KeyError):
                continue

        patterns = []

        # Peak hours (top 3 if significant)
        if hour_counts:
            peak_hours = hour_counts.most_common(3)
            total = sum(hour_counts.values())
            significant = [(h, c) for h, c in peak_hours if c / total > 0.2]
            if significant:
                hours_str = ", ".join(f"{h}시" for h, _ in significant)
                patterns.append({
                    "type": "time_pattern",
                    "key": "활동_시간대",
                    "value": f"주로 {hours_str}에 대화함",
                    "confidence": min(0.9, len(episodes) / 50),
                    "data": {h: c for h, c in peak_hours},
                })

        # Peak days
        day_names = ["월", "화", "수", "목", "금", "토", "일"]
        if weekday_counts:
            peak_days = weekday_counts.most_common(3)
            total = sum(weekday_counts.values())
            significant = [(d, c) for d, c in peak_days if c / total > 0.2]
            if significant:
                days_str = ", ".join(f"{day_names[d]}요일" for d, _ in significant)
                patterns.append({
                    "type": "time_pattern",
                    "key": "활동_요일",
                    "value": f"주로 {days_str}에 대화함",
                    "confidence": min(0.9, len(episodes) / 50),
                    "data": {day_names[d]: c for d, c in peak_days},
                })

        return patterns

    def _analyze_category_trends(self) -> list[dict[str, Any]]:
        """Detect which preference categories are growing or dominant."""
        prefs = self.store.get_preferences()
        if len(prefs) < 3:
            return []

        category_counts: Counter[str] = Counter()
        for p in prefs:
            category_counts[p["category"]] += 1

        patterns = []
        total = sum(category_counts.values())

        for cat, count in category_counts.most_common(3):
            ratio = count / total
            if ratio > 0.25:
                patterns.append({
                    "type": "category_trend",
                    "key": f"관심_분야_{cat}",
                    "value": f"{cat} 관련 선호도가 전체의 {ratio:.0%}를 차지 (총 {count}개)",
                    "confidence": min(0.85, ratio + 0.2),
                    "data": {"category": cat, "count": count, "ratio": ratio},
                })

        return patterns

    def _analyze_preference_shifts(self) -> list[dict[str, Any]]:
        """Detect preferences that have changed over time (confidence shifts)."""
        episodes = self.store.get_recent_episodes(limit=50)
        patterns = []

        # Look at preferences extracted from recent vs older episodes
        recent_prefs: Counter[str] = Counter()
        older_prefs: Counter[str] = Counter()
        midpoint = len(episodes) // 2

        for i, ep in enumerate(episodes):
            extracted = ep.get("preferences_extracted")
            if not extracted:
                continue
            for p in extracted:
                cat = p.get("category", "unknown")
                if i < midpoint:
                    recent_prefs[cat] += 1
                else:
                    older_prefs[cat] += 1

        # Find categories with significant growth
        all_cats = set(recent_prefs.keys()) | set(older_prefs.keys())
        for cat in all_cats:
            recent = recent_prefs.get(cat, 0)
            older = older_prefs.get(cat, 0)
            if recent > older + 2:
                patterns.append({
                    "type": "preference_shift",
                    "key": f"관심_증가_{cat}",
                    "value": f"최근 {cat}에 대한 관심이 증가하는 추세",
                    "confidence": 0.6,
                    "data": {"category": cat, "recent": recent, "older": older},
                })

        return patterns

    def get_patterns_for_prompt(self) -> str:
        """Generate a prompt-ready text block of detected patterns."""
        patterns = self.analyze_all()
        if not patterns:
            return ""

        lines = ["## 감지된 행동 패턴"]
        for p in patterns:
            lines.append(f"- {p['value']}")
        return "\n".join(lines)
