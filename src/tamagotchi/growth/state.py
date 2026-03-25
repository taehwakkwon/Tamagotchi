"""Tamagotchi growth system — levels, XP, and evolution."""

from __future__ import annotations

from tamagotchi.memory.store import MemoryStore

# XP rewards
XP_CONVERSATION = 10
XP_NEW_PREFERENCE = 25
XP_RECOMMENDATION_ACCEPTED = 50
XP_RECOMMENDATION_REJECTED = 5

LEVELS: dict[int, dict] = {
    1: {
        "name": "알 (Egg)",
        "min_xp": 0,
        "min_conversations": 0,
        "min_preferences": 0,
        "description": "막 태어난 다마고치. 기본 대화만 가능합니다.",
    },
    2: {
        "name": "아기 (Baby)",
        "min_xp": 100,
        "min_conversations": 10,
        "min_preferences": 0,
        "description": "사용자에 대해 조금씩 배우고 있어요.",
    },
    3: {
        "name": "어린이 (Child)",
        "min_xp": 500,
        "min_conversations": 50,
        "min_preferences": 20,
        "description": "사용자 취향을 파악하고 추천을 시작해요.",
    },
    4: {
        "name": "청소년 (Teen)",
        "min_xp": 2000,
        "min_conversations": 200,
        "min_preferences": 50,
        "description": "선제적으로 컨텐츠를 추천할 수 있어요.",
    },
    5: {
        "name": "성인 (Adult)",
        "min_xp": 5000,
        "min_conversations": 500,
        "min_preferences": 100,
        "description": "복잡한 의사결정도 도와줄 수 있어요.",
    },
    6: {
        "name": "마스터 (Master)",
        "min_xp": 10000,
        "min_conversations": 1000,
        "min_preferences": 200,
        "description": "사용자의 분신 수준으로 이해하고 있어요.",
    },
}


class GrowthManager:
    """Manages XP, levels, and evolution checks."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def get_state(self) -> dict:
        return self.store.get_growth_state()

    def add_xp_for_conversation(self, new_preferences: int = 0) -> int:
        """Award XP after a conversation. Returns total XP gained."""
        state = self.store.get_growth_state()
        xp_gained = XP_CONVERSATION + (XP_NEW_PREFERENCE * new_preferences)
        self.store.update_growth_state(
            xp=state["xp"] + xp_gained,
            total_conversations=state["total_conversations"] + 1,
            total_preferences=state["total_preferences"] + new_preferences,
        )
        return xp_gained

    def add_xp_for_recommendation(self, accepted: bool) -> int:
        state = self.store.get_growth_state()
        xp = XP_RECOMMENDATION_ACCEPTED if accepted else XP_RECOMMENDATION_REJECTED
        updates: dict = {"xp": state["xp"] + xp}
        if accepted:
            updates["total_recommendations_accepted"] = state["total_recommendations_accepted"] + 1
        else:
            updates["total_recommendations_rejected"] = state["total_recommendations_rejected"] + 1
        self.store.update_growth_state(**updates)
        return xp

    def check_level_up(self) -> bool:
        """Check if the Tamagotchi should level up. Returns True if leveled up."""
        state = self.store.get_growth_state()
        current_level = state["level"]
        next_level = current_level + 1

        if next_level not in LEVELS:
            return False

        req = LEVELS[next_level]
        total_prefs = self.store.count_preferences()

        if (
            state["xp"] >= req["min_xp"]
            and state["total_conversations"] >= req["min_conversations"]
            and total_prefs >= req["min_preferences"]
        ):
            self.store.update_growth_state(level=next_level)
            return True

        return False

    def xp_to_next_level(self) -> int | None:
        """Returns XP needed for next level, or None if max level."""
        state = self.store.get_growth_state()
        next_level = state["level"] + 1
        if next_level not in LEVELS:
            return None
        return max(0, LEVELS[next_level]["min_xp"] - state["xp"])
