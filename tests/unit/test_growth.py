"""Tests for growth/state system."""

import pytest

from tamagotchi.growth.state import GrowthManager, LEVELS, XP_CONVERSATION, XP_NEW_PREFERENCE
from tamagotchi.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def growth(store):
    return GrowthManager(store)


class TestGrowthManager:
    def test_initial_state(self, growth):
        state = growth.get_state()
        assert state["level"] == 1
        assert state["xp"] == 0

    def test_add_xp_for_conversation(self, growth):
        xp = growth.add_xp_for_conversation(new_preferences=0)
        assert xp == XP_CONVERSATION

        state = growth.get_state()
        assert state["xp"] == XP_CONVERSATION
        assert state["total_conversations"] == 1

    def test_add_xp_with_preferences(self, growth):
        xp = growth.add_xp_for_conversation(new_preferences=3)
        assert xp == XP_CONVERSATION + (XP_NEW_PREFERENCE * 3)

        state = growth.get_state()
        assert state["total_preferences"] == 3

    def test_level_up(self, growth, store):
        # Manually set state to meet level 2 requirements
        store.update_growth_state(xp=100, total_conversations=10)

        assert growth.check_level_up() is True
        assert growth.get_state()["level"] == 2

    def test_no_level_up_insufficient_xp(self, growth, store):
        store.update_growth_state(xp=50, total_conversations=10)
        assert growth.check_level_up() is False

    def test_xp_to_next_level(self, growth):
        xp_needed = growth.xp_to_next_level()
        assert xp_needed == LEVELS[2]["min_xp"]  # 100

    def test_xp_to_next_level_max(self, growth, store):
        store.update_growth_state(level=6, xp=99999)
        assert growth.xp_to_next_level() is None

    def test_add_xp_for_recommendation(self, growth):
        xp = growth.add_xp_for_recommendation(accepted=True)
        assert xp == 50

        state = growth.get_state()
        assert state["total_recommendations_accepted"] == 1

        xp2 = growth.add_xp_for_recommendation(accepted=False)
        assert xp2 == 5
        state = growth.get_state()
        assert state["total_recommendations_rejected"] == 1
