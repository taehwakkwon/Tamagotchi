"""Tests for memory store and profile management."""

import tempfile
from pathlib import Path

import pytest

from tamagotchi.memory.store import MemoryStore
from tamagotchi.memory.profile import ProfileManager, UserProfile


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def profile_mgr(store):
    return ProfileManager(store)


class TestMemoryStore:
    def test_upsert_and_get_preferences(self, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=1.0)
        store.upsert_preference("food", "매운음식", "좋아함", confidence=0.8)
        store.upsert_preference("shopping", "나이키", "선호 브랜드", confidence=0.9)

        all_prefs = store.get_preferences()
        assert len(all_prefs) == 3

        food_prefs = store.get_preferences("food")
        assert len(food_prefs) == 2
        assert food_prefs[0]["key"] == "커피"  # higher confidence first

    def test_upsert_overwrites(self, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=0.5)
        store.upsert_preference("food", "커피", "매우 좋아함", confidence=1.0)

        prefs = store.get_preferences("food")
        assert len(prefs) == 1
        assert prefs[0]["value"] == "매우 좋아함"
        assert prefs[0]["confidence"] == 1.0

    def test_delete_preference(self, store):
        store.upsert_preference("food", "커피", "좋아함")
        assert store.delete_preference("food", "커피") is True
        assert store.delete_preference("food", "없는것") is False
        assert len(store.get_preferences("food")) == 0

    def test_delete_by_category(self, store):
        store.upsert_preference("food", "커피", "좋아함")
        store.upsert_preference("food", "차", "좋아함")
        store.upsert_preference("shopping", "나이키", "선호")

        count = store.delete_preferences_by_category("food")
        assert count == 2
        assert len(store.get_preferences("food")) == 0
        assert len(store.get_preferences("shopping")) == 1

    def test_count_preferences(self, store):
        assert store.count_preferences() == 0
        store.upsert_preference("food", "커피", "좋아함")
        store.upsert_preference("food", "차", "좋아함")
        assert store.count_preferences() == 2

    def test_save_and_get_episodes(self, store):
        messages = [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요!"},
        ]
        ep_id = store.save_episode("인사 대화", messages)
        assert ep_id >= 1

        episodes = store.get_recent_episodes(10)
        assert len(episodes) == 1
        assert episodes[0]["summary"] == "인사 대화"
        assert episodes[0]["messages"] == messages

    def test_growth_state_init(self, store):
        state = store.get_growth_state()
        assert state["level"] == 1
        assert state["xp"] == 0
        assert state["total_conversations"] == 0

    def test_growth_state_update(self, store):
        store.get_growth_state()  # init
        state = store.update_growth_state(xp=100, total_conversations=5)
        assert state["xp"] == 100
        assert state["total_conversations"] == 5
        assert state["level"] == 1  # unchanged


class TestProfileManager:
    def test_load_empty(self, profile_mgr):
        profile = profile_mgr.load()
        assert isinstance(profile, UserProfile)
        assert len(profile.preferences) == 0
        assert profile.growth_level == 1

    def test_add_and_load(self, profile_mgr):
        profile_mgr.add_preference("food", "커피", "좋아함", confidence=0.9)
        profile_mgr.add_preference("food", "차", "보통", confidence=0.5)

        profile = profile_mgr.load()
        assert len(profile.preferences) == 2
        assert profile.preferences[0].key == "커피"

    def test_forget_specific(self, profile_mgr):
        profile_mgr.add_preference("food", "커피", "좋아함")
        profile_mgr.add_preference("food", "차", "좋아함")

        count = profile_mgr.forget("food", "커피")
        assert count == 1

        profile = profile_mgr.load()
        assert len(profile.preferences) == 1

    def test_forget_category(self, profile_mgr):
        profile_mgr.add_preference("food", "커피", "좋아함")
        profile_mgr.add_preference("food", "차", "좋아함")
        profile_mgr.add_preference("shopping", "나이키", "선호")

        count = profile_mgr.forget("food")
        assert count == 2

        profile = profile_mgr.load()
        assert len(profile.preferences) == 1

    def test_to_prompt(self, profile_mgr):
        profile_mgr.add_preference("food", "커피", "좋아함", confidence=0.95)
        profile = profile_mgr.load()

        prompt = profile.to_prompt()
        assert "food" in prompt
        assert "커피" in prompt
        assert "좋아함" in prompt
        assert "확실" in prompt

    def test_to_prompt_empty(self, profile_mgr):
        profile = profile_mgr.load()
        prompt = profile.to_prompt()
        assert "아직" in prompt
