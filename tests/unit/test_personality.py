"""Tests for personality development and data export/import."""

import json

import pytest

from tamagotchi.growth.personality import PersonalityManager, PersonalityTraits, TRAIT_SHIFT
from tamagotchi.data_manager import export_data, import_data
from tamagotchi.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def personality(store):
    return PersonalityManager(store)


class TestPersonalityTraits:
    def test_default_values(self):
        traits = PersonalityTraits()
        assert traits.warmth == 0.5
        assert traits.humor == 0.5
        assert traits.curiosity == 0.5

    def test_to_prompt_neutral(self):
        traits = PersonalityTraits()
        prompt = traits.to_prompt()
        assert "균형" in prompt

    def test_to_prompt_warm(self):
        traits = PersonalityTraits(warmth=0.9, humor=0.8)
        prompt = traits.to_prompt()
        assert "따뜻" in prompt
        assert "유머" in prompt

    def test_to_prompt_cool(self):
        traits = PersonalityTraits(warmth=0.2, humor=0.2)
        prompt = traits.to_prompt()
        assert "간결" in prompt
        assert "진지" in prompt

    def test_get_dominant_traits(self):
        traits = PersonalityTraits(warmth=0.9, humor=0.2, energy=0.8)
        dominant = traits.get_dominant_traits()
        assert "따뜻한" in dominant
        assert "진지한" in dominant
        assert "활발한" in dominant

    def test_get_dominant_traits_neutral(self):
        traits = PersonalityTraits()
        assert traits.get_dominant_traits() == []

    def test_summary(self):
        traits = PersonalityTraits(warmth=0.9)
        assert "따뜻한" in traits.summary()

    def test_summary_neutral(self):
        traits = PersonalityTraits()
        assert traits.summary() == "균형 잡힌 성격"


class TestPersonalityManager:
    def test_load_initial(self, personality):
        traits = personality.load()
        assert traits.warmth == 0.5
        assert traits.humor == 0.5

    def test_apply_signal(self, personality):
        traits = personality.apply_signal("user_laughed")
        assert traits.humor > 0.5
        assert traits.warmth > 0.5

    def test_apply_signals_multiple(self, personality):
        traits = personality.apply_signals(["user_laughed", "user_emotional", "user_casual"])
        assert traits.humor > 0.5
        assert traits.empathy > 0.5
        assert traits.formality < 0.5

    def test_signal_persistence(self, personality):
        personality.apply_signal("user_laughed")
        traits = personality.load()
        assert traits.humor > 0.5

    def test_trait_bounds(self, personality):
        # Apply many signals to test bounds
        for _ in range(100):
            personality.apply_signal("user_laughed")
        traits = personality.load()
        assert traits.humor <= 1.0
        assert traits.warmth <= 1.0

    def test_detect_signals_laugh(self, personality):
        msgs = [
            {"role": "user", "content": "ㅋㅋㅋ 진짜 웃기다"},
            {"role": "assistant", "content": "감사합니다!"},
        ]
        signals = personality.detect_signals(msgs)
        assert "user_laughed" in signals

    def test_detect_signals_emotional(self, personality):
        msgs = [
            {"role": "user", "content": "오늘 너무 힘들었어... 슬프다"},
            {"role": "assistant", "content": "힘내세요"},
        ]
        signals = personality.detect_signals(msgs)
        assert "user_emotional" in signals

    def test_detect_signals_casual(self, personality):
        msgs = [
            {"role": "user", "content": "ㅇㅇ ㄱㄱ"},
            {"role": "assistant", "content": "네!"},
        ]
        signals = personality.detect_signals(msgs)
        assert "user_casual" in signals

    def test_detect_signals_long_conversation(self, personality):
        msgs = [{"role": "user", "content": f"메시지 {i}"} for i in range(10)]
        signals = personality.detect_signals(msgs)
        assert "long_conversation" in signals

    def test_detect_signals_short_conversation(self, personality):
        msgs = [{"role": "user", "content": "안녕"}]
        signals = personality.detect_signals(msgs)
        assert "short_conversation" in signals

    def test_detect_signals_empty(self, personality):
        assert personality.detect_signals([]) == []


class TestDataExportImport:
    def test_export(self, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=0.9)
        store.upsert_preference("shopping", "나이키", "선호")
        store.get_growth_state()  # init
        store.update_growth_state(xp=100, total_conversations=5)

        data = export_data(store)
        assert data["version"] == "1.0"
        assert len(data["preferences"]) == 2
        assert data["growth_state"]["xp"] == 100
        assert data["personality"]["warmth"] == 0.5

    def test_import(self, tmp_path):
        # Export from one store
        store1 = MemoryStore(db_path=tmp_path / "source.db")
        store1.upsert_preference("food", "커피", "좋아함")
        store1.upsert_preference("food", "차", "보통")
        store1.get_growth_state()
        store1.update_growth_state(xp=200, level=2, total_conversations=15)
        data = export_data(store1)
        store1.close()

        # Import to another store
        store2 = MemoryStore(db_path=tmp_path / "target.db")
        counts = import_data(store2, data)
        assert counts["preferences"] == 2

        prefs = store2.get_preferences()
        assert len(prefs) == 2
        state = store2.get_growth_state()
        assert state["xp"] == 200
        assert state["level"] == 2
        store2.close()

    def test_export_with_tasks_and_events(self, store):
        store.get_growth_state()  # init
        store._conn.execute(
            "INSERT INTO tasks (title, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("할일", "pending", "high", "2026-01-01", "2026-01-01"),
        )
        store._conn.execute(
            "INSERT INTO calendar_events (title, event_date, created_at) VALUES (?, ?, ?)",
            ("이벤트", "2026-04-01", "2026-01-01"),
        )
        store._conn.commit()

        data = export_data(store)
        assert len(data["tasks"]) == 1
        assert len(data["calendar_events"]) == 1

    def test_import_tasks_and_events(self, tmp_path):
        data = {
            "version": "1.0",
            "growth_state": {"xp": 0, "level": 1, "total_conversations": 0,
                             "total_preferences": 0, "total_recommendations_accepted": 0,
                             "total_recommendations_rejected": 0},
            "personality": {"warmth": 0.7, "humor": 0.3, "curiosity": 0.5,
                            "formality": 0.5, "energy": 0.5, "empathy": 0.8},
            "preferences": [],
            "episodes": [],
            "tasks": [{"title": "할일", "status": "pending", "priority": "high"}],
            "calendar_events": [{"title": "회의", "event_date": "2026-04-01"}],
        }

        store = MemoryStore(db_path=tmp_path / "import.db")
        counts = import_data(store, data)
        assert counts["tasks"] == 1
        assert counts["calendar_events"] == 1

        # Check personality was imported
        pm = PersonalityManager(store)
        traits = pm.load()
        assert traits.warmth == 0.7
        assert traits.empathy == 0.8
        store.close()
