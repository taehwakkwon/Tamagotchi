"""Tests for pattern recognition and feedback learning."""

import pytest

from tamagotchi.growth.state import GrowthManager
from tamagotchi.learning.feedback import FeedbackTracker
from tamagotchi.learning.patterns import PatternAnalyzer
from tamagotchi.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def analyzer(store):
    return PatternAnalyzer(store)


@pytest.fixture
def tracker(store):
    growth = GrowthManager(store)
    return FeedbackTracker(store, growth)


class TestPatternAnalyzer:
    def test_empty_patterns(self, analyzer):
        patterns = analyzer.analyze_all()
        assert patterns == []

    def test_category_trends(self, analyzer, store):
        # Add enough preferences to trigger category trend detection
        for i in range(10):
            store.upsert_preference("food", f"item_{i}", "좋아함")
        for i in range(3):
            store.upsert_preference("shopping", f"item_{i}", "선호")

        patterns = analyzer.analyze_all()
        category_patterns = [p for p in patterns if p["type"] == "category_trend"]
        assert len(category_patterns) > 0
        # food should be dominant
        food_patterns = [p for p in category_patterns if "food" in p["key"]]
        assert len(food_patterns) == 1

    def test_patterns_for_prompt(self, analyzer, store):
        for i in range(10):
            store.upsert_preference("food", f"item_{i}", "좋아함")

        text = analyzer.get_patterns_for_prompt()
        assert "행동 패턴" in text or text == ""

    def test_patterns_for_prompt_empty(self, analyzer):
        assert analyzer.get_patterns_for_prompt() == ""


class TestFeedbackTracker:
    def test_record_accepted(self, tracker, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=0.7)

        new_conf = tracker.record_feedback("food", "커피", accepted=True)
        assert new_conf is not None
        assert new_conf > 0.7

    def test_record_rejected(self, tracker, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=0.7)

        new_conf = tracker.record_feedback("food", "커피", accepted=False)
        assert new_conf is not None
        assert new_conf < 0.7

    def test_record_nonexistent_preference(self, tracker):
        result = tracker.record_feedback("food", "없는것", accepted=True)
        assert result is None

    def test_confidence_bounds(self, tracker, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=0.99)
        new_conf = tracker.record_feedback("food", "커피", accepted=True)
        assert new_conf <= 1.0

        store.upsert_preference("food", "차", "싫어함", confidence=0.15)
        new_conf = tracker.record_feedback("food", "차", accepted=False)
        assert new_conf >= 0.1

    def test_acceptance_rate(self, tracker, store):
        assert tracker.get_acceptance_rate() is None

        store.upsert_preference("food", "커피", "좋아함", confidence=0.7)
        tracker.record_feedback("food", "커피", accepted=True)
        tracker.record_feedback("food", "커피", accepted=True)
        tracker.record_feedback("food", "커피", accepted=False)

        rate = tracker.get_acceptance_rate()
        assert rate is not None
        assert abs(rate - 2 / 3) < 0.01

    def test_category_accuracy(self, tracker, store):
        store.upsert_preference("food", "커피", "좋아함", confidence=0.9)
        store.upsert_preference("food", "차", "좋아함", confidence=0.8)
        store.upsert_preference("shopping", "나이키", "선호", confidence=0.5)

        acc = tracker.get_category_accuracy()
        assert "food" in acc
        assert "shopping" in acc
        assert acc["food"] > acc["shopping"]
