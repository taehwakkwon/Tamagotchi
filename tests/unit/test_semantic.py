"""Tests for semantic memory (ChromaDB) and episodic memory."""

import pytest

from tamagotchi.memory.semantic import SemanticMemory
from tamagotchi.memory.episodic import EpisodicMemory
from tamagotchi.memory.store import MemoryStore


@pytest.fixture
def semantic(tmp_path):
    return SemanticMemory(persist_dir=tmp_path / "chroma")


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def episodic(store, tmp_path):
    sem = SemanticMemory(persist_dir=tmp_path / "chroma")
    return EpisodicMemory(store, sem)


class TestSemanticMemory:
    def test_add_and_search_episode(self, semantic):
        semantic.add_episode(1, "커피 추천 대화", "사용자: 커피 좋아해\nAI: 어떤 커피를 좋아하세요?")
        semantic.add_episode(2, "운동 이야기", "사용자: 오늘 헬스장 갔어\nAI: 어떤 운동 하셨어요?")

        results = semantic.search_episodes("카페 라떼 마시고 싶어")
        assert len(results) >= 1
        # Coffee-related episode should be more relevant
        assert results[0]["metadata"]["episode_id"] == 1

    def test_empty_search(self, semantic):
        results = semantic.search_episodes("아무거나")
        assert results == []

    def test_episode_count(self, semantic):
        assert semantic.episode_count() == 0
        semantic.add_episode(1, "test", "content")
        assert semantic.episode_count() == 1

    def test_delete_episode(self, semantic):
        semantic.add_episode(1, "test", "content")
        semantic.delete_episode(1)
        assert semantic.episode_count() == 0

    def test_add_and_search_preferences(self, semantic):
        semantic.add_preference_context("food_coffee", "커피를 좋아함, 특히 아메리카노", {"category": "food"})
        semantic.add_preference_context("sport_running", "달리기를 자주 함", {"category": "lifestyle"})

        results = semantic.search_preferences("카페인 음료")
        assert len(results) >= 1
        assert results[0]["id"] == "food_coffee"

    def test_reset(self, semantic):
        semantic.add_episode(1, "test", "content")
        semantic.add_preference_context("test", "test", {})
        semantic.reset()
        assert semantic.episode_count() == 0


class TestEpisodicMemory:
    def test_save_and_recall(self, episodic):
        messages = [
            {"role": "user", "content": "나는 매운 떡볶이를 좋아해"},
            {"role": "assistant", "content": "매운 떡볶이 좋아하시는군요!"},
        ]
        ep_id = episodic.save("매운 음식 대화", messages)
        assert ep_id >= 1

        results = episodic.recall("떡볶이 먹고 싶어")
        assert len(results) == 1
        assert results[0]["relevance"] > 0

    def test_recall_for_prompt(self, episodic):
        messages = [
            {"role": "user", "content": "주말에 등산 갔어"},
            {"role": "assistant", "content": "어디 산 가셨어요?"},
        ]
        episodic.save("등산 대화", messages)

        prompt_text = episodic.recall_for_prompt("이번 주말 뭐 할까")
        assert "관련 과거 대화" in prompt_text
        assert "등산" in prompt_text

    def test_recall_empty(self, episodic):
        results = episodic.recall("아무거나")
        assert results == []

    def test_recall_for_prompt_empty(self, episodic):
        assert episodic.recall_for_prompt("test") == ""

    def test_get_recent(self, episodic):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        episodic.save("인사", messages)
        episodic.save("두번째 대화", messages)

        recent = episodic.get_recent(limit=5)
        assert len(recent) == 2
