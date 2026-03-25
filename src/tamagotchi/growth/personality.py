"""Personality development — traits evolve based on interaction patterns."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from tamagotchi.memory.store import MemoryStore

# Trait range: 0.0 to 1.0
TRAIT_MIN = 0.0
TRAIT_MAX = 1.0
TRAIT_SHIFT = 0.03  # How much a trait changes per interaction signal


class PersonalityTraits(BaseModel):
    """Six personality dimensions that evolve over time."""

    warmth: float = Field(0.5, ge=0.0, le=1.0, description="따뜻함 — 친근하고 다정한 정도")
    humor: float = Field(0.5, ge=0.0, le=1.0, description="유머 — 재미있고 장난스러운 정도")
    curiosity: float = Field(0.5, ge=0.0, le=1.0, description="호기심 — 적극적으로 질문하는 정도")
    formality: float = Field(0.5, ge=0.0, le=1.0, description="격식 — 존댓말/격식체 정도")
    energy: float = Field(0.5, ge=0.0, le=1.0, description="에너지 — 활발하고 적극적인 정도")
    empathy: float = Field(0.5, ge=0.0, le=1.0, description="공감 — 감정에 공감하고 위로하는 정도")

    def to_prompt(self) -> str:
        """Convert personality to behavior instructions for the system prompt."""
        instructions = []

        if self.warmth > 0.7:
            instructions.append("매우 따뜻하고 다정하게 대화하세요. 애칭이나 관심 표현을 자연스럽게 사용하세요.")
        elif self.warmth < 0.3:
            instructions.append("간결하고 실용적으로 대화하세요.")

        if self.humor > 0.7:
            instructions.append("적절한 유머와 위트를 섞어 대화하세요. 가벼운 농담도 괜찮습니다.")
        elif self.humor < 0.3:
            instructions.append("진지하고 차분한 톤을 유지하세요.")

        if self.curiosity > 0.7:
            instructions.append("사용자에 대해 적극적으로 질문하고 관심을 보이세요.")
        elif self.curiosity < 0.3:
            instructions.append("필요한 경우에만 질문하세요.")

        if self.formality > 0.7:
            instructions.append("격식 있는 존댓말을 사용하세요.")
        elif self.formality < 0.3:
            instructions.append("편안한 반말 톤으로 대화하세요.")

        if self.energy > 0.7:
            instructions.append("활발하고 열정적으로 반응하세요. 감탄사를 적절히 사용하세요.")
        elif self.energy < 0.3:
            instructions.append("차분하고 조용한 톤을 유지하세요.")

        if self.empathy > 0.7:
            instructions.append("사용자의 감정에 깊이 공감하고, 위로나 격려의 말을 건네세요.")

        if not instructions:
            return "균형 잡힌 중립적 톤으로 대화하세요."

        return "\n".join(f"- {i}" for i in instructions)

    def get_dominant_traits(self) -> list[str]:
        """Return trait names where the value is significantly high or low."""
        traits = []
        mapping = {
            "warmth": ("따뜻한", "쿨한"),
            "humor": ("유머러스한", "진지한"),
            "curiosity": ("호기심 많은", "과묵한"),
            "formality": ("격식있는", "편한"),
            "energy": ("활발한", "차분한"),
            "empathy": ("공감적인", "이성적인"),
        }
        for attr, (high, low) in mapping.items():
            val = getattr(self, attr)
            if val > 0.65:
                traits.append(high)
            elif val < 0.35:
                traits.append(low)
        return traits

    def summary(self) -> str:
        """One-line personality summary."""
        dominant = self.get_dominant_traits()
        if not dominant:
            return "균형 잡힌 성격"
        return ", ".join(dominant)


# Interaction signals → trait adjustments
SIGNAL_MAP: dict[str, dict[str, float]] = {
    "user_laughed": {"humor": TRAIT_SHIFT, "warmth": TRAIT_SHIFT * 0.5},
    "user_emotional": {"empathy": TRAIT_SHIFT, "warmth": TRAIT_SHIFT * 0.5},
    "user_casual": {"formality": -TRAIT_SHIFT, "warmth": TRAIT_SHIFT * 0.5},
    "user_formal": {"formality": TRAIT_SHIFT},
    "user_asked_question": {"curiosity": TRAIT_SHIFT * 0.5},
    "long_conversation": {"warmth": TRAIT_SHIFT * 0.5, "energy": TRAIT_SHIFT * 0.3},
    "short_conversation": {"energy": -TRAIT_SHIFT * 0.3},
    "recommendation_accepted": {"energy": TRAIT_SHIFT * 0.3, "curiosity": TRAIT_SHIFT * 0.3},
    "recommendation_rejected": {"curiosity": -TRAIT_SHIFT * 0.2},
    "user_shared_personal": {"empathy": TRAIT_SHIFT, "warmth": TRAIT_SHIFT},
}


class PersonalityManager:
    """Manages personality trait evolution."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def load(self) -> PersonalityTraits:
        row = self.store._conn.execute("SELECT * FROM personality WHERE id = 1").fetchone()
        if row is None:
            now = datetime.now(timezone.utc).isoformat()
            self.store._conn.execute(
                "INSERT INTO personality (id, warmth, humor, curiosity, formality, energy, empathy, updated_at) VALUES (1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ?)",
                (now,),
            )
            self.store._conn.commit()
            return PersonalityTraits()
        return PersonalityTraits(
            warmth=row["warmth"],
            humor=row["humor"],
            curiosity=row["curiosity"],
            formality=row["formality"],
            energy=row["energy"],
            empathy=row["empathy"],
        )

    def apply_signal(self, signal: str) -> PersonalityTraits:
        """Apply an interaction signal to shift personality traits."""
        traits = self.load()
        adjustments = SIGNAL_MAP.get(signal, {})

        for attr, delta in adjustments.items():
            current = getattr(traits, attr)
            new_val = max(TRAIT_MIN, min(TRAIT_MAX, current + delta))
            setattr(traits, attr, round(new_val, 3))

        self._save(traits)
        return traits

    def apply_signals(self, signals: list[str]) -> PersonalityTraits:
        """Apply multiple interaction signals."""
        traits = self.load()
        for signal in signals:
            adjustments = SIGNAL_MAP.get(signal, {})
            for attr, delta in adjustments.items():
                current = getattr(traits, attr)
                new_val = max(TRAIT_MIN, min(TRAIT_MAX, current + delta))
                setattr(traits, attr, round(new_val, 3))

        self._save(traits)
        return traits

    def _save(self, traits: PersonalityTraits) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.store._conn.execute(
            """UPDATE personality SET
                warmth = ?, humor = ?, curiosity = ?,
                formality = ?, energy = ?, empathy = ?,
                updated_at = ?
            WHERE id = 1""",
            (traits.warmth, traits.humor, traits.curiosity,
             traits.formality, traits.energy, traits.empathy, now),
        )
        self.store._conn.commit()

    def detect_signals(self, messages: list[dict[str, str]]) -> list[str]:
        """Detect interaction signals from a conversation."""
        signals = []
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]

        if not user_msgs:
            return signals

        all_text = " ".join(user_msgs).lower()

        # Detect humor signals
        laugh_markers = ["ㅋㅋ", "ㅎㅎ", "ㅋ", "ㅎ", "lol", "haha", "😂", "🤣", "재밌", "웃기"]
        if any(m in all_text for m in laugh_markers):
            signals.append("user_laughed")

        # Detect emotional signals
        emotion_markers = ["슬프", "힘들", "우울", "짜증", "화가", "기쁘", "행복", "감사", "고마워", "사랑"]
        if any(m in all_text for m in emotion_markers):
            signals.append("user_emotional")

        # Detect formality
        casual_markers = ["ㅇㅇ", "ㄱㄱ", "ㄴㄴ", "ㅇㅋ", "ㅎ", "ㅋ"]
        formal_markers = ["감사합니다", "부탁드립니다", "하십시오", "입니다"]
        if any(m in all_text for m in casual_markers):
            signals.append("user_casual")
        elif any(m in all_text for m in formal_markers):
            signals.append("user_formal")

        # Detect personal sharing
        personal_markers = ["나는", "내가", "우리", "제가", "저는", "솔직히", "사실은"]
        if sum(1 for m in personal_markers if m in all_text) >= 2:
            signals.append("user_shared_personal")

        # Conversation length
        if len(user_msgs) >= 8:
            signals.append("long_conversation")
        elif len(user_msgs) <= 2:
            signals.append("short_conversation")

        # Question frequency
        question_count = sum(1 for msg in user_msgs if "?" in msg or "뭐" in msg or "어떻게" in msg)
        if question_count >= 3:
            signals.append("user_asked_question")

        return signals
