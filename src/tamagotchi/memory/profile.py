"""User profile management — builds a structured view of learned preferences."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tamagotchi.memory.store import MemoryStore


class Preference(BaseModel):
    category: str
    key: str
    value: str
    confidence: float = 1.0
    source: str = "explicit"


class UserProfile(BaseModel):
    """Aggregated view of everything the Tamagotchi knows about the user."""

    preferences: list[Preference] = Field(default_factory=list)
    total_conversations: int = 0
    growth_level: int = 1
    growth_xp: int = 0

    def to_prompt(self) -> str:
        """Convert profile to a text block for system prompt injection."""
        if not self.preferences:
            return "아직 사용자에 대해 파악한 정보가 없습니다. 대화를 통해 알아가세요."

        by_category: dict[str, list[Preference]] = {}
        for p in self.preferences:
            by_category.setdefault(p.category, []).append(p)

        lines = []
        for cat, prefs in sorted(by_category.items()):
            lines.append(f"### {cat}")
            for p in sorted(prefs, key=lambda x: -x.confidence):
                conf_label = _confidence_label(p.confidence)
                lines.append(f"- {p.key}: {p.value} ({conf_label})")
            lines.append("")

        return "\n".join(lines).strip()

    def get_categories(self) -> list[str]:
        return sorted({p.category for p in self.preferences})


class ProfileManager:
    """Reads/writes the user profile via MemoryStore."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def load(self) -> UserProfile:
        prefs = self.store.get_preferences()
        growth = self.store.get_growth_state()
        return UserProfile(
            preferences=[Preference(**{k: p[k] for k in ("category", "key", "value", "confidence", "source")}) for p in prefs],
            total_conversations=growth["total_conversations"],
            growth_level=growth["level"],
            growth_xp=growth["xp"],
        )

    def add_preference(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> None:
        self.store.upsert_preference(category, key, value, confidence, source)

    def forget(self, category: str, key: str | None = None) -> int:
        if key:
            return 1 if self.store.delete_preference(category, key) else 0
        return self.store.delete_preferences_by_category(category)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.9:
        return "확실"
    if confidence >= 0.7:
        return "높음"
    if confidence >= 0.4:
        return "보통"
    return "추측"
