"""Feedback loop — learns from recommendation acceptance/rejection."""

from __future__ import annotations

from tamagotchi.growth.state import GrowthManager
from tamagotchi.memory.store import MemoryStore


# How much confidence changes per feedback event
CONFIDENCE_BOOST = 0.1
CONFIDENCE_DECAY = 0.15
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 1.0


class FeedbackTracker:
    """Adjusts preference confidence based on recommendation outcomes."""

    def __init__(self, store: MemoryStore, growth: GrowthManager):
        self.store = store
        self.growth = growth

    def record_feedback(
        self,
        category: str,
        key: str,
        accepted: bool,
    ) -> float | None:
        """Record whether a recommendation was accepted or rejected.

        Adjusts the preference confidence and awards XP.
        Returns the new confidence, or None if the preference doesn't exist.
        """
        prefs = self.store.get_preferences(category)
        target = None
        for p in prefs:
            if p["key"] == key:
                target = p
                break

        if target is None:
            # Preference not found — still award XP
            self.growth.add_xp_for_recommendation(accepted)
            return None

        old_confidence = target["confidence"]

        if accepted:
            new_confidence = min(MAX_CONFIDENCE, old_confidence + CONFIDENCE_BOOST)
        else:
            new_confidence = max(MIN_CONFIDENCE, old_confidence - CONFIDENCE_DECAY)

        self.store.upsert_preference(
            category=category,
            key=key,
            value=target["value"],
            confidence=new_confidence,
            source=target["source"],
        )

        self.growth.add_xp_for_recommendation(accepted)

        return new_confidence

    def get_acceptance_rate(self) -> float | None:
        """Get overall recommendation acceptance rate."""
        state = self.store.get_growth_state()
        total = state["total_recommendations_accepted"] + state["total_recommendations_rejected"]
        if total == 0:
            return None
        return state["total_recommendations_accepted"] / total

    def get_category_accuracy(self) -> dict[str, float]:
        """Estimate per-category accuracy based on average confidence.

        Higher average confidence in a category suggests better accuracy there.
        """
        prefs = self.store.get_preferences()
        cat_confidences: dict[str, list[float]] = {}
        for p in prefs:
            cat_confidences.setdefault(p["category"], []).append(p["confidence"])

        return {
            cat: sum(confs) / len(confs)
            for cat, confs in cat_confidences.items()
        }
