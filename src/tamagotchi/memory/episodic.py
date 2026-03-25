"""Episodic memory — combines SQLite storage with ChromaDB semantic search."""

from __future__ import annotations

from typing import Any

from tamagotchi.memory.semantic import SemanticMemory
from tamagotchi.memory.store import MemoryStore


class EpisodicMemory:
    """Manages episode lifecycle: save to SQLite + index in ChromaDB for retrieval."""

    def __init__(self, store: MemoryStore, semantic: SemanticMemory):
        self.store = store
        self.semantic = semantic

    def save(
        self,
        summary: str,
        messages: list[dict[str, str]],
        preferences_extracted: list[dict] | None = None,
    ) -> int:
        """Save an episode to both SQLite and vector store."""
        episode_id = self.store.save_episode(summary, messages, preferences_extracted)

        messages_text = "\n".join(
            f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in messages
        )

        metadata: dict[str, Any] = {
            "message_count": len(messages),
            "has_preferences": bool(preferences_extracted),
        }
        if preferences_extracted:
            categories = list({p.get("category", "") for p in preferences_extracted})
            metadata["preference_categories"] = ",".join(categories)

        self.semantic.add_episode(
            episode_id=episode_id,
            summary=summary,
            messages_text=messages_text,
            metadata=metadata,
        )

        return episode_id

    def recall(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        """Retrieve relevant past episodes for the given query/context.

        Returns a list of episodes with their summaries and relevance scores.
        """
        semantic_results = self.semantic.search_episodes(query, n_results=n_results)

        recalled = []
        for result in semantic_results:
            # Extract the summary (first line of the document)
            doc = result["document"]
            summary = doc.split("\n")[0] if doc else ""

            recalled.append({
                "episode_id": result["metadata"].get("episode_id"),
                "summary": summary,
                "relevance": 1.0 - (result["distance"] or 0.0),  # cosine distance → similarity
                "full_text": doc,
            })

        return recalled

    def recall_for_prompt(self, query: str, n_results: int = 3, max_chars: int = 1000) -> str:
        """Get a formatted text block of relevant memories for prompt injection."""
        episodes = self.recall(query, n_results=n_results)

        if not episodes:
            return ""

        lines = ["## 관련 과거 대화"]
        total_chars = 0
        for ep in episodes:
            relevance_pct = f"{ep['relevance']:.0%}"
            entry = f"- [{relevance_pct} 관련] {ep['summary']}"
            if total_chars + len(entry) > max_chars:
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent episodes from SQLite (chronological, not semantic)."""
        return self.store.get_recent_episodes(limit)
