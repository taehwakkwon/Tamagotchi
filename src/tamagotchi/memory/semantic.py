"""ChromaDB-based vector store for semantic memory search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from tamagotchi.memory.store import DEFAULT_DB_DIR

DEFAULT_CHROMA_DIR = DEFAULT_DB_DIR / "chroma"


class SemanticMemory:
    """Vector store for semantic similarity search over past episodes."""

    def __init__(self, persist_dir: Path = DEFAULT_CHROMA_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._episodes = self._client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
        )
        self._preferences = self._client.get_or_create_collection(
            name="preferences",
            metadata={"hnsw:space": "cosine"},
        )

    def add_episode(
        self,
        episode_id: int,
        summary: str,
        messages_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store an episode embedding for later semantic retrieval."""
        doc = f"{summary}\n\n{messages_text}"
        meta = metadata or {}
        meta["episode_id"] = episode_id
        self._episodes.upsert(
            ids=[str(episode_id)],
            documents=[doc],
            metadatas=[meta],
        )

    def search_episodes(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Find episodes semantically similar to the query."""
        if self._episodes.count() == 0:
            return []

        results = self._episodes.query(
            query_texts=[query],
            n_results=min(n_results, self._episodes.count()),
        )

        episodes = []
        for i in range(len(results["ids"][0])):
            episodes.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return episodes

    def add_preference_context(
        self,
        pref_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store preference context for semantic lookup."""
        meta = metadata if metadata else None
        self._preferences.upsert(
            ids=[pref_id],
            documents=[text],
            metadatas=[meta] if meta else None,
        )

    def search_preferences(
        self,
        query: str,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Find preferences semantically related to a query."""
        if self._preferences.count() == 0:
            return []

        results = self._preferences.query(
            query_texts=[query],
            n_results=min(n_results, self._preferences.count()),
        )

        prefs = []
        for i in range(len(results["ids"][0])):
            prefs.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return prefs

    def episode_count(self) -> int:
        return self._episodes.count()

    def delete_episode(self, episode_id: int) -> None:
        self._episodes.delete(ids=[str(episode_id)])

    def reset(self) -> None:
        """Delete all collections and recreate them."""
        self._client.delete_collection("episodes")
        self._client.delete_collection("preferences")
        self._episodes = self._client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
        )
        self._preferences = self._client.get_or_create_collection(
            name="preferences",
            metadata={"hnsw:space": "cosine"},
        )
