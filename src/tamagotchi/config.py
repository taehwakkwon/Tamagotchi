"""Configuration — model selection and global settings."""

from __future__ import annotations

import os

# Available models with display info
MODELS = {
    "sonnet": {
        "id": "claude-sonnet-4-20250514",
        "name": "Sonnet",
        "description": "균형 잡힌 성능과 비용 (기본값)",
        "cost_per_chat": "~$0.02",
        "context_window": 200_000,
    },
    "haiku": {
        "id": "claude-haiku-4-5-20251001",
        "name": "Haiku",
        "description": "빠르고 저렴 (~80% 비용 절감)",
        "cost_per_chat": "~$0.004",
        "context_window": 200_000,
    },
    "opus": {
        "id": "claude-opus-4-20250514",
        "name": "Opus",
        "description": "최고 성능 (비용 높음)",
        "cost_per_chat": "~$0.10",
        "context_window": 200_000,
    },
}

# The extraction model is always Haiku (low cost, sufficient for JSON extraction)
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

DEFAULT_MODEL = "sonnet"
MAX_TOKENS = 16384  # max output tokens per response
MAX_TOOL_ROUNDS = 5

# Conversation history management
# Reserve tokens for system prompt + new response, use the rest for history
MAX_HISTORY_MESSAGES = 100  # hard cap on message count
MAX_HISTORY_CHARS = 300_000  # ~75K tokens worth of conversation history

# Episodic memory
EPISODIC_RECALL_RESULTS = 5
EPISODIC_RECALL_MAX_CHARS = 3000


def get_chat_model(model_name: str | None = None) -> str:
    """Resolve chat model ID from name or env var.

    Priority: explicit arg > TAMAGOTCHI_MODEL env var > default (sonnet)
    """
    name = model_name or os.environ.get("TAMAGOTCHI_MODEL", DEFAULT_MODEL)
    name = name.lower().strip()

    if name in MODELS:
        return MODELS[name]["id"]

    # Allow passing a full model ID directly
    if name.startswith("claude-"):
        return name

    return MODELS[DEFAULT_MODEL]["id"]


def get_model_display_name(model_id: str) -> str:
    """Get a friendly display name for a model ID."""
    for info in MODELS.values():
        if info["id"] == model_id:
            return info["name"]
    return model_id
