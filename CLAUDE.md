# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Tamagotchi?

A personalizable AI agent that learns user preferences through conversation — like raising a Tamagotchi. It builds a persistent profile (shopping habits, content preferences, routines, etc.), develops a unique personality based on interactions, and uses Claude tool use for tasks, calendar, shopping, content recommendations, and web search.

## Build & Run

```bash
uv sync --extra dev               # Install all dependencies (dev + test)
uv sync --extra web --extra dev   # Include web server dependencies
export ANTHROPIC_API_KEY=sk-...   # Required for chat

uv run tamagotchi chat            # Start conversation (CLI, with tool use)
uv run tamagotchi status          # View growth state + personality traits
uv run tamagotchi profile         # View learned preferences
uv run tamagotchi profile forget <category> [key]  # Delete memories
uv run tamagotchi history         # Recent conversation history
uv run tamagotchi serve           # Start web server (http://localhost:8000)
uv run tamagotchi export [file]   # Export all data to JSON
uv run tamagotchi import <file>   # Import data from JSON
uv run tamagotchi reset --yes     # Factory reset
```

## Testing

```bash
uv run pytest tests/unit/ -v      # All 86 unit tests (no API key needed)
uv run pytest tests/unit/test_memory.py -v           # Memory store
uv run pytest tests/unit/test_semantic.py -v          # Semantic/episodic memory
uv run pytest tests/unit/test_patterns.py -v          # Pattern & feedback
uv run pytest tests/unit/test_integrations.py -v      # Tasks, calendar, tools
uv run pytest tests/unit/test_personality.py -v       # Personality, export/import
uv run pytest tests/unit/test_growth.py -v            # Growth/XP system
```

## Architecture

**Data flow per conversation turn:**
1. User input → Agent loads profile + retrieves relevant episodes (ChromaDB) + detects patterns + loads personality
2. Dynamic system prompt: profile + episodic context + patterns + personality traits + growth-level behavior
3. Claude API call with 7 tool definitions
4. If tools called → execute → return results → Claude responds (up to 5 rounds)
5. Post-session: Haiku extracts preferences, personality signals detected → all state updated → XP/level check

**Key modules:**

| Module | Purpose |
|--------|---------|
| `agent/core.py` | `TamagotchiAgent` — conversation loop with tool use + personality evolution |
| `agent/tools.py` | 7 tools: `manage_tasks`, `manage_calendar`, `recommend_content`, `search_web`, `shopping_recommend`, `remember_preference`, `search_memories` |
| `agent/prompts.py` | `build_system_prompt()` — profile + episodic + patterns + personality → system prompt |
| `memory/store.py` | SQLite — tables: `preferences`, `episodes`, `growth_state`, `personality`, `tasks`, `calendar_events` |
| `memory/profile.py` | `UserProfile` + `ProfileManager`. `to_prompt()` for system prompt injection |
| `memory/semantic.py` | ChromaDB vector store — `episodes` + `preferences` collections |
| `memory/episodic.py` | `EpisodicMemory` — SQLite + ChromaDB combined. `recall_for_prompt()` for context |
| `learning/extractor.py` | Post-conversation preference extraction via Haiku |
| `learning/patterns.py` | `PatternAnalyzer` — time patterns, category trends, preference shifts |
| `learning/feedback.py` | `FeedbackTracker` — adjusts confidence on accept/reject |
| `growth/state.py` | 6-level XP system with composite level-up requirements |
| `growth/personality.py` | `PersonalityTraits` (6 dimensions) + `PersonalityManager` + signal detection |
| `growth/display.py` | Rich terminal: ASCII art, stats, personality trait bars |
| `integrations/tasks.py` | `TaskManager` — local todo CRUD |
| `integrations/calendar.py` | `CalendarManager` — local event CRUD |
| `integrations/content.py` | `ContentRecommender` — profile-based content via Haiku |
| `integrations/shopping.py` | `ShoppingAssistant` — profile-based shopping via Haiku |
| `integrations/web.py` | `WebSearcher` — knowledge-based search via Haiku |
| `data_manager.py` | `export_data()` / `import_data()` — full JSON backup/restore |
| `api/server.py` | FastAPI app + static `web/` frontend |
| `api/routes.py` | REST: `/api/chat`, `/api/status`, `/api/profile`, `/api/feedback`, `/api/history` |
| `cli.py` | Typer CLI: `chat`, `status`, `profile`, `history`, `serve`, `export`, `import`, `reset` |

**Personality system:** 6 traits (warmth, humor, curiosity, formality, energy, empathy) on 0.0-1.0 scale. Traits shift via interaction signals (e.g., user uses "ㅋㅋ" → humor increases). Dominant traits inject behavior instructions into system prompt.

**Growth levels:** Levels 1-2 = learn mode, Level 3 = recommendations, Level 4+ = proactive. Level-up requires XP + conversation count + preference count thresholds.

**Storage:** All local at `~/.tamagotchi/` — `memory.db` (SQLite) + `chroma/` (ChromaDB vectors).

## Tech Stack

- Python 3.12+, uv
- Claude API (Anthropic SDK) — Sonnet + tool use for chat, Haiku for extraction/recommendations
- SQLite + ChromaDB (both local, no server needed)
- Typer + Rich (CLI), FastAPI + Uvicorn (web)
- Pydantic (data models)
