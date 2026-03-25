# Tamagotchi

A personalizable AI agent that learns your preferences like a Tamagotchi.

## Quick Start

```bash
uv sync --extra dev
export ANTHROPIC_API_KEY=your-key-here
uv run tamagotchi chat
```

## Commands

```bash
tamagotchi chat              # Start a conversation
tamagotchi status            # View growth status
tamagotchi profile           # View learned preferences
tamagotchi profile forget    # Delete specific memories
tamagotchi history           # View conversation history
tamagotchi reset             # Reset all data
```
