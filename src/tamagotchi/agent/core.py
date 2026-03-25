"""Agent core — conversation loop with tool use, supports Anthropic & OpenAI endpoints."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console

from tamagotchi.agent.prompts import build_system_prompt
from tamagotchi.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from tamagotchi.config import (
    EPISODIC_RECALL_MAX_CHARS,
    EPISODIC_RECALL_RESULTS,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_TOKENS,
    MAX_TOOL_ROUNDS,
)
from tamagotchi.growth.personality import PersonalityManager
from tamagotchi.growth.state import GrowthManager
from tamagotchi.learning.extractor import extract_preferences
from tamagotchi.learning.patterns import PatternAnalyzer
from tamagotchi.llm import LLMClient, create_llm_client
from tamagotchi.memory.episodic import EpisodicMemory
from tamagotchi.memory.profile import ProfileManager, UserProfile
from tamagotchi.memory.semantic import SemanticMemory
from tamagotchi.memory.store import MemoryStore

console = Console()


class TamagotchiAgent:
    """Main agent that orchestrates conversation, learning, and growth."""

    def __init__(
        self,
        store: MemoryStore,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        use_tools: bool = True,
    ):
        self.store = store
        self.use_tools = use_tools
        self.llm = create_llm_client(
            provider=provider, model=model, base_url=base_url, api_key=api_key,
        )
        self.profile_mgr = ProfileManager(store)
        self.growth_mgr = GrowthManager(store)
        self.personality_mgr = PersonalityManager(store)
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory(store, self.semantic)
        self.patterns = PatternAnalyzer(store)
        self.tool_executor = ToolExecutor(store, client=None, semantic=self.semantic)  # type: ignore
        self.messages: list[dict[str, Any]] = []
        self._full_messages: list[dict[str, Any]] = []

    def chat_loop(self) -> None:
        """Interactive chat loop — runs until user types 'quit' or Ctrl+C."""
        profile = self.profile_mgr.load()

        console.print()
        console.print(f"[dim]모델: {self.llm.model} ({type(self.llm).__name__})[/dim]")
        self._show_greeting(profile)
        console.print("[dim]'quit' 또는 'exit'로 대화를 종료합니다.[/dim]\n")

        try:
            while True:
                try:
                    user_input = console.input("[bold cyan]나> [/bold cyan]")
                except EOFError:
                    break

                if user_input.strip().lower() in ("quit", "exit", "종료"):
                    break

                if not user_input.strip():
                    continue

                self.messages.append({"role": "user", "content": user_input})
                self._full_messages.append({"role": "user", "content": user_input})

                # Build context-aware system prompt
                episodic_context = self.episodic.recall_for_prompt(
                    user_input,
                    n_results=EPISODIC_RECALL_RESULTS,
                    max_chars=EPISODIC_RECALL_MAX_CHARS,
                )
                patterns_context = self.patterns.get_patterns_for_prompt()
                profile = self.profile_mgr.load()
                personality = self.personality_mgr.load()
                system_prompt = build_system_prompt(
                    profile, self.growth_mgr,
                    episodic_context=episodic_context,
                    patterns_context=patterns_context,
                    personality=personality,
                )

                trimmed = _trim_messages(self.messages)
                assistant_text = self._run_with_tools(system_prompt, trimmed)
                console.print()

        except KeyboardInterrupt:
            console.print("\n")

        self._on_session_end(profile)

    def _run_with_tools(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
        """Run LLM with tool use — handles multi-turn tool calls."""
        final_text = ""
        tools = TOOL_DEFINITIONS if self.use_tools else None

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.llm.chat(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=MAX_TOKENS,
            )

            # Display text
            if response.text:
                console.print(f"[bold green]다마고치> [/bold green]{response.text}")
                final_text += response.text

            # Build assistant message in the right format for the provider
            assistant_msg = self.llm.build_assistant_message(response)
            messages.append(assistant_msg)
            self.messages.append(assistant_msg)
            self._full_messages.append(assistant_msg)

            # If no tool calls, we're done
            if not response.tool_calls:
                break

            # Execute tools
            results = []
            for tc in response.tool_calls:
                console.print(f"[dim]  [{tc.name}] 실행 중...[/dim]")
                result = self.tool_executor.execute(tc.name, tc.input)
                results.append(result)

            # Build tool result messages in the right format
            tool_result_msgs = self.llm.build_tool_result_messages(response.tool_calls, results)

            if isinstance(self.llm, _get_openai_class()):
                # OpenAI: each tool result is a separate message
                for trm in tool_result_msgs:
                    messages.append(trm)
                    self.messages.append(trm)
                    self._full_messages.append(trm)
            else:
                # Anthropic: tool results are wrapped in a user message
                tool_msg = {"role": "user", "content": tool_result_msgs}
                messages.append(tool_msg)
                self.messages.append(tool_msg)
                self._full_messages.append(tool_msg)

        return final_text

    def _show_greeting(self, profile: UserProfile) -> None:
        level = profile.growth_level
        convos = profile.total_conversations
        if convos == 0:
            console.print("[bold yellow]안녕하세요! 저는 당신의 다마고치입니다. 저를 키워주세요![/bold yellow]")
        elif level < 3:
            console.print(f"[bold yellow]다시 만나서 반가워요! (Lv.{level}) 당신에 대해 더 알고 싶어요.[/bold yellow]")
        else:
            console.print(f"[bold yellow]어서오세요! (Lv.{level}) 오늘도 좋은 하루 보내세요.[/bold yellow]")

    def _on_session_end(self, profile_before: UserProfile) -> None:
        """Post-conversation: extract preferences, update growth."""
        if len(self._full_messages) < 2:
            console.print("[dim]대화가 너무 짧아 학습할 내용이 없어요.[/dim]")
            return

        console.print("\n[dim]대화 내용을 학습하는 중...[/dim]")
        text_messages = _extract_text_messages(self._full_messages)

        # Extract preferences
        extracted: list[dict] = []
        try:
            extracted = extract_preferences(self.llm, text_messages)
            for pref in extracted:
                self.profile_mgr.add_preference(
                    category=pref["category"],
                    key=pref["key"],
                    value=pref["value"],
                    confidence=pref.get("confidence", 0.8),
                    source="implicit",
                )
            new_prefs = len(extracted)
        except Exception:
            new_prefs = 0

        # Save episode
        try:
            summary = self._summarize_conversation()
            self.episodic.save(
                summary=summary,
                messages=text_messages,
                preferences_extracted=extracted if new_prefs > 0 else None,
            )
        except Exception:
            pass

        # Update personality
        try:
            signals = self.personality_mgr.detect_signals(text_messages)
            if signals:
                self.personality_mgr.apply_signals(signals)
        except Exception:
            pass

        # Update growth
        xp_gained = self.growth_mgr.add_xp_for_conversation(new_preferences=new_prefs)
        leveled_up = self.growth_mgr.check_level_up()

        console.print(f"[dim]+{xp_gained} XP 획득! ", end="")
        if new_prefs > 0:
            console.print(f"새로운 선호도 {new_prefs}개 학습!", end="")
        console.print("[/dim]")

        if leveled_up:
            state = self.growth_mgr.get_state()
            from tamagotchi.growth.state import LEVELS
            level_info = LEVELS[state["level"]]
            console.print(f"\n[bold magenta]레벨 업! Lv.{state['level']} {level_info['name']}으로 성장했어요![/bold magenta]")

    def _summarize_conversation(self) -> str:
        text_messages = _extract_text_messages(self._full_messages)
        user_msgs = [m["content"] for m in text_messages if m["role"] == "user"]
        if len(user_msgs) <= 2:
            return " / ".join(user_msgs)
        return f"{user_msgs[0][:50]}... ({len(user_msgs)}개 메시지)"


def _get_openai_class():
    """Lazy import to avoid circular dependency."""
    from tamagotchi.llm import OpenAIClient
    return OpenAIClient


def _trim_messages(
    messages: list[dict[str, Any]],
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[dict[str, Any]]:
    """Trim conversation history to fit within limits."""
    if len(messages) <= max_messages:
        total_chars = sum(_message_chars(m) for m in messages)
        if total_chars <= max_chars:
            return messages

    result: list[dict[str, Any]] = []
    char_count = 0
    for msg in reversed(messages):
        msg_chars = _message_chars(msg)
        if len(result) >= max_messages or char_count + msg_chars > max_chars:
            break
        result.append(msg)
        char_count += msg_chars

    result.reverse()
    if result and result[0]["role"] != "user":
        result = result[1:]
    return result if result else messages[-2:]


def _message_chars(msg: dict[str, Any]) -> int:
    content = msg.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if hasattr(block, "text"):
                total += len(block.text)
            elif isinstance(block, dict):
                total += len(str(block.get("content", "")))
        return total
    return len(str(content))


def _extract_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract plain text messages from a conversation that may include tool use blocks."""
    text_msgs: list[dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "tool":
            continue  # Skip OpenAI-format tool results

        if isinstance(content, str):
            text_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            texts = []
            for block in content:
                if hasattr(block, "type") and block.type == "text":
                    texts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block["text"])
            if texts:
                text_msgs.append({"role": role, "content": " ".join(texts)})

    return text_msgs
