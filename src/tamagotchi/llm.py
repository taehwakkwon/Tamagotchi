"""LLM client abstraction — supports Anthropic and OpenAI-compatible endpoints.

Usage:
    # Anthropic (default)
    client = create_llm_client(provider="anthropic")

    # Self-hosted via vLLM / SGLang / Ollama
    client = create_llm_client(
        provider="openai",
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B",
    )

    # Same interface for both
    response = client.chat(system_prompt, messages, tools)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # Original provider response


@dataclass
class ToolCall:
    """A single tool call from the LLM."""

    id: str
    name: str
    input: dict[str, Any]


class LLMClient:
    """Abstract base for LLM clients."""

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16384,
    ) -> LLMResponse:
        raise NotImplementedError

    def extract(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Simple single-turn call for extraction tasks (preferences, recommendations)."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, model: str):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16384,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
        )

    def extract(self, prompt: str, max_tokens: int = 1024) -> str:
        from tamagotchi.config import EXTRACTION_MODEL

        response = self.client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def build_tool_result_messages(self, tool_calls: list[ToolCall], results: list[str]) -> list[dict]:
        """Build Anthropic-format tool result messages."""
        return [
            {
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            }
            for tc, result in zip(tool_calls, results)
        ]

    def build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        """Build Anthropic-format assistant message for history."""
        return {"role": "assistant", "content": response.raw.content}


class OpenAIClient(LLMClient):
    """OpenAI-compatible API client (vLLM, SGLang, Ollama, etc.)."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
        )
        # For extraction, use the same model (no separate Haiku equivalent)
        self.extraction_model = model

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16384,
    ) -> LLMResponse:
        # Convert Anthropic-style messages to OpenAI format
        oai_messages = [{"role": "system", "content": system_prompt}]
        oai_messages.extend(_to_openai_messages(messages))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=response)

    def extract(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.extraction_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def build_tool_result_messages(self, tool_calls: list[ToolCall], results: list[str]) -> list[dict]:
        """Build OpenAI-format tool result messages."""
        return [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            for tc, result in zip(tool_calls, results)
        ]

    def build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        """Build OpenAI-format assistant message for history."""
        msg: dict[str, Any] = {"role": "assistant", "content": response.text}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
            # OpenAI requires content to be null when there are tool calls with no text
            if not response.text:
                msg["content"] = None
        return msg


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Factory function to create the right LLM client.

    Priority for each setting: explicit arg > env var > default

    Env vars:
        TAMAGOTCHI_PROVIDER: "anthropic" or "openai"
        TAMAGOTCHI_MODEL: model name/id
        TAMAGOTCHI_BASE_URL: OpenAI-compatible endpoint URL
        OPENAI_API_KEY: API key for OpenAI-compatible endpoint
    """
    from tamagotchi.config import get_chat_model

    provider = provider or os.environ.get("TAMAGOTCHI_PROVIDER", "anthropic")
    provider = provider.lower().strip()

    if provider == "openai":
        base_url = base_url or os.environ.get("TAMAGOTCHI_BASE_URL", "http://localhost:8000/v1")
        resolved_model = model or os.environ.get("TAMAGOTCHI_MODEL", "default")
        return OpenAIClient(model=resolved_model, base_url=base_url, api_key=api_key)
    else:
        resolved_model = get_chat_model(model)
        return AnthropicClient(model=resolved_model)


# ── Format conversion helpers ──


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style message history to OpenAI format."""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "assistant":
                # Anthropic content blocks → OpenAI assistant message
                texts = []
                tool_calls_oai = []
                for block in content:
                    if hasattr(block, "type"):
                        # Anthropic SDK objects
                        if block.type == "text":
                            texts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls_oai.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input, ensure_ascii=False),
                                },
                            })
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block["text"])

                oai_msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
                if tool_calls_oai:
                    oai_msg["tool_calls"] = tool_calls_oai
                result.append(oai_msg)

            elif role == "user":
                # Could be tool results (Anthropic format) or regular content
                if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                    # Convert Anthropic tool_result to OpenAI tool messages
                    for block in content:
                        result.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        })
                else:
                    # Regular user content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            texts.append(block["text"])
                        elif isinstance(block, str):
                            texts.append(block)
                    if texts:
                        result.append({"role": "user", "content": " ".join(texts)})
        else:
            result.append({"role": role, "content": str(content)})

    return result


def _to_openai_tools(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        for tool in anthropic_tools
    ]
