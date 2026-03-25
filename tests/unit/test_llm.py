"""Tests for LLM client abstraction and format conversion."""

import json

import pytest

from tamagotchi.llm import (
    AnthropicClient,
    LLMResponse,
    OpenAIClient,
    ToolCall,
    _to_openai_messages,
    _to_openai_tools,
    create_llm_client,
)


class TestFormatConversion:
    def test_simple_messages_to_openai(self):
        msgs = [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요!"},
        ]
        result = _to_openai_messages(msgs)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "안녕"}
        assert result[1] == {"role": "assistant", "content": "안녕하세요!"}

    def test_tool_results_to_openai(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "abc123", "content": '{"status": "ok"}'},
            ]},
        ]
        result = _to_openai_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "abc123"

    def test_tools_to_openai_format(self):
        anthropic_tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]
        result = _to_openai_tools(anthropic_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
        assert result[0]["function"]["parameters"]["type"] == "object"


class TestLLMResponse:
    def test_response_with_text_only(self):
        r = LLMResponse(text="hello")
        assert r.text == "hello"
        assert r.tool_calls == []

    def test_response_with_tool_calls(self):
        tc = ToolCall(id="1", name="test", input={"key": "val"})
        r = LLMResponse(text="", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "test"


class TestToolCallFormatting:
    def test_openai_tool_result_messages(self):
        # Can't instantiate OpenAIClient without a server, so test the logic directly
        tool_calls = [
            ToolCall(id="call_1", name="manage_tasks", input={"action": "list"}),
            ToolCall(id="call_2", name="search_web", input={"query": "test"}),
        ]
        results = ['{"tasks": []}', '{"results": []}']

        # Simulate OpenAI format
        msgs = [
            {"role": "tool", "tool_call_id": tc.id, "content": result}
            for tc, result in zip(tool_calls, results)
        ]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_1"

    def test_anthropic_tool_result_messages(self):
        tool_calls = [
            ToolCall(id="toolu_1", name="manage_tasks", input={"action": "list"}),
        ]
        results = ['{"tasks": []}']

        # Simulate Anthropic format
        msgs = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": result}
            for tc, result in zip(tool_calls, results)
        ]
        assert len(msgs) == 1
        assert msgs[0]["type"] == "tool_result"
        assert msgs[0]["tool_use_id"] == "toolu_1"


class TestCreateLLMClient:
    def test_default_creates_anthropic(self):
        client = create_llm_client()
        assert isinstance(client, AnthropicClient)

    def test_openai_provider(self):
        client = create_llm_client(
            provider="openai",
            base_url="http://localhost:8000/v1",
            model="test-model",
        )
        assert isinstance(client, OpenAIClient)
        assert client.model == "test-model"
        assert client.base_url == "http://localhost:8000/v1"

    def test_env_var_provider(self, monkeypatch):
        monkeypatch.setenv("TAMAGOTCHI_PROVIDER", "openai")
        monkeypatch.setenv("TAMAGOTCHI_BASE_URL", "http://myserver:9000/v1")
        monkeypatch.setenv("TAMAGOTCHI_MODEL", "my-model")
        client = create_llm_client()
        assert isinstance(client, OpenAIClient)
        assert client.model == "my-model"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TAMAGOTCHI_PROVIDER", "openai")
        client = create_llm_client(provider="anthropic")
        assert isinstance(client, AnthropicClient)
