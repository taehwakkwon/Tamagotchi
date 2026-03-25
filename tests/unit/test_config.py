"""Tests for configuration and model selection."""

import os

from tamagotchi.config import MODELS, get_chat_model, get_model_display_name


class TestModelSelection:
    def test_default_model(self):
        model = get_chat_model()
        assert model == MODELS["sonnet"]["id"]

    def test_select_haiku(self):
        model = get_chat_model("haiku")
        assert model == MODELS["haiku"]["id"]

    def test_select_opus(self):
        model = get_chat_model("opus")
        assert model == MODELS["opus"]["id"]

    def test_case_insensitive(self):
        assert get_chat_model("Haiku") == MODELS["haiku"]["id"]
        assert get_chat_model("SONNET") == MODELS["sonnet"]["id"]

    def test_full_model_id_passthrough(self):
        full_id = "claude-sonnet-4-20250514"
        assert get_chat_model(full_id) == full_id

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("TAMAGOTCHI_MODEL", "haiku")
        assert get_chat_model() == MODELS["haiku"]["id"]

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TAMAGOTCHI_MODEL", "haiku")
        assert get_chat_model("opus") == MODELS["opus"]["id"]

    def test_unknown_falls_back_to_default(self):
        model = get_chat_model("unknown_model")
        assert model == MODELS["sonnet"]["id"]

    def test_display_name(self):
        assert get_model_display_name(MODELS["sonnet"]["id"]) == "Sonnet"
        assert get_model_display_name(MODELS["haiku"]["id"]) == "Haiku"
        assert get_model_display_name("unknown-id") == "unknown-id"
