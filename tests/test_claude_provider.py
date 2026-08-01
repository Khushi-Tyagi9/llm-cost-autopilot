"""
Tests for src/models/providers/claude_provider.py and the router_client's
dispatch to it.

The project's multi-provider claim rests partly on this adapter, but it
had zero test coverage and had never actually been called end-to-end.
These tests mock the Anthropic SDK entirely, so they verify the
adapter's logic (response parsing, token extraction, error handling)
without requiring a real API key or making a real network call.

This does not prove the adapter works against the real Anthropic API -
only a live call can prove that. It proves the adapter's own logic is
correct given a well-formed SDK response, and that a missing API key
fails clearly instead of silently.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

import src.models.providers.claude_provider as claude_provider_module
from src.models.providers.claude_provider import call_claude, _get_client
from src.models.router_client import send_request
from src.models.config import ModelConfig


def make_fake_anthropic_message(text="a claude answer", input_tokens=12, output_tokens=8):
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = text

    fake_message = MagicMock()
    fake_message.content = [fake_block]
    fake_message.usage.input_tokens = input_tokens
    fake_message.usage.output_tokens = output_tokens
    return fake_message


@pytest.fixture(autouse=True)
def reset_client_cache():
    """The provider module caches a client in a module-level global -
    reset it between tests so mocks/env changes in one test don't leak
    into another."""
    claude_provider_module._client = None
    yield
    claude_provider_module._client = None


class TestGetClient:
    def test_raises_clear_error_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            _get_client()

    @patch("src.models.providers.claude_provider.Anthropic")
    def test_creates_client_when_key_present(self, mock_anthropic_cls, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-testing")
        _get_client()
        mock_anthropic_cls.assert_called_once_with(api_key="fake-key-for-testing")

    @patch("src.models.providers.claude_provider.Anthropic")
    def test_reuses_cached_client_on_second_call(self, mock_anthropic_cls, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-testing")
        _get_client()
        _get_client()
        assert mock_anthropic_cls.call_count == 1


class TestCallClaude:
    @patch("src.models.providers.claude_provider._get_client")
    def test_extracts_text_from_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_fake_anthropic_message(text="hello from claude")
        mock_get_client.return_value = mock_client

        text, input_tokens, output_tokens, latency = call_claude("a prompt", "claude-sonnet-4-6")

        assert text == "hello from claude"

    @patch("src.models.providers.claude_provider._get_client")
    def test_extracts_token_counts(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_fake_anthropic_message(
            input_tokens=25, output_tokens=40
        )
        mock_get_client.return_value = mock_client

        _, input_tokens, output_tokens, _ = call_claude("a prompt", "claude-sonnet-4-6")

        assert input_tokens == 25
        assert output_tokens == 40

    @patch("src.models.providers.claude_provider._get_client")
    def test_returns_nonnegative_latency(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_fake_anthropic_message()
        mock_get_client.return_value = mock_client

        _, _, _, latency = call_claude("a prompt", "claude-sonnet-4-6")

        assert latency >= 0

    @patch("src.models.providers.claude_provider._get_client")
    def test_passes_model_id_and_prompt_to_sdk(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_fake_anthropic_message()
        mock_get_client.return_value = mock_client

        call_claude("what is the capital of Spain", "claude-sonnet-4-6")

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["messages"][0]["content"] == "what is the capital of Spain"

    @patch("src.models.providers.claude_provider._get_client")
    def test_ignores_non_text_content_blocks(self, mock_get_client):
        """Anthropic responses can include non-text blocks (e.g. tool
        use); the adapter should only concatenate text blocks."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "the actual answer"

        other_block = MagicMock()
        other_block.type = "tool_use"

        fake_message = MagicMock()
        fake_message.content = [other_block, text_block]
        fake_message.usage.input_tokens = 5
        fake_message.usage.output_tokens = 5

        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_message
        mock_get_client.return_value = mock_client

        text, _, _, _ = call_claude("prompt", "claude-sonnet-4-6")

        assert text == "the actual answer"


class TestRouterClientDispatchesToClaudeCorrectly:
    """Confirms send_request() correctly routes to the Claude adapter
    when given an anthropic ModelConfig, not just Groq."""

    @patch("src.models.router_client.call_claude")
    def test_anthropic_provider_calls_claude_adapter(self, mock_call_claude):
        mock_call_claude.return_value = ("claude says hi", 10, 10, 0.5)

        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            cost_per_input_token=0.000003,
            cost_per_output_token=0.000015,
            quality_tier=3,
        )

        response = send_request("a prompt", config)

        assert response.text == "claude says hi"
        assert response.provider == "anthropic"
        mock_call_claude.assert_called_once()

    @patch("src.models.router_client.call_claude")
    def test_anthropic_response_cost_is_calculated_correctly(self, mock_call_claude):
        mock_call_claude.return_value = ("answer", 1000, 500, 0.5)

        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            cost_per_input_token=0.000003,
            cost_per_output_token=0.000015,
            quality_tier=3,
        )

        response = send_request("a prompt", config)

        expected_cost = (1000 * 0.000003) + (500 * 0.000015)
        assert response.cost == pytest.approx(expected_cost)
