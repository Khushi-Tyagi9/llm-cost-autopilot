"""
Tests for src/models/providers/openai_provider.py and the router_client's
dispatch to it.

Same approach and same limitation as test_claude_provider.py: these
tests mock the OpenAI SDK entirely, so they verify the adapter's own
logic (response parsing, token extraction, error handling) without a
real API key or network call. They do NOT prove the adapter works
against the real OpenAI API - only a live call can prove that.
"""
import pytest
from unittest.mock import patch, MagicMock

import src.models.providers.openai_provider as openai_provider_module
from src.models.providers.openai_provider import call_openai, _get_client
from src.models.router_client import send_request
from src.models.config import ModelConfig


def make_fake_openai_completion(text="an openai answer", prompt_tokens=15, completion_tokens=10):
    fake_message = MagicMock()
    fake_message.content = text

    fake_choice = MagicMock()
    fake_choice.message = fake_message

    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]
    fake_completion.usage.prompt_tokens = prompt_tokens
    fake_completion.usage.completion_tokens = completion_tokens
    return fake_completion


@pytest.fixture(autouse=True)
def reset_client_cache():
    openai_provider_module._client = None
    yield
    openai_provider_module._client = None


class TestGetClient:
    def test_raises_clear_error_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            _get_client()

    @patch("src.models.providers.openai_provider.OpenAI")
    def test_creates_client_when_key_present(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        _get_client()
        mock_openai_cls.assert_called_once_with(api_key="fake-key-for-testing")

    @patch("src.models.providers.openai_provider.OpenAI")
    def test_reuses_cached_client_on_second_call(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        _get_client()
        _get_client()
        assert mock_openai_cls.call_count == 1


class TestCallOpenai:
    @patch("src.models.providers.openai_provider._get_client")
    def test_extracts_text_from_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_fake_openai_completion(
            text="hello from gpt"
        )
        mock_get_client.return_value = mock_client

        text, _, _, _ = call_openai("a prompt", "gpt-4o")

        assert text == "hello from gpt"

    @patch("src.models.providers.openai_provider._get_client")
    def test_extracts_token_counts(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_fake_openai_completion(
            prompt_tokens=30, completion_tokens=45
        )
        mock_get_client.return_value = mock_client

        _, input_tokens, output_tokens, _ = call_openai("a prompt", "gpt-4o")

        assert input_tokens == 30
        assert output_tokens == 45

    @patch("src.models.providers.openai_provider._get_client")
    def test_returns_nonnegative_latency(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_fake_openai_completion()
        mock_get_client.return_value = mock_client

        _, _, _, latency = call_openai("a prompt", "gpt-4o")

        assert latency >= 0

    @patch("src.models.providers.openai_provider._get_client")
    def test_passes_model_id_and_prompt_to_sdk(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_fake_openai_completion()
        mock_get_client.return_value = mock_client

        call_openai("what is the capital of Italy", "gpt-4o")

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["messages"][0]["content"] == "what is the capital of Italy"


class TestRouterClientDispatchesToOpenaiCorrectly:
    @patch("src.models.router_client.call_openai")
    def test_openai_provider_calls_openai_adapter(self, mock_call_openai):
        mock_call_openai.return_value = ("gpt says hi", 10, 10, 0.5)

        config = ModelConfig(
            provider="openai",
            model_id="gpt-4o",
            cost_per_input_token=0.0000025,
            cost_per_output_token=0.00001,
            quality_tier=3,
        )

        response = send_request("a prompt", config)

        assert response.text == "gpt says hi"
        assert response.provider == "openai"
        mock_call_openai.assert_called_once()

    @patch("src.models.router_client.call_openai")
    def test_openai_response_cost_is_calculated_correctly(self, mock_call_openai):
        mock_call_openai.return_value = ("answer", 1000, 500, 0.5)

        config = ModelConfig(
            provider="openai",
            model_id="gpt-4o",
            cost_per_input_token=0.0000025,
            cost_per_output_token=0.00001,
            quality_tier=3,
        )

        response = send_request("a prompt", config)

        expected_cost = (1000 * 0.0000025) + (500 * 0.00001)
        assert response.cost == pytest.approx(expected_cost)
