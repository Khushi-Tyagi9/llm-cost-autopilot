"""
Tests for src/api/main.py

All model calls, verification, and logging are mocked so these tests
run instantly, cost nothing, and never touch the real database or a
real provider. This is the layer real users actually hit, and it had
zero coverage before this file - these tests target the two failure
modes already seen in this project: schema/field mismatches, and
unhandled provider errors surfacing as raw 500s.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from groq import RateLimitError

from src.api.main import app
from src.models.response import Response


client = TestClient(app)


def make_fake_response(text="a test answer", model_id="llama-3.1-8b-instant",
                        provider="groq", cost=0.0001, latency=0.3):
    return Response(
        text=text,
        input_tokens=10,
        output_tokens=15,
        latency=latency,
        cost=cost,
        model_id=model_id,
        provider=provider,
    )


def make_fake_rate_limit_error():
    """RateLimitError needs a response object; a MagicMock stands in fine
    since the API only needs to catch the exception type, not read it."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    return RateLimitError(
        message="rate limited",
        response=mock_response,
        body=None,
    )


class TestRoot:
    def test_root_returns_status(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()


class TestCompletions:
    @patch("src.api.main.get_cached_response", return_value=None)
    @patch("src.api.main.store_cached_response")
    @patch("src.api.main.log_request")
    @patch("src.api.main.verify_response")
    @patch("src.api.main.should_verify", return_value=True)
    @patch("src.api.main.send_request")
    @patch("src.api.main.predict_tier", return_value=2)
    def test_completion_response_does_not_wait_for_verification(
    self,
    mock_predict,
    mock_send,
    mock_should_verify,
    mock_verify,
    mock_log,
    mock_store_cache,
    mock_get_cache,
):
        """Verification now runs as a background task after the response
        is sent, so the immediate API response never carries verified/
        escalated data - those are only available via /v1/stats or the
        database, not synchronously. This is the correct, intended
        behavior for true async verification."""
        mock_send.return_value = make_fake_response()
        mock_verify.return_value = {
            "premium_text": "premium answer",
            "premium_cost": 0.0005,
            "judge_verdict": "DIVERGE",
            "judge_cost": 0.00008,
            "escalate": True,
        }

        response = client.post("/v1/completions", json={"prompt": "Summarize this"})

        assert response.status_code == 200
        body = response.json()
        assert body["verified"] is False
        assert body["escalated"] is None

    def test_empty_prompt_returns_400(self):
        response = client.post("/v1/completions", json={"prompt": ""})
        assert response.status_code == 400

    def test_whitespace_only_prompt_returns_400(self):
        response = client.post("/v1/completions", json={"prompt": "   "})
        assert response.status_code == 400

    def test_missing_prompt_field_returns_422(self):
        response = client.post("/v1/completions", json={})
        assert response.status_code == 422

    @patch("src.api.main.get_cached_response", return_value=None)
    @patch("src.api.main.predict_tier", return_value=1)
    @patch("src.api.main.send_request")
    def test_provider_rate_limit_returns_429_not_500(self, mock_send, mock_predict, mock_get_cache,):
        mock_send.side_effect = make_fake_rate_limit_error()

        response = client.post("/v1/completions", json={"prompt": "test prompt"})

        assert response.status_code == 429
        assert "retry" in response.json()["detail"].lower()

    @patch("src.api.main.get_cached_response", return_value=None)
    @patch("src.api.main.store_cached_response")
    @patch("src.api.main.log_request")
    @patch("src.api.main.should_verify", return_value=True)
    @patch("src.api.main.verify_response")
    @patch("src.api.main.send_request")
    @patch("src.api.main.predict_tier", return_value=1)
    def test_verification_rate_limit_does_not_fail_whole_request(
    self,
    mock_predict,
    mock_send,
    mock_verify,
    mock_should_verify,
    mock_log,
    mock_store_cache,
    mock_get_cache,
):
    
        """If only the verification call hits a rate limit, the primary
        answer should still be returned rather than failing the request."""
        mock_send.return_value = make_fake_response()
        mock_verify.side_effect = make_fake_rate_limit_error()

        response = client.post("/v1/completions", json={"prompt": "test prompt"})

        assert response.status_code == 200
        assert response.json()["verified"] is False


class TestModels:
    def test_list_models_returns_200(self):
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)


class TestStats:
    def test_stats_returns_expected_shape(self):
        response = client.get("/v1/stats")
        assert response.status_code == 200
        body = response.json()
        assert "total_requests" in body
        assert "total_cost" in body
        assert "avg_cost_per_request" in body
        assert "total_escalated" in body
