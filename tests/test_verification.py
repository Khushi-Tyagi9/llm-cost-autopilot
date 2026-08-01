"""
Tests for src/verification/judge.py, llm_judge.py, and scheduler.py

Uses mocking for all model calls - these tests never hit a real API,
so they run instantly, cost nothing, and don't depend on network
availability or provider rate limits.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.verification.scheduler import should_verify
from src.verification.llm_judge import llm_judge_verdict
from src.verification.judge import verify_response
from src.models.response import Response


def make_fake_response(text="mocked response", cost=0.0001):
    return Response(
        text=text,
        input_tokens=10,
        output_tokens=10,
        latency=0.1,
        cost=cost,
        model_id="fake-model",
        provider="fake-provider",
    )


class TestShouldVerify:
    def test_sample_rate_zero_never_verifies(self):
        results = [should_verify(0.0) for _ in range(100)]
        assert all(r is False for r in results)

    def test_sample_rate_one_always_verifies(self):
        results = [should_verify(1.0) for _ in range(100)]
        assert all(r is True for r in results)

    def test_sample_rate_produces_roughly_expected_ratio(self):
        # Statistical test - not exact, but should be in a sane range
        results = [should_verify(0.5) for _ in range(2000)]
        true_ratio = sum(results) / len(results)
        assert 0.4 < true_ratio < 0.6


class TestLlmJudgeVerdict:
    @patch("src.verification.llm_judge.send_request")
    def test_match_verdict_does_not_escalate(self, mock_send):
        mock_send.return_value = make_fake_response(text="MATCH")
        result = llm_judge_verdict("some prompt", "answer a", "answer b")
        assert result["escalate"] is False

    @patch("src.verification.llm_judge.send_request")
    def test_diverge_verdict_escalates(self, mock_send):
        mock_send.return_value = make_fake_response(text="DIVERGE")
        result = llm_judge_verdict("some prompt", "answer a", "answer b")
        assert result["escalate"] is True

    @patch("src.verification.llm_judge.send_request")
    def test_verdict_is_case_insensitive(self, mock_send):
        mock_send.return_value = make_fake_response(text="match")
        result = llm_judge_verdict("prompt", "a", "b")
        assert result["escalate"] is False

    @patch("src.verification.llm_judge.send_request")
    def test_judge_cost_is_passed_through(self, mock_send):
        mock_send.return_value = make_fake_response(text="MATCH", cost=0.00007)
        result = llm_judge_verdict("prompt", "a", "b")
        assert result["judge_cost"] == 0.00007


class TestVerifyResponse:
    @patch("src.verification.judge.llm_judge_verdict")
    @patch("src.verification.judge.send_request")
    def test_returns_expected_keys(self, mock_send, mock_judge):
        mock_send.return_value = make_fake_response(text="premium answer")
        mock_judge.return_value = {
            "verdict": "MATCH",
            "escalate": False,
            "judge_cost": 0.00005,
        }

        result = verify_response("prompt", "cheap answer", MagicMock())

        assert "premium_text" in result
        assert "escalate" in result
        assert "judge_cost" in result
        assert result["escalate"] is False

    @patch("src.verification.judge.llm_judge_verdict")
    @patch("src.verification.judge.send_request")
    def test_escalate_flows_through_from_judge(self, mock_send, mock_judge):
        mock_send.return_value = make_fake_response(text="premium answer")
        mock_judge.return_value = {
            "verdict": "DIVERGE",
            "escalate": True,
            "judge_cost": 0.00005,
        }

        result = verify_response("prompt", "cheap answer", MagicMock())

        assert result["escalate"] is True
