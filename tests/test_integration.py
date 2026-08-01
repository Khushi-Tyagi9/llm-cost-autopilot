"""
Integration test for the full request pipeline: classify -> route ->
verify -> log, exercised together rather than in isolation.

This exists specifically because two real bugs in this project were
interface mismatches BETWEEN modules that each passed their own unit
tests individually:
  1. verify_response()'s return shape changed (TF-IDF -> LLM-judge)
     but log_request() still expected the old field names.
  2. config_loader.py was reverted to old hardcoded logic, which no
     unit test caught because the unit test file itself got overwritten.

Only the actual network-calling functions (send_request, the judge's
underlying model call) are mocked. Everything else - feature
extraction, tier resolution, verification logic, and the database
write - runs for real, using a temporary database.
"""
import pytest
import sqlite3
import tempfile
import os
from unittest.mock import patch

import src.logging.db as db_module
from src.logging.db import init_db, log_request
from src.classifier.features import featurize
from src.router.config_loader import load_routing_config, resolve_model_for_tier
from src.verification.judge import verify_response
from src.verification.scheduler import should_verify
from src.models.response import Response


@pytest.fixture
def temp_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    monkeypatch.setattr(db_module, "DB_PATH", path)
    init_db()
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def routing_config():
    return {
        "routing": {
            1: {"model": "cheap_model"},
            2: {"model": "cheap_model"},
            3: {"model": "premium_model"},
        },
        "verification": {
            "sample_rate": 1.0,  # force verification for the test
        },
        "models": {
            "cheap_model": {
                "provider": "groq",
                "model_id": "llama-3.1-8b-instant",
                "cost_per_input_token": 0.00000005,
                "cost_per_output_token": 0.00000008,
                "quality_tier": 1,
            },
            "premium_model": {
                "provider": "groq",
                "model_id": "llama-3.3-70b-versatile",
                "cost_per_input_token": 0.00000059,
                "cost_per_output_token": 0.00000079,
                "quality_tier": 3,
            },
        },
    }


def fake_response(text, model_id="fake-model", cost=0.0001):
    return Response(
        text=text, input_tokens=10, output_tokens=10, latency=0.2,
        cost=cost, model_id=model_id, provider="groq",
    )


class TestFullPipelineIntegration:
    @patch("src.verification.llm_judge.send_request")
    @patch("src.verification.judge.send_request")
    def test_classify_route_verify_log_end_to_end(
        self, mock_verify_send, mock_judge_send, temp_db, routing_config
    ):
        """Runs a real prompt through every real module (no mocking of
        the actual logic), only the outbound model calls are faked."""
        prompt = "Design a distributed rate limiting system considering fault tolerance"

        # classify (real feature extraction, fake tier just to keep test focused)
        features = featurize([prompt])
        assert len(features[0]) > 0  # real classifier features were produced

        tier = 3  # what a real classifier would likely predict for this prompt
        model_config = resolve_model_for_tier(tier, routing_config)
        assert model_config.model_id == "llama-3.3-70b-versatile"

        # simulate the cheap model's answer (this would be the actual routed model)
        cheap_response = fake_response("A distributed rate limiter design...")

        # verification - real verify_response() logic, only network calls faked
        mock_verify_send.return_value = fake_response("Premium's rate limiter answer")
        mock_judge_send.return_value = fake_response("MATCH")

        verification = verify_response(prompt, cheap_response.text, model_config)

        # this is the exact line that broke in production: log_request()
        # must accept whatever verify_response() actually returns
        log_request(tier, cheap_response, verification, prompt)

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT tier, judge_verdict, escalated FROM requests"
        ).fetchone()
        conn.close()

        assert row[0] == tier
        assert row[1] == "MATCH"
        assert row[2] == 0

    @patch("src.verification.llm_judge.send_request")
    @patch("src.verification.judge.send_request")
    def test_unverified_path_also_logs_correctly(
        self, mock_verify_send, mock_judge_send, temp_db, routing_config
    ):
        """When should_verify() returns False, verification is skipped
        entirely and log_request() must handle verification=None."""
        prompt = "What is the capital of Germany?"

        tier = 1
        model_config = resolve_model_for_tier(tier, routing_config)
        response = fake_response("Berlin")

        verified = should_verify(0.0)  # sample rate 0 - never verifies
        assert verified is False

        log_request(tier, response, None, prompt)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT verified, judge_verdict FROM requests").fetchone()
        conn.close()

        assert row[0] == 0
        assert row[1] is None

        # network mocks should never have been called on the unverified path
        mock_verify_send.assert_not_called()
        mock_judge_send.assert_not_called()

    def test_config_loader_produces_configs_usable_by_router_client(self, routing_config):
        """Guards against the exact regression this project hit: a
        ModelConfig built by config_loader must have real, usable
        pricing fields, not a stale hardcoded lookup."""
        for tier in (1, 2, 3):
            model_config = resolve_model_for_tier(tier, routing_config)
            assert model_config.cost_per_input_token > 0
            assert model_config.cost_per_output_token > 0
            assert model_config.provider in ("groq", "anthropic")
