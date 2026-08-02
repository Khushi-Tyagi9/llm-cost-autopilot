"""
Tests for src/logging/db.py

Uses a temporary database file per test, so tests never touch the real
data/requests.db and can run in full isolation/parallel safely.
"""
import pytest
import sqlite3
import tempfile
import os

import src.logging.db as db_module
from src.logging.db import init_db, log_request, _hash_prompt


@pytest.fixture
def temp_db(monkeypatch):
    """Points DB_PATH at a temporary file for the duration of one test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # init_db will create it fresh
    monkeypatch.setattr(db_module, "DB_PATH", path)
    init_db()
    yield path
    if os.path.exists(path):
        os.unlink(path)


def make_fake_response(model_id="test-model", provider="groq", cost=0.0001,
                        latency=0.5, input_tokens=10, output_tokens=20):
    class FakeResponse:
        pass
    r = FakeResponse()
    r.model_id = model_id
    r.provider = provider
    r.cost = cost
    r.latency = latency
    r.input_tokens = input_tokens
    r.output_tokens = output_tokens
    return r


class TestHashPrompt:
    def test_same_prompt_hashes_identically(self):
        assert _hash_prompt("hello world") == _hash_prompt("hello world")

    def test_different_prompts_hash_differently(self):
        assert _hash_prompt("hello") != _hash_prompt("world")

    def test_hash_does_not_contain_raw_text(self):
        prompt = "sensitive information here"
        hashed = _hash_prompt(prompt)
        assert prompt not in hashed

    def test_hash_is_reasonably_short(self):
        assert len(_hash_prompt("any prompt")) == 16


class TestInitDb:
    def test_creates_requests_table(self, temp_db):
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='requests'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_is_idempotent(self, temp_db):
        # calling init_db() twice should not raise
        init_db()
        init_db()

    def test_table_has_expected_columns(self, temp_db):
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("PRAGMA table_info(requests)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "id", "timestamp", "prompt_hash", "tier", "model_id", "provider",
            "cost", "latency", "input_tokens", "output_tokens", "verified",
            "judge_verdict", "escalated", "premium_cost", "judge_cost",
            "routing_override",
        }
        assert expected.issubset(columns)


class TestLogRequest:
    def test_logs_unverified_request(self, temp_db):
        response = make_fake_response()
        log_request(tier=1, response=response, verification=None, prompt="test prompt")

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT * FROM requests").fetchone()
        conn.close()

        assert row is not None

    def test_unverified_request_has_verified_flag_zero(self, temp_db):
        response = make_fake_response()
        log_request(tier=1, response=response, verification=None, prompt="test")

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT verified, escalated FROM requests"
        ).fetchone()
        conn.close()

        assert row[0] == 0
        assert row[1] == 0

    def test_verified_request_stores_verification_fields(self, temp_db):
        response = make_fake_response()
        verification = {
            "judge_verdict": "MATCH",
            "escalate": False,
            "premium_cost": 0.0005,
            "judge_cost": 0.00008,
        }
        log_request(tier=2, response=response, verification=verification, prompt="test")

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT verified, judge_verdict, escalated, premium_cost, judge_cost FROM requests"
        ).fetchone()
        conn.close()

        assert row[0] == 1
        assert row[1] == "MATCH"
        assert row[2] == 0
        assert row[3] == pytest.approx(0.0005)
        assert row[4] == pytest.approx(0.00008)

    def test_escalated_request_stores_escalated_flag(self, temp_db):
        response = make_fake_response()
        verification = {
            "judge_verdict": "DIVERGE",
            "escalate": True,
            "premium_cost": 0.0005,
            "judge_cost": 0.00008,
        }
        log_request(tier=3, response=response, verification=verification, prompt="test")

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT escalated FROM requests").fetchone()
        conn.close()

        assert row[0] == 1

    def test_prompt_is_stored_hashed_not_raw(self, temp_db):
        secret_prompt = "this exact text should never appear in the db"
        response = make_fake_response()
        log_request(tier=1, response=response, verification=None, prompt=secret_prompt)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT prompt_hash FROM requests").fetchone()
        conn.close()

        assert row[0] != secret_prompt
        assert secret_prompt not in row[0]

    def test_multiple_requests_all_persist(self, temp_db):
        response = make_fake_response()
        for i in range(5):
            log_request(tier=1, response=response, verification=None, prompt=f"prompt {i}")

        conn = sqlite3.connect(temp_db)
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        conn.close()

        assert count == 5
