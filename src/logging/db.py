"""
SQLite logging for every routed request. This is the single source of
truth the dashboard (Grafana) will read from.
"""
import sqlite3
import hashlib
import time
from contextlib import contextmanager

DB_PATH = "data/requests.db"


def _hash_prompt(prompt: str) -> str:
    """We hash prompts rather than storing raw text, for basic privacy."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                prompt_hash TEXT NOT NULL,
                tier INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                cost REAL NOT NULL,
                latency REAL NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                judge_verdict TEXT,
                escalated INTEGER NOT NULL DEFAULT 0,
                premium_cost REAL,
                judge_cost REAL
            )
        """)
        conn.commit()


def log_request(tier: int, response, verification: dict | None, prompt: str):
    """
    response: a Response object from send_request()
    verification: dict from verify_response(), or None if not verified
    """
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO requests (
                timestamp, prompt_hash, tier, model_id, provider,
                cost, latency, input_tokens, output_tokens,
                verified, judge_verdict, escalated, premium_cost, judge_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            _hash_prompt(prompt),
            tier,
            response.model_id,
            response.provider,
            response.cost,
            response.latency,
            response.input_tokens,
            response.output_tokens,
            1 if verification else 0,
            verification["judge_verdict"] if verification else None,
            1 if (verification and verification["escalate"]) else 0,
            verification["premium_cost"] if verification else None,
            verification["judge_cost"] if verification else None,
        ))
        conn.commit()