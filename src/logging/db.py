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
                judge_cost REAL,
                routing_override TEXT,
                classifier_confidence REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                prompt_hash TEXT PRIMARY KEY,
                response_text TEXT NOT NULL,
                tier INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                original_cost REAL NOT NULL,
                cached_at REAL NOT NULL
            )
        """)
        cursor = conn.execute("PRAGMA table_info(requests)")
        columns = {row[1] for row in cursor.fetchall()}

        if "classifier_confidence" not in columns:
           conn.execute("""
           ALTER TABLE requests
           ADD COLUMN classifier_confidence REAL
        """)
        conn.commit()


def log_request(tier: int, response, verification: dict | None, prompt: str, routing_override: str | None = None, classifier_confidence: float | None = None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO requests (
                timestamp, prompt_hash, tier, model_id, provider,
                cost, latency, input_tokens, output_tokens,
                verified, judge_verdict, escalated, premium_cost, judge_cost,
                routing_override, classifier_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), _hash_prompt(prompt), tier, response.model_id, response.provider,
            response.cost, response.latency, response.input_tokens, response.output_tokens,
            1 if verification else 0,
            verification["judge_verdict"] if verification else None,
            1 if (verification and verification["escalate"]) else 0,
            verification["premium_cost"] if verification else None,
            verification["judge_cost"] if verification else None,
            routing_override,
            classifier_confidence,
        ))
        conn.commit()

def get_cached_response(prompt: str):
    """Returns a cached response dict if this exact prompt was seen
    before, or None if it's not cached."""
    prompt_hash = _hash_prompt(prompt)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT response_text, tier, model_id, provider, original_cost FROM response_cache WHERE prompt_hash = ?",
            (prompt_hash,)
        ).fetchone()
    if row is None:
        return None
    return {
        "text": row[0],
        "tier": row[1],
        "model_id": row[2],
        "provider": row[3],
        "original_cost": row[4],
    }


def store_cached_response(prompt: str, text: str, tier: int, model_id: str, provider: str, cost: float):
    prompt_hash = _hash_prompt(prompt)
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO response_cache
            (prompt_hash, response_text, tier, model_id, provider, original_cost, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (prompt_hash, text, tier, model_id, provider, cost, time.time()))
        conn.commit()