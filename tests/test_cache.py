import pytest
import sqlite3
import tempfile
import os

import src.logging.db as db_module
from src.logging.db import init_db, get_cached_response, store_cached_response


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


def test_uncached_prompt_returns_none(temp_db):
    assert get_cached_response("never seen this before") is None


def test_stored_prompt_can_be_retrieved(temp_db):
    store_cached_response("what is 2+2", "4", tier=1, model_id="test-model", provider="groq", cost=0.001)
    result = get_cached_response("what is 2+2")
    assert result is not None
    assert result["text"] == "4"
    assert result["tier"] == 1


def test_cache_is_prompt_specific(temp_db):
    store_cached_response("prompt A", "answer A", tier=1, model_id="m", provider="groq", cost=0.001)
    assert get_cached_response("prompt B") is None