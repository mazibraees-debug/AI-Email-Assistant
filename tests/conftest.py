"""Shared pytest fixtures for the API test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from assistant.pipeline import EmailAssistant
from assistant.storage import reset_db
from assistant.testing import HeuristicLLM


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Give every test its own isolated SQLite database."""
    return reset_db(str(tmp_path / "t.db"))


@pytest.fixture
def stub_assistant(tmp_db, monkeypatch: pytest.MonkeyPatch):
    """Force the offline deterministic stub instead of a real LLM provider."""
    monkeypatch.setattr(EmailAssistant, "_default_llm", staticmethod(lambda: HeuristicLLM()))


@pytest.fixture
def client(stub_assistant):
    """A TestClient whose lifespan boots the stub-backed assistant."""
    with TestClient(app) as tc:
        yield tc
