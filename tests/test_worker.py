"""Offline tests for the background worker. No network access."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import worker.scheduler as ws
from assistant.extractor import Email
from worker.scheduler import DigestJob, TriageWorker, run_digest_once, run_triage_once


def _env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for k, v in values.items():
        monkeypatch.setenv(k, v)


class _FakeAssistant:
    def __init__(self, emails):
        self.emails = emails
        self.calls = []

    def process_inbox_detailed(self, emails):
        self.calls.append(emails)
        results = []
        for e in emails:
            from assistant.schemas import Analysis, Category

            analysis = Analysis(
                category=Category.WORK,
                priority=4 if "urgent" in (e.subject + " " + e.body).lower() else 2,
                priority_reason="stub",
                summary="stub summary",
                action_items=[],
                needs_reply=True,
            )
            results.append(SimpleNamespace(email=e, analysis=analysis))
        return results

    def digest(self):
        from assistant.schemas import Digest

        return Digest(
            generated_at="2026-10-21T08:00:00+00:00",
            email_count=len(self.emails),
            urgent_count=1,
            needs_reply_count=1,
            top_emails=[],
            action_items=[],
            by_category={"Work": len(self.emails)},
            text="digest text",
        )


def _monkeypatch_fetch(monkeypatch: pytest.MonkeyPatch, emails):
    import worker.scheduler as ws

    def fake_fetch(limit=10, **kwargs):
        return list(emails)

    monkeypatch.setattr(ws, "configured", lambda: True)
    monkeypatch.setattr(ws, "fetch_unread", fake_fetch)


def test_run_once_processes_and_summarizes(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import assistant.storage as storage

    storage.reset_db(str(tmp_path / "worker.db"))
    _env(monkeypatch, EMAIL_ADDRESS="me@example.com", EMAIL_PASSWORD="pw")
    emails = [
        Email(sender="a@b.com", subject="Urgent: outage", body="production is down"),
        Email(sender="b@b.com", subject="Newsletter", body="weekly digest"),
    ]
    _monkeypatch_fetch(monkeypatch, emails)
    worker = TriageWorker(assistant=_FakeAssistant(emails))
    summary = worker.run_once()
    assert summary["status"] == "ok"
    assert summary["processed"] == 2
    assert summary["urgent"] == 1


def test_run_once_skips_when_not_configured(monkeypatch: pytest.MonkeyPatch):
    import assistant.mailer as mailer

    monkeypatch.setattr(mailer, "configured", lambda: False)
    worker = TriageWorker(assistant=_FakeAssistant([]))
    summary = worker.run_once()
    assert summary["status"] == "skipped"
    assert summary["processed"] == 0


def test_digest_does_not_send_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import assistant.storage as storage
    import worker.scheduler as ws

    storage.reset_db(str(tmp_path / "digest.db"))
    monkeypatch.setattr(ws, "configured", lambda: True)
    sent = []

    def fake_send(to, subject, body, cc=None):
        sent.append((to, subject, body))

    monkeypatch.setattr(ws, "send_message", fake_send)
    monkeypatch.setattr(ws, "SEND_DIGEST", False)
    job = DigestJob(assistant=_FakeAssistant([]))
    result = job.run_once()
    assert result["sent"] is False
    assert sent == []


def test_digest_sends_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path):
    sent = []

    def fake_send(to, subject, body, cc=None):
        sent.append((to, subject, body))

    monkeypatch.setattr(ws, "configured", lambda: True)
    monkeypatch.setattr(ws, "send_message", fake_send)
    monkeypatch.setattr(ws, "SEND_DIGEST", True)
    monkeypatch.setattr(ws, "DIGEST_RECIPIENT", "me@example.com")
    job = DigestJob(assistant=_FakeAssistant([]))
    result = job.run_once()
    assert result["sent"] is True
    assert len(sent) == 1


def test_run_triage_once_wrapper(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import assistant.storage as storage
    import worker.scheduler as ws

    storage.reset_db(str(tmp_path / "wrap.db"))
    _env(monkeypatch, EMAIL_ADDRESS="me@example.com", EMAIL_PASSWORD="pw")
    monkeypatch.setattr(ws, "configured", lambda: True)
    emails = [Email(sender="a@b.com", subject="x", body="y")]
    _monkeypatch_fetch(monkeypatch, emails)
    summary = run_triage_once()
    assert summary["status"] == "ok"
    assert summary["processed"] == 1


def test_run_digest_once_wrapper(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import assistant.storage as storage

    storage.reset_db(str(tmp_path / "wrap2.db"))
    result = run_digest_once()
    assert result["status"] == "ok"
