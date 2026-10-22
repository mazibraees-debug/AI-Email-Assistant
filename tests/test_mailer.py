"""Offline tests for the mailbox integration. No network access."""

from __future__ import annotations

import pytest

import assistant.mailer as mailer
from assistant.mailer import MailboxError, configured, fetch_unread, mailbox_status, send_message


def _env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for k, v in values.items():
        monkeypatch.setenv(k, v)


def test_configured_false_with_placeholders(monkeypatch: pytest.MonkeyPatch):
    _env(monkeypatch, EMAIL_ADDRESS="you@example.com", EMAIL_PASSWORD="your_app_password")
    assert configured() is False


def test_configured_true_with_real_creds(monkeypatch: pytest.MonkeyPatch):
    _env(monkeypatch, EMAIL_ADDRESS="me@example.com", EMAIL_PASSWORD="realpassword123")
    assert configured() is True


def test_configured_false_missing(monkeypatch: pytest.MonkeyPatch):
    _env(monkeypatch, EMAIL_ADDRESS="", EMAIL_PASSWORD="")
    assert configured() is False


def test_mailbox_status(monkeypatch: pytest.MonkeyPatch):
    _env(
        monkeypatch,
        EMAIL_ADDRESS="me@example.com",
        EMAIL_PASSWORD="pw",
        IMAP_HOST="imap.example.com",
        SMTP_HOST="smtp.example.com",
        SMTP_PORT="465",
    )
    status = mailbox_status()
    assert status["configured"] is True
    assert status["address"] == "me@example.com"
    assert status["imap_host"] == "imap.example.com"
    assert status["smtp_port"] == 465


class _FakeImap:
    """Full stand-in for imaplib.IMAP4_SSL returning one canned email."""

    RAW_EMAIL = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.com>\r\n"
        b"Subject: Hello there\r\n"
        b"Date: Wed, 21 Oct 2026 10:00:00 +0000\r\n"
        b"Message-ID: <abc123@example.com>\r\n"
        b"\r\n"
        b"Hi Bob,\n\nJust checking in.\n\nAlice\r\n"
    )

    def __init__(self, host, timeout=None):
        self.host = host
        self.logged_in = False
        self.selected = None
        self.readonly = True
        self.fetched = []

    def login(self, user, password):
        self.logged_in = True

    def select(self, mailbox, readonly=True):
        self.selected = mailbox
        self.readonly = readonly
        return ("OK", [b"1"])

    def search(self, charset, *criteria):
        self.criteria = criteria
        return ("OK", [b"1"])

    def fetch(self, msg_id, parts):
        self.fetched.append((msg_id, parts))
        return ("OK", [(msg_id, self.RAW_EMAIL)])

    def logout(self):
        pass


def test_fetch_unread_uses_peek_and_preserves_seen(monkeypatch: pytest.MonkeyPatch):
    _env(monkeypatch, EMAIL_ADDRESS="me@example.com", EMAIL_PASSWORD="pw")
    fake = _FakeImap("imap.example.com")
    monkeypatch.setattr(mailer.imaplib, "IMAP4_SSL", lambda host, timeout=None: fake)
    emails = fetch_unread(limit=5)
    assert len(emails) == 1
    assert emails[0].sender == "Alice <alice@example.com>"
    assert emails[0].sender_email == "alice@example.com"
    assert emails[0].subject == "Hello there"
    assert "Just checking in" in emails[0].body
    assert any("BODY.PEEK[]" in parts for _, parts in fake.fetched)
    assert not any("RFC822" in parts for _, parts in fake.fetched)
    assert fake.readonly is True


def test_fetch_unread_connection_failure_raises_mailbox_error(monkeypatch: pytest.MonkeyPatch):
    _env(monkeypatch, EMAIL_ADDRESS="me@example.com", EMAIL_PASSWORD="pw")

    def boom(host, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(mailer.imaplib, "IMAP4_SSL", boom)
    with pytest.raises(MailboxError):
        fetch_unread()


class _FakeSMTP:
    """Records the outbound message instead of sending it."""

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.started_tls = False
        self.logged_in = False
        self.sent: list[dict[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, user, password):
        self.logged_in = True

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.sent.append(
            {
                "from": from_addr,
                "to": ",".join(to_addrs or []),
                "subject": msg["Subject"],
                "in_reply_to": msg.get("In-Reply-To"),
                "references": msg.get("References"),
                "body": msg.get_content(),
            }
        )


def _monkeypatch_smtp(monkeypatch: pytest.MonkeyPatch, port=587, ssl=False):
    last: list[_FakeSMTP] = []

    class Wrapper:
        def __init__(self, host, port, timeout=None):
            inst = _FakeSMTP(host, port, timeout)
            last.append(inst)
            self._inst = inst

        def __enter__(self):
            return self._inst

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mailer.smtplib, "SMTP", Wrapper)
    if ssl:
        monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", Wrapper)
    return last


def test_send_reply_builds_headers(monkeypatch: pytest.MonkeyPatch):
    _env(
        monkeypatch,
        EMAIL_ADDRESS="me@example.com",
        EMAIL_PASSWORD="pw",
        SMTP_HOST="smtp.example.com",
        SMTP_PORT="587",
    )
    last = _monkeypatch_smtp(monkeypatch)
    from assistant.extractor import Email

    original = Email(
        sender="Alice <alice@example.com>",
        subject="Project update",
        body="Hi",
        message_id="<orig-1@example.com>",
    )
    mailer.send_reply(original, "Sure, sounds good.")
    assert len(last) == 1
    sent = last[0].sent[0]
    assert "alice@example.com" in sent["to"]
    assert sent["subject"] == "Re: Project update"
    assert sent["in_reply_to"] == "<orig-1@example.com>"
    assert "orig-1" in sent["references"]
    assert "Sure, sounds good" in sent["body"]


def test_send_reply_strips_existing_re_prefix(monkeypatch: pytest.MonkeyPatch):
    _env(
        monkeypatch,
        EMAIL_ADDRESS="me@example.com",
        EMAIL_PASSWORD="pw",
        SMTP_HOST="smtp.example.com",
        SMTP_PORT="587",
    )
    last = _monkeypatch_smtp(monkeypatch)
    from assistant.extractor import Email

    original = Email(
        sender="Alice <alice@example.com>",
        subject="Re: Project update",
        body="Hi",
        message_id="<orig-2@example.com>",
    )
    mailer.send_reply(original, "reply body")
    assert last[0].sent[0]["subject"] == "Re: Project update"


def test_send_message_generic(monkeypatch: pytest.MonkeyPatch):
    _env(
        monkeypatch,
        EMAIL_ADDRESS="me@example.com",
        EMAIL_PASSWORD="pw",
        SMTP_HOST="smtp.example.com",
        SMTP_PORT="587",
    )
    last = _monkeypatch_smtp(monkeypatch)
    send_message("friend@example.com", "Digest", "Hello digest body")
    assert last[0].sent[0]["to"] == "friend@example.com"
    assert last[0].sent[0]["subject"] == "Digest"
    assert last[0].sent[0]["body"].strip() == "Hello digest body"


def test_send_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch):
    _env(monkeypatch, EMAIL_ADDRESS="", EMAIL_PASSWORD="")
    from assistant.extractor import Email

    original = Email(sender="a@b.com", subject="s", body="b")
    with pytest.raises(MailboxError):
        mailer.send_reply(original, "body")
