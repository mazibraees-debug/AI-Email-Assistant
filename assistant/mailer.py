"""Optional: connect to a real mailbox (IMAP to read, SMTP to send).

Security notes
--------------
* IMAP fetches use ``BODY.PEEK[]`` and ``readonly=True`` by default so the
  ``\\Seen`` flag is never flipped without the caller explicitly opting in.
* Passwords and full message bodies are never logged.
* Nothing is ever sent automatically: :func:`send_reply` and
  :func:`send_message` require an explicit call (human-in-the-loop).
"""

from __future__ import annotations

import email as email_lib
import imaplib
import os
import smtplib
import time
from contextlib import suppress
from email import policy
from email.message import EmailMessage
from typing import Any

from .extractor import Email, from_message
from .observability import get_logger

log = get_logger("mailer")


class MailboxError(RuntimeError):
    """Raised when a mailbox operation fails."""


def configured() -> bool:
    addr = os.getenv("EMAIL_ADDRESS", "").strip()
    pwd = os.getenv("EMAIL_PASSWORD", "").strip()
    return bool(addr and pwd and addr != "you@example.com" and pwd != "your_app_password")


def mailbox_status() -> dict[str, Any]:
    """Describe mailbox configuration without opening a connection."""
    addr = os.getenv("EMAIL_ADDRESS", "").strip()
    return {
        "configured": configured(),
        "address": addr or None,
        "imap_host": os.getenv("IMAP_HOST", "imap.gmail.com"),
        "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587") or 587),
    }


def _connect_imap() -> imaplib.IMAP4_SSL:
    host = os.getenv("IMAP_HOST", "imap.gmail.com")
    try:
        imap = imaplib.IMAP4_SSL(host, timeout=30)
    except (OSError, imaplib.IMAP4.error) as exc:
        raise MailboxError(f"IMAP connection failed: {exc}") from exc
    try:
        imap.login(os.environ["EMAIL_ADDRESS"], os.environ["EMAIL_PASSWORD"])
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            imap.logout()
        raise MailboxError(f"IMAP login failed: {exc}") from exc
    return imap


def fetch_unread(
    limit: int = 10,
    mailbox: str = "INBOX",
    since_days: int | None = None,
    mark_as_read: bool = False,
) -> list[Email]:
    """Fetch unseen emails, newest first.

    Uses ``BODY.PEEK[]`` so the unread flag is preserved unless
    ``mark_as_read=True`` is passed explicitly.
    """
    if not configured():
        raise MailboxError("Mailbox is not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in .env")
    imap = _connect_imap()
    try:
        imap.select(mailbox, readonly=not mark_as_read)
        search_criteria = ["UNSEEN"]
        if since_days:
            date_str = time.strftime("%d-%b-%Y")
            search_criteria = ["SINCE", date_str, "UNSEEN"]
        typ, data = imap.search(None, *search_criteria)
        if typ != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()
        # newest first
        ids = ids[-limit:]
        out: list[Email] = []
        for msg_id in reversed(ids):
            fetch_cmd = "(RFC822)" if mark_as_read else "(BODY.PEEK[])"
            typ, msg_data = imap.fetch(msg_id, fetch_cmd)
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            if isinstance(raw, tuple):
                raw = raw[1]
            msg = email_lib.message_from_bytes(raw, policy=policy.default)
            out.append(from_message(msg, source="imap"))
        return out
    except imaplib.IMAP4.error as exc:
        raise MailboxError(f"IMAP error: {exc}") from exc
    except OSError as exc:
        raise MailboxError(f"IMAP connection failed: {exc}") from exc
    finally:
        with suppress(Exception):
            imap.logout()


def fetch_thread(thread_id: str, limit: int = 20) -> list[Email]:
    """Fetch the conversation for a given thread id (References/In-Reply-To)."""
    if not configured():
        raise MailboxError("Mailbox is not configured.")
    imap = _connect_imap()
    try:
        imap.select("INBOX", readonly=True)
        typ, data = imap.search(None, "HEADER", "References", thread_id)
        if typ != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()[-limit:]
        out: list[Email] = []
        for msg_id in reversed(ids):
            typ, msg_data = imap.fetch(msg_id, "(BODY.PEEK[])")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            if isinstance(raw, tuple):
                raw = raw[1]
            msg = email_lib.message_from_bytes(raw, policy=policy.default)
            out.append(from_message(msg, source="imap"))
        return out
    except imaplib.IMAP4.error as exc:
        raise MailboxError(f"IMAP error: {exc}") from exc
    finally:
        with suppress(Exception):
            imap.logout()


def _clean_subject(subject: str) -> str:
    subject = (subject or "").strip()
    lowered = subject.lower()
    for prefix in ("re:", "fwd:", "fw:"):
        if lowered.startswith(prefix):
            subject = subject[len(prefix) :].strip()
            lowered = subject.lower()
    return subject


def _prefix_subject(subject: str) -> str:
    subject = _clean_subject(subject)
    return f"Re: {subject}" if subject else "Re: (no subject)"


def send_reply(
    original: Email,
    body: str,
    attachments: list[str] | None = None,
    cc: str | None = None,
) -> None:
    """Send a reply. Only ever invoked after explicit human approval."""
    me = os.environ["EMAIL_ADDRESS"]
    pwd = os.environ["EMAIL_PASSWORD"]
    msg = EmailMessage()
    msg["From"] = me
    msg["To"] = original.sender
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = _prefix_subject(original.subject)
    if original.message_id:
        msg["In-Reply-To"] = original.message_id
        refs = original.references or []
        if original.message_id not in refs:
            refs = refs + [original.message_id]
        msg["References"] = " ".join(f"<{r.strip('<>')}>" for r in refs)
    msg.set_content(body)
    _send_smtp(msg, me, pwd, [original.sender] + ([cc] if cc else []))


def send_message(to: str, subject: str, body: str, cc: str | None = None) -> None:
    """Send a generic outbound email (used for digests)."""
    me = os.environ["EMAIL_ADDRESS"]
    pwd = os.environ["EMAIL_PASSWORD"]
    msg = EmailMessage()
    msg["From"] = me
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = subject
    msg.set_content(body)
    _send_smtp(msg, me, pwd, [to] + ([cc] if cc else []))


def _send_smtp(msg: EmailMessage, me: str, pwd: str, recipients: list[str]) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587") or 587)
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as s:
                s.login(me, pwd)
                s.send_message(msg, from_addr=me, to_addrs=recipients)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls()
                s.login(me, pwd)
                s.send_message(msg, from_addr=me, to_addrs=recipients)
    except smtplib.SMTPException as exc:
        raise MailboxError(f"SMTP error: {exc}") from exc
    except OSError as exc:
        raise MailboxError(f"SMTP connection failed: {exc}") from exc
