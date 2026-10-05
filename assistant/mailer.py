"""Optional: connect to a real mailbox (IMAP to read, SMTP to send)."""
from __future__ import annotations

import email as email_lib
import imaplib
import os
import smtplib
from email import policy
from email.message import EmailMessage

from .extractor import Email, from_message


def configured() -> bool:
    addr = os.getenv("EMAIL_ADDRESS", "").strip()
    pwd = os.getenv("EMAIL_PASSWORD", "").strip()
    return bool(addr and pwd and addr != "you@example.com" and pwd != "your_app_password")


def fetch_unread(limit: int = 10) -> list[Email]:
    host = os.getenv("IMAP_HOST", "imap.gmail.com")
    with imaplib.IMAP4_SSL(host) as imap:
        imap.login(os.environ["EMAIL_ADDRESS"], os.environ["EMAIL_PASSWORD"])
        imap.select("INBOX", readonly=True)  # readonly: does not mark mail as read
        _, data = imap.search(None, "UNSEEN")
        ids = data[0].split()[-limit:]
        out = []
        for i in reversed(ids):
            _, msg_data = imap.fetch(i, "(BODY.PEEK[])")
            msg = email_lib.message_from_bytes(msg_data[0][1], policy=policy.default)
            out.append(from_message(msg))
        return out


def send_reply(original: Email, body: str) -> None:
    """Only ever called after the user explicitly clicks Send."""
    me = os.environ["EMAIL_ADDRESS"]
    msg = EmailMessage()
    msg["From"], msg["To"] = me, original.sender
    subj = original.subject
    msg["Subject"] = subj if subj.lower().startswith("re:") else f"Re: {subj}"
    if original.message_id:
        msg["In-Reply-To"] = msg["References"] = original.message_id
    msg.set_content(body)
    with smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(me, os.environ["EMAIL_PASSWORD"])
        s.send_message(msg)
