"""Step 1: Email extraction - turn raw messages into a clean Email object."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from email import policy
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    to: str = ""
    date: str = ""
    message_id: str = ""
    attachments: list[str] = field(default_factory=list)


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def _html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return " ".join(p.parts)


def clean_body(text: str, max_chars: int = 6000) -> str:
    """Drop quoted replies and excess whitespace."""
    lines = []
    for line in text.splitlines():
        if line.startswith(">"):
            continue
        if re.match(r"^On .+ wrote:$", line.strip()):
            break
        lines.append(line.rstrip())
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return out[:max_chars]


def from_message(msg: Message) -> Email:
    body, attachments = "", []
    if msg.is_multipart():
        for part in msg.walk():
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                attachments.append(part.get_filename() or "unnamed")
            elif part.get_content_type() == "text/plain" and not body:
                body = part.get_content()
            elif part.get_content_type() == "text/html" and not body:
                body = _html_to_text(part.get_content())
    else:
        content = msg.get_content()
        body = _html_to_text(content) if msg.get_content_type() == "text/html" else content
    return Email(
        sender=str(msg.get("From", "")),
        to=str(msg.get("To", "")),
        subject=str(msg.get("Subject", "(no subject)")),
        date=str(msg.get("Date", "")),
        message_id=str(msg.get("Message-ID", "")),
        body=clean_body(body),
        attachments=attachments,
    )


def parse_eml(raw: bytes) -> Email:
    return from_message(BytesParser(policy=policy.default).parsebytes(raw))


def from_text(sender: str, subject: str, body: str) -> Email:
    return Email(sender=sender, subject=subject, body=clean_body(body))
