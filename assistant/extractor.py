"""Step 1: turn raw messages into a clean, structured :class:`Email`.

Handles the messy parts of real email: HTML-only bodies, quoted reply chains,
signature blocks, attachments, and threading headers.
"""

from __future__ import annotations

import re
from email import policy
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser
from typing import Any

from .schemas import Email

__all__ = [
    "Email",
    "from_message",
    "parse_eml",
    "from_text",
    "clean_body",
    "extract_thread_id",
    "html_to_text",
]


class _HTMLText(HTMLParser):
    """HTML to plain text that keeps block structure readable."""

    BLOCK_TAGS = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "table",
        "section",
        "article",
        "header",
        "footer",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._link_href: str | None = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "head"):
            self._skip += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._link_href = href
                self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a" and self._link_href:
            text = " ".join(self._link_text).strip()
            href = self._link_href.strip()
            if text and href and not href.lower().startswith("mailto:"):
                self.parts.append(f"{text} ({href})")
            elif href and not text:
                self.parts.append(href)
            self._link_href = None
            self._link_text = []

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._link_href is not None:
            self._link_text.append(data)
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(html: str) -> str:
    """Convert an HTML body to readable plain text."""
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not break triage
        return re.sub(r"<[^>]+>", " ", html)
    text = parser.text()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


QUOTE_HEADER = re.compile(
    r"^\s*(?:-{2,}\s*original message\s*-{2,}"
    r"|on .{0,120}\s+wrote\s*:"
    r"|_{5,}"
    r"|from:\s.+$)",
    re.IGNORECASE,
)

SIGNATURE_MARKERS = (
    "\n--\n",
    "\n-- \n",
    "sent from my ",
    "this email and any attachments",
    "get outlook for ",
)

ATTACHMENT_SIGNATURE = re.compile(r"^\s*(?:--\s*$|_{10,}\s*$|-{3,}\s*$)", re.MULTILINE)


def _strip_signature(text: str) -> str:
    """Remove a trailing signature block (best effort, never destructive)."""
    lowered = text.lower()
    cut = len(text)
    for marker in SIGNATURE_MARKERS:
        idx = lowered.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    if cut < len(text):
        head = text[:cut].rstrip()
        if len(head) > 120:
            return head
    return text


def clean_body(text: str, max_chars: int = 8000) -> str:
    """Drop quoted replies, signatures and excess whitespace."""
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if QUOTE_HEADER.match(stripped):
            break
        lines.append(line.rstrip())
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    out = _strip_signature(out)
    return out[:max_chars]


def _snippet(body: str, limit: int = 200) -> str:
    collapsed = " ".join(body.split())
    return collapsed[:limit]


def extract_thread_id(message: Message) -> str:
    """Best-effort thread key from References/In-Reply-To headers."""
    refs = message.get("References") or ""
    ids = re.findall(r"<[^>]+>", str(refs))
    if ids:
        return ids[0].strip("<>")
    in_reply_to = message.get("In-Reply-To")
    if in_reply_to:
        found = re.findall(r"<[^>]+>", str(in_reply_to))
        if found:
            return found[0].strip("<>")
    return ""


def _address_of(header_value: str) -> str:
    match = re.search(r"<([^>]+)>", header_value or "")
    if match:
        return match.group(1).strip()
    return (header_value or "").strip()


def _body_and_attachments(msg: Message) -> tuple[str, list[str], list[dict[str, Any]]]:
    body = ""
    html_body = ""
    attachments: list[str] = []
    attachment_meta: list[dict[str, Any]] = []

    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            content_type = part.get_content_type()
            filename = part.get_filename()
            if "attachment" in disposition.lower() or (
                filename and content_type not in ("text/plain", "text/html")
            ):
                name = filename or "unnamed"
                attachments.append(name)
                try:
                    size = len(part.get_payload(decode=True) or b"")
                except Exception:  # noqa: BLE001
                    size = 0
                attachment_meta.append({"filename": name, "content_type": content_type, "size": size})
                continue
            if content_type == "text/plain" and not body:
                body = _decode_part(part)
            elif content_type == "text/html" and not html_body:
                html_body = _decode_part(part)
    else:
        content = _decode_part(msg)
        if msg.get_content_type() == "text/html":
            html_body = content
        else:
            body = content

    if not body and html_body:
        body = html_to_text(html_body)
    return body, attachments, attachment_meta


def _decode_part(part: Message) -> str:
    try:
        payload = part.get_content()
    except Exception:  # noqa: BLE001
        try:
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    if not isinstance(payload, str):
        return ""
    return payload


def from_message(msg: Message, source: str = "imap") -> Email:
    """Build an :class:`Email` from a parsed :mod:`email` message."""
    body, attachments, attachment_meta = _body_and_attachments(msg)
    cleaned = clean_body(body)
    sender = str(msg.get("From", "") or "")
    references = re.findall(r"<[^>]+>", str(msg.get("References", "") or ""))
    return Email(
        sender=sender,
        sender_email=_address_of(sender),
        to=str(msg.get("To", "") or ""),
        subject=str(msg.get("Subject", "(no subject)") or "(no subject)"),
        date=str(msg.get("Date", "") or ""),
        body=cleaned,
        message_id=str(msg.get("Message-ID", "") or "").strip(),
        thread_id=extract_thread_id(msg),
        in_reply_to=str(msg.get("In-Reply-To", "") or ""),
        references=references,
        attachments=attachments,
        attachment_meta=attachment_meta,
        snippet=_snippet(cleaned),
        source=source,
    )


def parse_eml(raw: bytes) -> Email:
    """Parse an exported ``.eml`` file into an :class:`Email`."""
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return from_message(message, source="eml")


def from_text(
    sender: str,
    subject: str,
    body: str,
    date: str = "",
    message_id: str = "",
    source: str = "manual",
    **extra: Any,
) -> Email:
    """Build an :class:`Email` from plain text (manual paste, samples, fixtures)."""
    cleaned = clean_body(body or "")
    return Email(
        sender=sender,
        sender_email=_address_of(sender),
        subject=subject or "(no subject)",
        body=cleaned,
        date=date,
        message_id=message_id,
        snippet=_snippet(cleaned),
        source=source,
        **extra,
    )
