"""SQLite persistence layer.

Deliberately ORM-free: the schema is small, the queries are explicit, and the
whole layer is standard library so the project stays dependency-light and the
SQL is reviewable in a portfolio setting.

Concurrency: one connection per thread (thread-local), WAL journal mode so
readers never block the API writer, and ``busy_timeout`` so concurrent writers
queue instead of failing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schemas import Analysis, Draft, Email, Feedback, utcnow

DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "data/assistant.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    thread_id TEXT,
    sender TEXT,
    sender_email TEXT,
    to_addr TEXT,
    subject TEXT,
    body TEXT,
    snippet TEXT,
    date TEXT,
    in_reply_to TEXT,
    references_json TEXT,
    attachments_json TEXT,
    source TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_emails_thread ON emails(thread_id);
CREATE INDEX IF NOT EXISTS idx_emails_created ON emails(created_at);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    category TEXT,
    priority INTEGER,
    priority_reason TEXT,
    summary TEXT,
    action_items_json TEXT,
    needs_reply INTEGER,
    sentiment TEXT,
    language TEXT,
    security_flags_json TEXT,
    model TEXT,
    prompt_version TEXT,
    latency_ms INTEGER,
    tokens INTEGER,
    cost_usd REAL,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_email ON analyses(email_id);

CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    body TEXT,
    tone TEXT,
    instructions TEXT,
    warnings_json TEXT,
    model TEXT,
    approved INTEGER DEFAULT 0,
    sent_at TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_drafts_email ON drafts(email_id);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER REFERENCES drafts(id) ON DELETE CASCADE,
    email_id INTEGER,
    rating INTEGER,
    correction TEXT,
    notes TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    provider TEXT,
    model TEXT,
    status TEXT,
    emails_count INTEGER,
    tokens INTEGER,
    cost_usd REAL,
    duration_ms INTEGER,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    kind TEXT,
    payload_json TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);
"""


class Database:
    """Thread-safe SQLite repository."""

    def __init__(self, path: str | None = None):
        self.path = str(path or DEFAULT_DB_PATH)
        if self.path != ":memory:":
            parent = Path(self.path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialised = False
        self._initialise()

    # ------------------------------------------------------------------ core
    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _initialise(self) -> None:
        with self._init_lock:
            if self._initialised:
                return
            conn = self._connect()
            conn.executescript(SCHEMA)
            conn.commit()
            self._initialised = True

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        conn = self._connect()
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        cur = self._connect().execute(sql, tuple(params))
        return cur.fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- emails
    def upsert_email(self, email: Email) -> int:
        """Insert or update an email, keyed on its Message-ID. Returns its id."""
        existing = None
        if email.message_id:
            existing = self.query_one("SELECT id FROM emails WHERE message_id = ?", (email.message_id,))
        if existing is None and not email.message_id:
            existing = self.query_one(
                "SELECT id FROM emails WHERE subject = ? AND sender = ? AND created_at = ?",
                (email.subject, email.sender, email.created_at or utcnow()),
            )
        payload = (
            email.message_id or None,
            email.thread_id or None,
            email.sender,
            email.sender_email,
            email.to,
            email.subject,
            email.body,
            email.snippet,
            email.date,
            email.in_reply_to or None,
            json.dumps(email.references),
            json.dumps(email.attachments),
            email.source,
            utcnow(),
        )
        if existing is not None:
            self.execute(
                """UPDATE emails SET message_id=?, thread_id=?, sender=?, sender_email=?, to_addr=?,
                   subject=?, body=?, snippet=?, date=?, in_reply_to=?, references_json=?,
                   attachments_json=?, source=? WHERE id=?""",
                payload + (existing["id"],),
            )
            return int(existing["id"])
        cur = self.execute(
            """INSERT INTO emails (message_id, thread_id, sender, sender_email, to_addr, subject,
               body, snippet, date, in_reply_to, references_json, attachments_json, source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            payload,
        )
        return int(cur.lastrowid or 0)

    def get_email(self, email_id: int) -> Email | None:
        row = self.query_one("SELECT * FROM emails WHERE id = ?", (email_id,))
        return self._row_to_email(row) if row else None

    def list_emails(
        self,
        category: str | None = None,
        needs_reply: bool | None = None,
        min_priority: int | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[Email, Analysis | None]]:
        sql = [
            """SELECT e.*, a.id AS a_id, a.category, a.priority, a.priority_reason, a.summary,
                      a.action_items_json, a.needs_reply, a.sentiment, a.language,
                      a.security_flags_json, a.model AS a_model, a.prompt_version,
                      a.latency_ms, a.tokens, a.cost_usd, a.created_at AS a_created
               FROM emails e
               LEFT JOIN analyses a ON a.id = (
                   SELECT id FROM analyses WHERE email_id = e.id ORDER BY id DESC LIMIT 1
               )"""
        ]
        params: list[Any] = []
        where: list[str] = []
        if category:
            where.append("a.category = ?")
            params.append(category)
        if needs_reply is not None:
            where.append("a.needs_reply = ?")
            params.append(1 if needs_reply else 0)
        if min_priority is not None:
            where.append("a.priority >= ?")
            params.append(min_priority)
        if search:
            where.append("(e.subject LIKE ? OR e.body LIKE ? OR e.sender LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY COALESCE(a.priority, 0) DESC, e.id DESC LIMIT ? OFFSET ?")
        params.extend([limit, offset])
        rows = self.query(" ".join(sql), params)
        return [(self._row_to_email(row), self._row_to_analysis(row)) for row in rows]

    def count_emails(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS c FROM emails")
        return int(row["c"]) if row else 0

    def _row_to_email(self, row: sqlite3.Row) -> Email:
        keys = row.keys()
        return Email(
            id=int(row["id"]) if "id" in keys and row["id"] is not None else None,
            sender=row["sender"] or "",
            sender_email=row["sender_email"] or "",
            to=row["to_addr"] or "",
            subject=row["subject"] or "",
            body=row["body"] or "",
            date=row["date"] or "",
            message_id=(row["message_id"] or "") if "message_id" in keys else "",
            thread_id=(row["thread_id"] or "") if "thread_id" in keys else "",
            in_reply_to=(row["in_reply_to"] or "") if "in_reply_to" in keys else "",
            references=json.loads(row["references_json"] or "[]") if "references_json" in keys else [],
            attachments=json.loads(row["attachments_json"] or "[]") if "attachments_json" in keys else [],
            snippet=(row["snippet"] or "") if "snippet" in keys else "",
            source=(row["source"] or "unknown") if "source" in keys else "unknown",
        )

    def _row_to_analysis(self, row: sqlite3.Row) -> Analysis | None:
        keys = row.keys()
        if "a_id" not in keys or row["a_id"] is None:
            return None
        return Analysis(
            category=row["category"],
            priority=int(row["priority"] or 3),
            priority_reason=row["priority_reason"] or "",
            summary=row["summary"] or "",
            action_items=json.loads(row["action_items_json"] or "[]"),
            needs_reply=bool(row["needs_reply"]),
            sentiment=row["sentiment"] or "neutral",
            language=row["language"] or "en",
            security_flags=json.loads(row["security_flags_json"] or "[]"),
            model=row["a_model"] or "",
            prompt_version=row["prompt_version"] or "",
            latency_ms=int(row["latency_ms"] or 0),
            tokens=int(row["tokens"] or 0),
            cost_usd=float(row["cost_usd"] or 0.0),
            created_at=row["a_created"] or "",
        )

    # -------------------------------------------------------------- analyses
    def save_analysis(self, email_id: int, analysis: Analysis) -> int:
        cur = self.execute(
            """INSERT INTO analyses (email_id, category, priority, priority_reason, summary,
               action_items_json, needs_reply, sentiment, language, security_flags_json,
               model, prompt_version, latency_ms, tokens, cost_usd, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                email_id,
                str(analysis.category),
                analysis.priority,
                analysis.priority_reason,
                analysis.summary,
                json.dumps(analysis.action_items),
                1 if analysis.needs_reply else 0,
                analysis.sentiment,
                analysis.language,
                json.dumps(analysis.security_flags),
                analysis.model,
                analysis.prompt_version,
                analysis.latency_ms,
                analysis.tokens,
                analysis.cost_usd,
                utcnow(),
            ),
        )
        return int(cur.lastrowid or 0)

    def latest_analysis(self, email_id: int) -> Analysis | None:
        rows = self.query(
            """SELECT * FROM analyses WHERE email_id = ? ORDER BY id DESC LIMIT 1""", (email_id,)
        )
        if not rows:
            return None
        row = rows[0]
        return Analysis(
            category=row["category"],
            priority=int(row["priority"] or 3),
            priority_reason=row["priority_reason"] or "",
            summary=row["summary"] or "",
            action_items=json.loads(row["action_items_json"] or "[]"),
            needs_reply=bool(row["needs_reply"]),
            sentiment=row["sentiment"] or "neutral",
            language=row["language"] or "en",
            security_flags=json.loads(row["security_flags_json"] or "[]"),
            model=row["model"] or "",
            prompt_version=row["prompt_version"] or "",
            latency_ms=int(row["latency_ms"] or 0),
            tokens=int(row["tokens"] or 0),
            cost_usd=float(row["cost_usd"] or 0.0),
            created_at=row["created_at"] or "",
        )

    # ---------------------------------------------------------------- drafts
    def save_draft(self, email_id: int, draft: Draft) -> int:
        cur = self.execute(
            """INSERT INTO drafts (email_id, body, tone, instructions, warnings_json, model,
               approved, sent_at, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                email_id,
                draft.body,
                str(draft.tone),
                draft.instructions,
                json.dumps(draft.warnings),
                draft.model,
                1 if draft.approved else 0,
                draft.sent_at,
                draft.created_at or utcnow(),
            ),
        )
        draft.id = int(cur.lastrowid or 0)
        return draft.id

    def get_draft(self, draft_id: int) -> Draft | None:
        row = self.query_one("SELECT * FROM drafts WHERE id = ?", (draft_id,))
        if not row:
            return None
        return Draft(
            id=row["id"],
            email_id=row["email_id"],
            body=row["body"] or "",
            tone=row["tone"],
            instructions=row["instructions"] or "",
            warnings=json.loads(row["warnings_json"] or "[]"),
            model=row["model"] or "",
            approved=bool(row["approved"]),
            sent_at=row["sent_at"],
            created_at=row["created_at"] or "",
        )

    def mark_draft_sent(self, draft_id: int) -> None:
        self.execute("UPDATE drafts SET sent_at = ?, approved = 1 WHERE id = ?", (utcnow(), draft_id))

    def drafts_for_email(self, email_id: int) -> list[Draft]:
        rows = self.query("SELECT id FROM drafts WHERE email_id = ? ORDER BY id DESC", (email_id,))
        return [d for d in (self.get_draft(r["id"]) for r in rows) if d]

    # -------------------------------------------------------------- feedback
    def save_feedback(self, feedback: Feedback) -> int:
        cur = self.execute(
            """INSERT INTO feedback (draft_id, email_id, rating, correction, notes, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                feedback.draft_id,
                feedback.email_id,
                feedback.rating,
                feedback.correction,
                feedback.notes,
                feedback.created_at or utcnow(),
            ),
        )
        return int(cur.lastrowid or 0)

    def feedback_for_draft(self, draft_id: int) -> list[Feedback]:
        rows = self.query("SELECT * FROM feedback WHERE draft_id = ? ORDER BY id DESC", (draft_id,))
        return [
            Feedback(
                id=r["id"],
                draft_id=r["draft_id"],
                email_id=r["email_id"],
                rating=r["rating"],
                correction=r["correction"] or "",
                notes=r["notes"] or "",
                created_at=r["created_at"] or "",
            )
            for r in rows
        ]

    def all_feedback(self, limit: int = 500) -> list[Feedback]:
        rows = self.query("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,))
        return [
            Feedback(
                id=r["id"],
                draft_id=r["draft_id"],
                email_id=r["email_id"],
                rating=r["rating"],
                correction=r["correction"] or "",
                notes=r["notes"] or "",
                created_at=r["created_at"] or "",
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ runs
    def start_run(self, kind: str, provider: str = "", model: str = "") -> int:
        cur = self.execute(
            "INSERT INTO runs (kind, provider, model, status, started_at) VALUES (?,?,?,?,?)",
            (kind, provider, model, "running", utcnow()),
        )
        return int(cur.lastrowid or 0)

    def finish_run(
        self,
        run_id: int,
        status: str = "ok",
        emails_count: int = 0,
        tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        error: str = "",
    ) -> None:
        self.execute(
            """UPDATE runs SET status=?, emails_count=?, tokens=?, cost_usd=?, duration_ms=?,
               finished_at=?, error=? WHERE id=?""",
            (status, emails_count, tokens, cost_usd, duration_ms, utcnow(), error, run_id),
        )

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def log_event(self, run_id: int | None, kind: str, payload: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO events (run_id, kind, payload_json, created_at) VALUES (?,?,?,?)",
            (run_id, kind, json.dumps(payload, default=str), utcnow()),
        )

    # -------------------------------------------------------------- settings
    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), utcnow()),
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return row["value"]

    def all_settings(self) -> dict[str, Any]:
        rows = self.query("SELECT key, value FROM settings")
        out: dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except (TypeError, ValueError):
                out[r["key"]] = r["value"]
        return out

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        by_category = {
            r["category"]: r["c"]
            for r in self.query("SELECT category, COUNT(*) AS c FROM analyses GROUP BY category")
        }
        totals = self.query_one(
            """SELECT COUNT(*) AS analyses, COALESCE(SUM(tokens),0) AS tokens,
               COALESCE(SUM(cost_usd),0) AS cost, COALESCE(AVG(latency_ms),0) AS latency
               FROM analyses"""
        )
        drafts = self.query_one(
            """SELECT COUNT(*) AS total,
               COALESCE(SUM(CASE WHEN sent_at IS NOT NULL THEN 1 ELSE 0 END),0) AS sent
               FROM drafts"""
        )
        return {
            "emails": self.count_emails(),
            "analyses": int(totals["analyses"] if totals else 0),
            "tokens": int(totals["tokens"] if totals else 0),
            "cost_usd": round(float(totals["cost"] if totals else 0.0), 6),
            "avg_latency_ms": int(totals["latency"] if totals else 0),
            "by_category": by_category,
            "drafts": int(drafts["total"] if drafts else 0),
            "drafts_sent": int(drafts["sent"] if drafts else 0),
        }


_db: Database | None = None
_db_lock = threading.Lock()


def get_db(path: str | None = None) -> Database:
    """Process-wide singleton database handle."""
    global _db
    with _db_lock:
        if _db is None:
            _db = Database(path)
        return _db


def reset_db(path: str | None = None) -> Database:
    """Replace the singleton (used by tests and CLI ``--db`` flag)."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = Database(path)
        return _db
