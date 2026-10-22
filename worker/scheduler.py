"""Background worker: scheduled inbox triage and daily digest.

Safe to run offline: when no LLM provider is configured it falls back to the
deterministic :class:`assistant.testing.HeuristicLLM`, and the digest is only
sent when ``SEND_DIGEST=true`` plus a real mailbox are configured.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from assistant import EmailAssistant
from assistant.mailer import MailboxError, configured, fetch_unread, send_message
from assistant.observability import get_logger
from assistant.pipeline import PipelineConfig
from assistant.testing import HeuristicLLM

log = get_logger("worker")

TRIAGE_INTERVAL_MINUTES = int(os.getenv("TRIAGE_INTERVAL_MINUTES", "30"))
DIGEST_CRON_HOUR = int(os.getenv("DIGEST_CRON_HOUR", "8"))
SEND_DIGEST = os.getenv("SEND_DIGEST", "false").lower() in ("1", "true", "yes")
DIGEST_RECIPIENT = os.getenv("DIGEST_RECIPIENT", "")


class _Scheduler:
    """Thin wrapper so the scheduler can only be started once per process."""

    def __init__(self) -> None:
        self._sched: BackgroundScheduler | None = None

    def start(self, triage_job, digest_job) -> None:
        if self._sched is not None:
            return
        sched = BackgroundScheduler()
        sched.add_job(triage_job, "interval", minutes=TRIAGE_INTERVAL_MINUTES, id="triage")
        sched.add_job(digest_job, "cron", hour=DIGEST_CRON_HOUR, id="digest")
        sched.start()
        self._sched = sched

    def shutdown(self) -> None:
        if self._sched is not None:
            self._sched.shutdown(wait=False)
            self._sched = None


_scheduler = _Scheduler()


def _build_assistant() -> EmailAssistant:
    try:
        return EmailAssistant()
    except Exception:  # noqa: BLE001 - degrade to the offline stub
        log.warning("no LLM provider configured - using offline heuristic stub")
        return EmailAssistant(llm=HeuristicLLM(), config=PipelineConfig(persist=False))


class TriageWorker:
    """Fetch unread mail, triage it, persist results, return a summary."""

    def __init__(self, assistant: EmailAssistant | None = None) -> None:
        self.assistant = assistant or _build_assistant()

    def run_once(self) -> dict[str, Any]:
        if not configured():
            return {
                "status": "skipped",
                "reason": "mailbox not configured",
                "processed": 0,
            }
        try:
            emails = fetch_unread(limit=20)
        except MailboxError as exc:  # pragma: no cover - network
            log.error("triage fetch failed", extra={"context": {"error": str(exc)[:300]}})
            return {"status": "error", "reason": str(exc), "processed": 0}

        if not emails:
            return {"status": "ok", "processed": 0, "urgent": 0, "needs_reply": 0}

        results = self.assistant.process_inbox_detailed(emails)
        urgent = sum(1 for r in results if r.analysis.priority >= 4)
        needs_reply = sum(1 for r in results if r.analysis.needs_reply)
        tokens = sum(r.analysis.tokens for r in results)
        cost = round(sum(r.analysis.cost_usd for r in results), 6)

        log.info(
            "triage complete",
            extra={
                "context": {
                    "processed": len(results),
                    "urgent": urgent,
                    "needs_reply": needs_reply,
                    "cost_usd": cost,
                }
            },
        )
        return {
            "status": "ok",
            "processed": len(results),
            "urgent": urgent,
            "needs_reply": needs_reply,
            "tokens": tokens,
            "cost_usd": cost,
            "top": [
                {
                    "subject": r.email.subject,
                    "sender": r.email.sender,
                    "category": str(r.analysis.category),
                    "priority": r.analysis.priority,
                    "summary": r.analysis.summary,
                }
                for r in results[:5]
            ],
        }


class DigestJob:
    """Build a digest and optionally email it."""

    def __init__(self, assistant: EmailAssistant | None = None) -> None:
        self.assistant = assistant or _build_assistant()

    def run_once(self) -> dict[str, Any]:
        digest = self.assistant.digest()
        sent = False
        if SEND_DIGEST and configured() and DIGEST_RECIPIENT:
            try:
                send_message(
                    to=DIGEST_RECIPIENT,
                    subject=f"AI Email Assistant - Daily digest ({digest.generated_at[:10]})",
                    body=digest.text,
                )
                sent = True
            except MailboxError as exc:  # pragma: no cover - network
                log.error("digest send failed", extra={"context": {"error": str(exc)[:300]}})
        return {
            "status": "ok",
            "generated_at": digest.generated_at,
            "email_count": digest.email_count,
            "urgent_count": digest.urgent_count,
            "needs_reply_count": digest.needs_reply_count,
            "sent": sent,
        }


def _install_signal_handlers(sched: _Scheduler) -> None:
    def handler(signum, frame):  # noqa: ARG001
        sched.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def run_triage_once() -> dict[str, Any]:
    return TriageWorker().run_once()


def run_digest_once() -> dict[str, Any]:
    return DigestJob().run_once()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Email Assistant background worker")
    parser.add_argument("--daemon", action="store_true", help="run the scheduler continuously")
    parser.add_argument("--once", action="store_true", help="run a single triage pass and exit")
    parser.add_argument(
        "--digest", action="store_true", help="build (and optionally send) a digest once and exit"
    )
    args = parser.parse_args(argv)

    if args.once:
        print(json.dumps(run_triage_once(), indent=2))
        return 0
    if args.digest:
        print(json.dumps(run_digest_once(), indent=2))
        return 0
    if args.daemon:
        if not SEND_DIGEST:
            log.info("digest sending disabled (SEND_DIGEST != true)")
        sched = BackgroundScheduler()
        sched.add_job(run_triage_once, "interval", minutes=TRIAGE_INTERVAL_MINUTES, id="triage")
        sched.add_job(run_digest_once, "cron", hour=DIGEST_CRON_HOUR, id="digest")
        _install_signal_handlers(sched)
        sched.start()
        log.info("worker daemon started")
        try:
            while True:
                signal.pause() if hasattr(signal, "pause") else __import__("time").sleep(3600)
        except KeyboardInterrupt:
            sched.shutdown()
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
