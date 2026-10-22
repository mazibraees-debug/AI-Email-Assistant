"""Terminal interface for the AI Email Assistant.

Subcommands:
    demo        Triaged sample emails and optionally draft a reply (offline-safe)
    triage      Print a prioritised table of sample emails
    draft       Generate a reply draft for one pasted email
    digest      Print a prioritised inbox briefing
    ask         Ask a question over the stored inbox
    eval        Run the offline evaluation harness
    serve       Start the FastAPI service
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assistant import EmailAssistant, from_text, PipelineConfig
from assistant.observability import configure_logging
from assistant.testing import HeuristicLLM


def _assistant(offline: bool = False, db_path: str | None = None) -> EmailAssistant:
    from assistant.storage import get_db, reset_db

    if db_path:
        reset_db(db_path)
    if offline:
        return EmailAssistant(llm=HeuristicLLM(), config=PipelineConfig(persist=False))
    try:
        return EmailAssistant(db=get_db())
    except Exception:
        return EmailAssistant(llm=HeuristicLLM(), config=PipelineConfig(persist=False))


def _sample_emails():
    data = json.loads((Path(__file__).parent / "data/sample_emails.json").read_text())
    return [from_text(**d) for d in data]


def cmd_demo(args: argparse.Namespace) -> int:
    bot = _assistant(offline=args.offline)
    results = bot.process_inbox(_sample_emails())
    for email, analysis in results:
        print(f"[{analysis.priority}/5] {analysis.category}: {email.subject}")
        print(f"  {analysis.summary}")
        for item in analysis.action_items:
            print(f"  - {item}")
    if args.reply:
        top, ana = results[0]
        print("\n--- Draft reply ---")
        print(bot.draft_reply(top, ana))
        print("\n(Review before sending - nothing is sent automatically.)")
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    bot = _assistant(offline=args.offline)
    results = bot.process_inbox_detailed(_sample_emails())
    print(f"{'P':>2} {'Category':<12} {'Subject'}")
    print("-" * 60)
    for r in results:
        print(f"{r.analysis.priority:>2} {str(r.analysis.category):<12} {r.email.subject}")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    bot = _assistant(offline=args.offline)
    sender = args.sender or "someone@example.com"
    subject = args.subject or "(no subject)"
    body = args.body or Path(args.file).read_text() if args.file else ""
    if not body:
        body = sys.stdin.read()
    email = from_text(sender, subject, body)
    analysis = bot.analyze(email)
    draft = bot.generate_draft(email, analysis, args.tone, args.instructions)
    print(f"Category: {analysis.category} | Priority: {analysis.priority}/5")
    print(f"Tone: {draft.tone}")
    if draft.warnings:
        print("Warnings: " + "; ".join(draft.warnings))
    print("\n--- Draft ---")
    print(draft.body)
    print("-------------")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    bot = _assistant(offline=args.offline)
    digest = bot.digest(_sample_emails() if not args.stored else None)
    print(digest.text)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    bot = _assistant(offline=args.offline)
    answer = bot.ask(args.question, top_k=args.top_k)
    print(answer["answer"])
    if answer["citations"]:
        print("\nCitations:")
        for c in answer["citations"]:
            print(f"  - {c['subject']}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from assistant.evaluate import run_evaluation

    report = run_evaluation(
        dataset_path=args.dataset,
        llm=HeuristicLLM() if args.offline else None,
        concurrency=args.concurrency,
    )
    print(report.to_markdown())
    if args.output:
        Path(args.output).write_text(report.to_markdown(), encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-email-assistant", description=__doc__)
    p.add_argument("--offline", action="store_true", help="use the deterministic stub (no API key)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("demo", help="triaged sample emails, optional draft")
    sp.add_argument("--reply", action="store_true")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("triage", help="prioritised table of sample emails")
    sp.set_defaults(func=cmd_triage)

    sp = sub.add_parser("draft", help="generate a reply draft")
    sp.add_argument("--sender", default=None)
    sp.add_argument("--subject", default=None)
    sp.add_argument("--body", default=None)
    sp.add_argument("--file", default=None, help="read body from file")
    sp.add_argument("--tone", default="Professional")
    sp.add_argument("--instructions", default="")
    sp.set_defaults(func=cmd_draft)

    sp = sub.add_parser("digest", help="inbox briefing")
    sp.add_argument("--stored", action="store_true", help="use stored emails instead of samples")
    sp.set_defaults(func=cmd_digest)

    sp = sub.add_parser("ask", help="ask a question over the inbox")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=5)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="run evaluation harness")
    sp.add_argument("--dataset", default="data/eval_dataset.json")
    sp.add_argument("--output", default=None)
    sp.add_argument("--concurrency", type=int, default=4)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("serve", help="start FastAPI service")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
