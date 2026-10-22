# 📧 AI Email Assistant

**Production-grade email triage, drafting and inbox-intelligence engine.**

Understands, categorises and prioritises incoming email, extracts action items,
drafts tone-adapted replies, and keeps a human in the loop — nothing is ever
sent without your explicit approval.

Designed as a credible portfolio project: it ships a labelled evaluation
harness, layered prompt-injection guardrails, structured observability, a
REST API, a background worker, and persistent storage — not just a thin LLM
wrapper.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/llm-openrouter%20%7C%20anthropic%20%7C%20openai%20%7C%20local-green" />
  <img src="https://img.shields.io/badge/api-fastapi-blue" />
  <img src="https://img.shields.io/badge/ui-streamlit-red" />
</p>

---

## What it does

| Capability | How |
|------------|-----|
| **Intelligent triage** | Classifies each email (Work / Important / Personal / Newsletter / Promotion / Spam), scores priority 1–5 with reasoning, and summarises it |
| **Action-item extraction** | Pulls concrete tasks the recipient must act on |
| **Tone-adapted drafting** | Professional / Friendly / Concise / Formal / Apologetic, plus custom instructions (e.g. "accept but push the deadline to Friday") |
| **Human-in-the-loop** | Drafts are editable; nothing is sent automatically |
| **Real mailbox** | IMAP fetch (`BODY.PEEK[]`, read-only by default) and SMTP send |
| **Security guardrails** | Prompt-injection detection/neutralisation, PII redaction, draft grounding checks |
| **Persistent memory** | SQLite store of emails, analyses, drafts, feedback and runs |
| **Feedback loop** | Thumbs-up/down and corrections fold back into a personal playbook |
| **Ask your inbox** | RAG-lite search: ask natural-language questions over stored email |
| **REST API** | FastAPI service with auth, CORS, metrics and structured logs |
| **Evaluation** | Labelled dataset + metrics (category F1, priority MAE, action-item F1, injection resistance) runnable offline in CI |

---

## Architecture

```
Incoming email
     │
     ▼
┌──────────────┐  ┌──────────────────┐  ┌────────────────┐
│ Extractor    │─▶│ Guardrails       │─▶│ LLM Triage     │
│ (parse .eml, │  │ (inject/PII scan │  │ (JSON contract │
│  HTML→text)  │  │  sanitise/redact)│  │  + cost track) │
└──────────────┘  └──────────────────┘  └───────┬────────┘
                                                 │
┌──────────────┐  ┌──────────────────┐  ┌───────▼────────┐
│ Digest       │◀─│ Storage (SQLite) │◀─│ Drafting       │
│ (briefing)   │  │ + feedback/playb │  │ (playbook +    │
└──────────────┘  │  book context)   │  │  thread ctx)   │
                  └──────────────────┘  └───────┬────────┘
                                                │
┌──────────────┐  ┌──────────────────┐  ┌───────▼────────┐
│ Review (UI)  │◀─│ Grounding check  │◀─│ Dispatch (SMTP │
│ (Streamlit)  │  │ (facts vs source)│  │  human must    │
└──────────────┘  └──────────────────┘  │  click Send)   │
                                        └────────────────┘
```

## Quick start

```bash
# 1. Clone and enter the project
cd ai-email-assistant

# 2. Create a venv and install
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# 3. Configure .env
cp .env.example .env
# edit .env and add your OPENROUTER_API_KEY (or another provider)

# 4. Run the UI
streamlit run app.py
# open http://localhost:8501

# 5. Or the terminal demo
python cli.py demo

# 6. Run the evaluation (offline, no API key needed)
python scripts/run_eval.py --offline
```

### With Docker

```bash
docker compose up -d        # API on :8000 + background worker
docker compose down
```

## Project structure

```
ai-email-assistant/
├── assistant/              # Core engine
│   ├── extractor.py        # Email parsing, HTML cleanup, threading
│   ├── guardrails.py       # Injection detection, PII redaction, grounding
│   ├── llm.py              # LLM client (retries, cost, caching, fallback)
│   ├── pipeline.py         # Triage, drafting, digest, feedback loop
│   ├── storage.py          # SQLite repository
│   ├── preferences.py      # Learned playbook from feedback
│   ├── search.py           # TF-IDF search + ask-your-inbox
│   ├── mailer.py           # IMAP fetch / SMTP send
│   ├── observability.py    # Structured logging, spans, metrics
│   ├── evaluate.py         # Evaluation harness
│   └── testing.py          # Offline deterministic LLM stub
├── api/
│   └── main.py             # FastAPI REST service
├── worker/
│   └── scheduler.py        # Scheduled triage + daily digest
├── data/
│   ├── sample_emails.json  # Demo dataset
│   └── eval_dataset.json   # 53 labelled emails for evaluation
├── scripts/
│   └── run_eval.py         # Evaluation CLI
├── tests/                  # pytest suite (offline, no API key)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SECURITY.md
│   └── EVALUATION.md
├── app.py                  # Streamlit UI
├── cli.py                  # Terminal CLI (demo, triage, draft, digest, ask, eval, serve)
├── pyproject.toml          # Packaging + ruff/mypy/pytest config
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # API + worker
├── Makefile                # Common tasks
└── .env.example            # Configuration template (never commit .env)
```

## Evaluation (the credibility layer)

Run the labelled harness offline:

```bash
python scripts/run_eval.py --offline
```

Sample output (heuristic stub; real-model numbers are higher):

| Metric | Value |
|--------|-------|
| category_accuracy | 0.72 |
| category_macro_f1 | 0.67 |
| priority_within_1_accuracy | 0.87 |
| injection_resistance_rate | 0.33 (stub) |
| summary_plausible_rate | 1.00 |

The dataset (53 items) includes 6 adversarial prompt-injection emails whose
expected category is the *genuine* category — not what the injection requests.
Use `--fail-under category_accuracy:0.80` as a CI regression gate, or
`--baseline eval-baseline.json` to detect regressions between runs.

See `docs/EVALUATION.md` for the full methodology.

## Security

Layered controls — not just a "please ignore instructions" prompt:

1. **Detect** — weighted heuristic scanner for injection, exfiltration, jailbreak and destructive patterns
2. **Neutralise** — control-char/zero-width stripping, role-marker removal, PII redaction, nonce-fenced untrusted-content block
3. **Verify** — grounding check that flags unsupported facts, dates, amounts and links in drafts

Human approval is mandatory: nothing is sent automatically. IMAP fetches are
read-only (`BODY.PEEK[]`) and never flip the `\\Seen` flag.

See `docs/SECURITY.md` for the threat model and known limitations.

## API

Start the service:

```bash
uvicorn api.main:app --port 8000
# or
make api
```

Key endpoints:

- `GET /health` — service + mailbox + database status
- `GET /metrics` — Prometheus exposition
- `POST /api/v1/emails` — store + triage one email
- `POST /api/v1/emails/triage` — batch triage (most urgent first)
- `GET /api/v1/emails` — list with category / priority / search filters
- `POST /api/v1/emails/{id}/draft` — generate a reviewable reply
- `POST /api/v1/drafts/{id}/send` — send (human approval, mailbox required)
- `POST /api/v1/feedback` — record rating + correction
- `POST /api/v1/search` / `POST /api/v1/ask` — search and ask-your-inbox
- `GET /api/v1/digest` — prioritised briefing
- `POST /api/v1/eval/run` — run the evaluation harness

Optional `X-API-Key` auth (set `API_KEY` in `.env`).

## Background worker

```bash
python -m worker.scheduler --once     # single triage pass
python -m worker.scheduler --daemon   # scheduled triage + daily digest
```

Configure `TRIAGE_INTERVAL_MINUTES`, `DIGEST_CRON_HOUR` and `SEND_DIGEST` in
`.env`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

All tests run offline with no API key or network access.

## Lint / typecheck

```bash
ruff check assistant api worker scripts
ruff format --check assistant api worker scripts
mypy assistant api worker
```

## License

MIT. Feel free to use and adapt.
