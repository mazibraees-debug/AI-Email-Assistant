# Architecture

The AI Email Assistant is a layered, human-in-the-loop email triage and
drafting engine. Every email passes through a fixed pipeline:

```text
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Inbox     │────▶│   Extractor      │────▶│  Guardrails │
│  (IMAP/.eml │     │  (parse_eml)     │     │ (scan/sanit │
│  /paste)    │     │                  │     │ ize/redact) │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                     │
┌─────────────┐     ┌──────────────────┐     ┌──────▼──────┐
│   Digest    │◀────│   Storage        │◀────│   LLM Triage│
│  (briefing) │     │  (SQLite)        │     │  (JSON      │
└─────────────┘     │  + feedback      │     │  contract)  │
                    │  + playbook      │     └──────┬──────┘
                    └──────────────────┘            │
                                                    │
┌─────────────┐     ┌──────────────────┐     ┌──────▼──────┐
│   Review    │◀────│   Drafting       │◀────│   Grounding │
│  (UI/CLI)   │     │  (playbook +     │     │  check      │
└─────────────┘     │   thread ctx)    │     └──────┬──────┘
                    └──────────────────┘            │
                                                    │
                                       ┌────────────▼────────────┐
                                       │   Dispatch (SMTP, human │
                                       │   must click Send)      │
                                       └─────────────────────────┘
```

## Components

| Module | Responsibility |
|--------|----------------|
| `assistant/extractor.py` | Parse `.eml`, HTML → text, quote/signature stripping, threading IDs, attachment metadata |
| `assistant/guardrails.py` | Injection detection, content sanitisation, PII redaction, draft grounding checks |
| `assistant/llm.py` | Provider-agnostic client (OpenRouter / OpenAI / Anthropic / local) with retries, cost tracking, caching, fallback |
| `assistant/pipeline.py` | Triage orchestration, persistence, concurrency, digest, ask-your-inbox, feedback loop |
| `assistant/storage.py` | SQLite repository (emails, analyses, drafts, feedback, runs, events, settings) |
| `assistant/preferences.py` | Learned playbook folded back into drafts from human feedback |
| `assistant/search.py` | TF-IDF search + RAG-lite "ask your inbox" |
| `assistant/mailer.py` | IMAP fetch (BODY.PEEK[], read-only by default) and SMTP send |
| `assistant/observability.py` | Structured JSON logging, spans, Prometheus metrics, cost accounting |
| `assistant/evaluate.py` | Labelled evaluation harness, metrics, CI gate |
| `api/main.py` | FastAPI REST service |
| `worker/scheduler.py` | Scheduled triage + daily digest |
| `app.py` | Streamlit human-in-the-loop UI |
| `cli.py` | Terminal demo / admin commands |

## Data model (SQLite, ORM-free)

- `emails` — normalised messages (keyed on Message-ID)
- `analyses` — one row per triage (category, priority, summary, action items, model, cost, latency)
- `drafts` — generated replies awaiting review (approved / sent flags)
- `feedback` — human ratings + corrections (feeds the playbook)
- `runs` / `events` — audit trail for batch triage jobs
- `settings` — user playbook (tone, sign-off, signature, phrases to avoid)

## Design decisions & trade-offs

- **stdlib SQLite, no ORM**: the schema is small and reviewable; explicit SQL is easier to audit in a security-sensitive AI product.
- **Layered guardrails**: "tell the model to ignore instructions" is a prompt, not a control. We detect, sanitise/redact, and verify (grounding) — findings are surfaced to the human, never silently blocked.
- **Human approval is mandatory**: nothing is ever sent automatically. `send_reply` is only reachable through an explicit API/UI action.
- **Prompt versioning**: `PROMPT_VERSION = "v3"` lets us track which prompt produced which analysis for regression analysis.
- **Offline-first eval**: `scripts/run_eval.py --offline` uses a deterministic stub so the harness runs in CI without network or a paid model.
- **API-key auth + CORS**: optional `API_KEY` gating for the REST service; CORS origins configurable.

## Adding a new LLM provider/model

1. Set the provider key/env vars (see `.env.example`).
2. `assistant.llm.LLM` auto-detects the provider or accepts `provider=` / `model=`.
3. Add pricing to `MODEL_PRICING` in `assistant/observability.py` for cost attribution.
4. Run `scripts/run_eval.py` to validate quality before promoting.

## Running

```bash
# UI
make dev                       # streamlit run app.py

# REST API
make api                       # uvicorn api.main:app

# Worker (one-off triage)
make worker                    # python -m worker.scheduler --once

# Evaluation (offline, CI-safe)
make eval                      # python scripts/run_eval.py --offline
```
