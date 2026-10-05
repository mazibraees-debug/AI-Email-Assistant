# 📧 AI Email Assistant

A GenAI project that **understands** incoming email, not just writes it.

```
Incoming Email -> Extraction -> LLM Analysis (category, priority, summary, actions)
              -> Reply Generation -> You review/edit -> Send
```

| Stage | File | What it does |
|---|---|---|
| Extraction | `assistant/extractor.py` | Parses .eml/IMAP/plain text, strips quoted replies and HTML |
| Classification, summary, priority | `assistant/pipeline.py` `analyze()` | One LLM call returns structured JSON |
| Reply generation | `assistant/pipeline.py` `draft_reply()` | Tone-aware, uses your instructions, never invents facts |
| Human review | `app.py` | Draft is editable; **nothing is ever sent automatically** |
| Send / fetch | `assistant/mailer.py` | Optional IMAP + SMTP |

## Setup
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add OPENROUTER_API_KEY (or ANTHROPIC_API_KEY)
streamlit run app.py        # web UI
python cli.py               # or terminal demo
pytest                      # tests (no API key needed)
```

## Using a real Gmail inbox
Enable 2FA, create an **App Password**, put it in `.env` as `EMAIL_PASSWORD`, choose "Real inbox (IMAP)" in the sidebar. Fetching is read-only and won't mark mail as read.

## Design notes
- Email content is treated as untrusted data in the prompts (prompt-injection guard).
- JSON parsing is tolerant; a bad model response degrades gracefully instead of crashing.
- The LLM is behind a tiny wrapper (`llm.py`), so swapping providers is a one-file change.

## Ideas to extend
Gmail OAuth, a learned "VIP senders" list, calendar-aware replies (check availability), a vector store of past replies to match your writing style, scheduled digest emails, a feedback loop (thumbs up/down on classifications).
