# 📧 AI Email Assistant

An intelligent, production-ready AI email triage and drafting engine that **understands, categorizes, and summarizes** incoming emails, extracts key action items, and generates tone-adapted replies.

Powered by **OpenRouter** (accessing models like Claude 3.5 Sonnet, GPT-4o, Llama 3.3, and free-tier open models) and direct **Anthropic Claude**.

---

## ⚡ Architecture & Workflow

```text
Incoming Email (IMAP / .eml / Text / Sample)
                     │
                     ▼
       1. Extractor (`extractor.py`)
       • Strips quotes, signatures & HTML
       • Sanitizes input against prompt injection
                     │
                     ▼
       2. LLM Triage (`pipeline.py` + `llm.py`)
       • Categorizes (Work, Important, Personal, Newsletter, Promotion, Spam)
       • Scores priority (1 to 5) with reasoning
       • Summarizes & extracts action items
                     │
                     ▼
       3. Reply Drafting (`pipeline.py`)
       • Adapts to chosen tone (Professional, Friendly, Concise, Formal, etc.)
       • Follows user guidance & prevents fact hallucination
                     │
                     ▼
       4. Human-in-the-Loop Review (`app.py` UI)
       • Review, edit, and approve drafts
       • NOTHING is ever sent automatically without your confirmation
                     │
                     ▼
       5. Optional Dispatch (`mailer.py`)
       • Sends via SMTP (Gmail App Password / Custom SMTP)
```

---

## 🚀 Key Features

- **Multi-Provider LLM Integration**: First-class support for **OpenRouter** and **Anthropic** with configurable models in `.env`.
- **Intelligent Triage & Prioritization**: Automatically sorts emails by urgency so you know what needs attention first.
- **Tone-Adapted Drafts**: Choose from *Professional*, *Friendly*, *Concise*, *Formal*, or *Apologetic*, and add custom instructions (e.g., "accept but request delay until Friday").
- **Multiple Email Sources**:
  - 📂 **Sample Emails**: Test instantly without any email setup.
  - ✍️ **Paste an Email**: Quick one-off triage.
  - 📄 **Upload `.eml`**: Process exported files directly.
  - 📬 **Live Inbox (IMAP)**: Read unseen emails in read-only mode (never marks emails as read without permission).
- **Safety First**:
  - Prompt injection mitigation (treats all email content strictly as untrusted data).
  - Credentials and environment secrets excluded via `.gitignore`.
  - Tolerant JSON parser with graceful fallbacks if model output is malformed.

---

## 📁 Project Structure

```text
ai-email-assistant/
├── assistant/
│   ├── __init__.py         # Package exports
│   ├── extractor.py        # Email parsing, HTML cleanup & quote stripping
│   ├── llm.py              # LLM client (OpenRouter & Anthropic with fallback)
│   ├── mailer.py           # Optional IMAP fetch & SMTP dispatch
│   └── pipeline.py         # Triage, classification, priority & reply logic
├── data/
│   └── sample_emails.json  # Realistic sample dataset for testing
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Unit tests with mock LLM
├── .env.example            # Environment variables template
├── .gitignore              # Ignores .env, virtual environments & caches
├── app.py                  # Streamlit Web UI
├── cli.py                  # Interactive CLI demo
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone & Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Copy the sample configuration file:
```bash
cp .env.example .env
```

Open `.env` and add your API key:

#### Option A: OpenRouter (Default & Recommended)
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=anthropic/claude-sonnet-5.5
```
*(You can use any model available on OpenRouter, such as `anthropic/claude-sonnet-5.5`, `openai/gpt-4o-mini`, or free models like `google/gemma-4-31b-it:free`)*.

#### Option B: Direct Anthropic
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

#### Optional: Real Mailbox (Gmail IMAP / SMTP)
```env
IMAP_HOST=imap.gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=you@example.com
EMAIL_PASSWORD=your_16_char_app_password
SENDER_NAME=Your Name
```
> *For Gmail: Enable 2-Step Verification and generate a 16-character **App Password** under Google Account > Security.*

---

## 🖥️ Running the Application

### Web UI (Streamlit)

Launch the interactive web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Terminal Demo (CLI)

Run the fast command-line triage on sample emails:
```bash
python cli.py
```

### Run Tests

Run the test suite (requires no API key; uses mock LLM):
```bash
pytest
```

---

## 🔒 Security & Privacy

- **Your `.env` is ignored**: Sensitive API keys and mailbox passwords are protected by `.gitignore` and won't be committed to version control.
- **Read-Only IMAP**: Fetching emails uses `BODY.PEEK[]` and `readonly=True`, meaning existing unread flags remain untouched.
- **Human Approval**: The application will **never** automatically send an email. You must review and click "Send".

---

## 📜 License

MIT License. Feel free to use and adapt this project!
