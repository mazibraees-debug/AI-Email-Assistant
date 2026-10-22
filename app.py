"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from assistant import EmailAssistant, from_text, parse_eml
from assistant import mailer
from assistant.pipeline import TONES
from assistant.schemas import Category, Tone
from assistant.storage import get_db

st.set_page_config(page_title="AI Email Assistant", page_icon="📧", layout="wide")
st.title("📧 AI Email Assistant")
st.caption("Triage, prioritise, draft and review — nothing is sent without your approval.")

PRIORITY_ICON = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🟢", 1: "⚪"}


@st.cache_resource
def get_assistant():
    return EmailAssistant(db=get_db())


try:
    bot = get_assistant()
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Email source")
    source = st.radio(
        "Load from",
        ["Sample emails", "Paste an email", "Upload .eml", "Real inbox (IMAP)"],
    )
    emails = []

    if source == "Sample emails":
        data = json.loads((Path(__file__).parent / "data/sample_emails.json").read_text())
        emails = [from_text(**d) for d in data]
    elif source == "Paste an email":
        s = st.text_input("From")
        sub = st.text_input("Subject")
        body = st.text_area("Body", height=200)
        if body:
            emails = [from_text(s, sub, body)]
    elif source == "Upload .eml":
        files = st.file_uploader("Upload", type="eml", accept_multiple_files=True)
        emails = [parse_eml(f.read()) for f in files or []]
    else:
        if not mailer.configured():
            st.warning("Set EMAIL_ADDRESS / EMAIL_PASSWORD in .env to connect to your inbox.")
        elif st.button("Fetch unread"):
            try:
                st.session_state["inbox"] = mailer.fetch_unread(10)
            except Exception as exc:
                st.error(f"Failed to fetch emails: {exc}")
        emails = st.session_state.get("inbox", [])

    tone = st.selectbox("Reply tone", TONES, index=0)
    instructions = st.text_input(
        "Instructions (optional)",
        placeholder="e.g. accept but ask to push the deadline to Friday",
    )
    run = st.button("Analyze emails", type="primary", disabled=not emails)

if run:
    with st.spinner("Classifying, summarizing and prioritizing..."):
        st.session_state["results"] = bot.process_inbox_detailed(emails)
        st.session_state["drafts"] = {}

results = st.session_state.get("results")
if not results:
    st.info("Pick an email source in the sidebar and click **Analyze emails**.")
    st.stop()

# ---------- Filters ----------
cat_filter = st.multiselect(
    "Filter by category",
    [c.value for c in Category],
    default=[c.value for c in Category],
)
min_priority = st.slider("Minimum priority", 1, 5, 1)
filtered = [
    r for r in results
    if r.analysis.category.value in cat_filter and r.analysis.priority >= min_priority
]

# ---------- Inbox list ----------
left, right = st.columns([1, 2])
with left:
    st.subheader(f"Inbox ({len(filtered)} of {len(results)})")
    labels = [
        f"{PRIORITY_ICON[r.analysis.priority]} P{r.analysis.priority} {r.analysis.category} - {r.email.subject[:40]}"
        for r in filtered
    ]
    idx = st.radio(
        "Emails", range(len(filtered)), format_func=lambda i: labels[i], label_visibility="collapsed"
    )

# ---------- Detail + reply ----------
result = filtered[idx]
email = result.email
analysis = result.analysis
with right:
    st.subheader(email.subject)
    st.caption(f"From: {email.sender}  |  {email.date}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", analysis.category)
    c2.metric("Priority", f"{analysis.priority}/5")
    c3.metric("Needs reply", "Yes" if analysis.needs_reply else "No")
    st.write(f"**Why this priority:** {analysis.priority_reason}")
    st.info(f"**Summary:** {analysis.summary}")
    if analysis.action_items:
        st.write("**Action items**")
        for item in analysis.action_items:
            st.write(f"- {item}")
    if analysis.security_flags:
        st.warning("Security flags: " + ", ".join(analysis.security_flags))
    with st.expander("Original email"):
        st.text(email.body)

    st.divider()
    st.subheader("Reply")
    instr = st.text_input(
        "Instructions (optional)",
        placeholder="e.g. say yes but ask to push the deadline to Friday",
        key=f"i{idx}",
        value=instructions,
    )
    if st.button("Generate reply", key=f"g{idx}"):
        with st.spinner("Writing..."):
            st.session_state["drafts"][idx] = bot.generate_draft(
                email, analysis, tone, instr
            )

    if idx in st.session_state["drafts"]:
        draft = st.session_state["drafts"][idx]
        st.text_area(
            "Review and edit before sending",
            draft.body,
            height=260,
            key=f"d{idx}",
        )
        edited = st.session_state.get(f"d{idx}", draft.body)

        if analysis.warnings:
            st.warning("Grounding warnings: " + "; ".join(analysis.warnings))

        col_send, col_rate = st.columns([1, 1])
        with col_send:
            if mailer.configured():
                if st.button("Send email", type="primary", key=f"s{idx}"):
                    try:
                        mailer.send_reply(email, edited)
                        st.success("Sent!")
                        del st.session_state["drafts"][idx]
                    except Exception as exc:
                        st.error(f"Failed to send: {exc}")
            else:
                st.caption("Sending disabled (no SMTP credentials).")
        with col_rate:
            rating = st.select_slider(
                "Rate this draft",
                options=[1, 2, 3, 4, 5],
                value=5,
                key=f"r{idx}",
            )
            correction = st.text_input("Correction (optional)", key=f"c{idx}")
            if st.button("Save feedback", key=f"f{idx}"):
                res = bot.record_feedback(draft, rating, correction)
                st.success(f"Feedback saved (avg rating: {res['learned'].get('average_rating', 'n/a')})")

# ---------- Digest & ask ----------
st.divider()
col_d, col_a = st.columns(2)
with col_d:
    st.subheader("Inbox digest")
    if st.button("Generate digest"):
        digest = bot.digest()
        st.text(digest.text)
with col_a:
    st.subheader("Ask your inbox")
    question = st.text_input("Question", placeholder="What is the deadline for the client demo?")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = bot.ask(question)
        st.write(answer["answer"])
        if answer["citations"]:
            st.caption("Citations: " + ", ".join(c["subject"] for c in answer["citations"]))
