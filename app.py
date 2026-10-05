"""Streamlit UI:  streamlit run app.py"""
import json
from pathlib import Path

import streamlit as st

from assistant import EmailAssistant, from_text, parse_eml
from assistant.pipeline import TONES
from assistant import mailer

st.set_page_config(page_title="AI Email Assistant", page_icon="📧", layout="wide")
st.title("📧 AI Email Assistant")

PRIORITY_ICON = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🟢", 1: "⚪"}


@st.cache_resource
def get_assistant():
    return EmailAssistant()


try:
    bot = get_assistant()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# ---------- Sidebar: choose email source ----------
with st.sidebar:
    st.header("Email source")
    source = st.radio("Load from", ["Sample emails", "Paste an email", "Upload .eml", "Real inbox (IMAP)"])
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
            st.warning("Set real EMAIL_ADDRESS and EMAIL_PASSWORD in .env to connect to your inbox.")
        elif st.button("Fetch unread"):
            try:
                st.session_state["inbox"] = mailer.fetch_unread(10)
            except Exception as e:
                st.error(f"Failed to fetch emails: {e}")
        emails = st.session_state.get("inbox", [])

    tone = st.selectbox("Reply tone", TONES)
    run = st.button("Analyze emails", type="primary", disabled=not emails)

if run:
    with st.spinner("Classifying, summarizing and prioritizing..."):
        st.session_state["results"] = bot.process_inbox(emails)
        st.session_state["drafts"] = {}

results = st.session_state.get("results")
if not results:
    st.info("Pick an email source in the sidebar and click **Analyze emails**.")
    st.stop()

# ---------- Inbox list ----------
left, right = st.columns([1, 2])
with left:
    st.subheader("Inbox (most urgent first)")
    labels = [f"{PRIORITY_ICON[a.priority]} {a.category} - {e.subject[:40]}" for e, a in results]
    idx = st.radio("Emails", range(len(results)), format_func=lambda i: labels[i], label_visibility="collapsed")

# ---------- Detail + reply ----------
email, a = results[idx]
with right:
    st.subheader(email.subject)
    st.caption(f"From: {email.sender}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", a.category)
    c2.metric("Priority", f"{a.priority}/5")
    c3.metric("Needs reply", "Yes" if a.needs_reply else "No")
    st.write(f"**Why this priority:** {a.priority_reason}")
    st.info(f"**Summary:** {a.summary}")
    if a.action_items:
        st.write("**Action items**")
        for item in a.action_items:
            st.write(f"- {item}")
    with st.expander("Original email"):
        st.text(email.body)

    st.divider()
    st.subheader("Reply")
    instr = st.text_input("Instructions (optional)", placeholder="e.g. say yes but ask to push the deadline to Friday",
                          key=f"i{idx}")
    if st.button("Generate reply", key=f"g{idx}"):
        with st.spinner("Writing..."):
            st.session_state["drafts"][idx] = bot.draft_reply(email, a, tone, instr)

    if idx in st.session_state["drafts"]:
        # User Reviews step: the draft is editable and nothing is sent automatically
        draft = st.text_area("Review and edit before sending", st.session_state["drafts"][idx],
                             height=260, key=f"d{idx}")
        if mailer.configured():
            if st.button("Send email", type="primary", key=f"s{idx}"):
                try:
                    mailer.send_reply(email, draft)
                    st.success("Sent!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
        else:
            st.caption("Sending is disabled (no SMTP credentials). Copy the draft above.")
