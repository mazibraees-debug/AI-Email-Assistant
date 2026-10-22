import json

from assistant import EmailAssistant, from_text
from assistant.extractor import clean_body


class FakeLLM:
    def complete(self, system, user, max_tokens=1000):
        if "triage engine" in system:
            return (
                "```json\n"
                + json.dumps(
                    {
                        "category": "Work",
                        "priority": 5,
                        "priority_reason": "Deadline today",
                        "summary": "Demo moved.",
                        "action_items": ["Send deck"],
                        "needs_reply": True,
                    }
                )
                + "\n```"
            )
        return "Hi Sarah, confirmed.\n\nBest,\nTester"


def test_analyze_parses_fenced_json():
    bot = EmailAssistant(llm=FakeLLM())
    a = bot.analyze(from_text("a@b.com", "Demo", "Please confirm"))
    assert a.category == "Work" and a.priority == 5 and a.action_items == ["Send deck"]


def test_inbox_returns_all():
    bot = EmailAssistant(llm=FakeLLM())
    out = bot.process_inbox([from_text("a", "s1", "b"), from_text("c", "s2", "d")])
    assert len(out) == 2


def test_draft_reply():
    bot = EmailAssistant(llm=FakeLLM(), sender_name="Tester")
    e = from_text("a@b.com", "Demo", "Confirm?")
    assert "confirmed" in bot.draft_reply(e).lower()


def test_clean_body_strips_quotes():
    assert clean_body("Thanks!\n> old text\nOn Mon, X wrote:\nmore") == "Thanks!"


def test_bad_llm_output_does_not_crash():
    class Bad(FakeLLM):
        def complete(self, *a, **k):
            return "not json"

    a = EmailAssistant(llm=Bad()).analyze(from_text("a", "s", "b"))
    assert a.priority == 3
