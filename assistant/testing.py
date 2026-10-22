"""Deterministic offline LLM stub used by tests, CI and the offline demo.

The stub implements the same ``complete`` / ``complete_response`` contract as
:class:`assistant.llm.LLM` but predicts with transparent keyword rules. It
exists so that:

* the evaluation harness can run in CI without an API key or network access,
* unit tests never depend on a paid model,
* the demo mode gives a reproducible walkthrough of the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .llm import LLMResponse
from .observability import LLMUsage
from .schemas import Category

URGENT_WORDS = {
    "urgent",
    "asap",
    "immediately",
    "overdue",
    "past due",
    "deadline",
    "final notice",
    "action required",
    "action needed",
    "escalation",
    "escalate",
    "critical",
    "outage",
    "breach",
    "expires",
    "expiring",
    "last day",
    "today",
    "tonight",
    "by 5pm",
    "by 6pm",
}

PROMO_WORDS = {
    "sale",
    "discount",
    "% off",
    "free shipping",
    "unsubscribe",
    "promo code",
    "limited time",
    "flash sale",
    "deals",
    "offers",
    "coupon",
    "black friday",
}

NEWSLETTER_WORDS = {
    "digest",
    "weekly",
    "newsletter",
    "issue",
    "this week in",
    "roundup",
    "curated",
    "subscribe",
    "read more",
    "stories you missed",
    "webinar recap",
}

SPAM_WORDS = {
    "lottery",
    "viagra",
    "crypto giveaway",
    "wire transfer",
    "click here to claim",
    "nigerian prince",
    "inheritance fund",
    "bitcoin bonus",
}

WORK_WORDS = {
    "meeting",
    "deadline",
    "project",
    "client",
    "invoice",
    "team",
    "standup",
    "review",
    "deploy",
    "release",
    "agenda",
    "schedule",
    "presentation",
    "contract",
    "budget",
    "interview",
    "onboarding",
    "report",
    "sprint",
    "roadmap",
}

PERSONAL_WORDS = {
    "birthday",
    "dinner",
    "family",
    "mom",
    "dad",
    "weekend",
    "concert",
    "holiday",
    "lunch",
    "baby",
    "wedding",
    "friend",
    "holiday party",
}

QUESTION_WORDS = ("?", "could you", "can you", "please confirm", "let me know", "any update")

ACTION_PATTERNS = [
    re.compile(
        r"(?:please\s+)?(?:confirm|send|share|reply|review|sign|approve|schedule|provide|submit|pay|update|prepare|send)\b[^.!?\n]{3,80}",
        re.I,
    ),
    re.compile(r"need(?:s)? (?:you|your team|to|us)[^.!?\n]{3,80}", re.I),
    re.compile(r"(?:make sure|remember to|don't forget to|reminder to)\s+[^.!?\n]{3,80}", re.I),
    re.compile(r"(?:action required|action needed|urgent)[:\s-]*[^.!?\n]{3,80}", re.I),
]


def _contains(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


class HeuristicLLM:
    """Keyword-based stand-in for a real model."""

    provider = "stub"
    model = "heuristic-stub-v1"

    def __init__(self, model: str | None = None):
        if model:
            self.model = model

    # ------------------------------------------------------------ interface
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        return self.complete_response(system, user, max_tokens=max_tokens).text

    def complete_response(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000,
        temperature: float | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse:
        if "triage engine" in system or "category" in system.lower():
            payload = self._analyse(user)
        else:
            payload = self._reply(user)
        text = json.dumps(payload) if json_mode else self._render(system, payload)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(
                provider=self.provider,
                model=self.model,
                prompt_tokens=max(1, len(user) // 4),
                completion_tokens=max(1, len(text) // 4),
                latency_ms=5,
            ),
        )

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        required_keys: list[str] | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = self.complete_response(system, user, max_tokens=max_tokens, json_mode=True)
        return json.loads(response.text), response

    # --------------------------------------------------------------- logic
    @staticmethod
    def _body(user: str) -> str:
        if "--- ORIGINAL EMAIL ---" in user:
            return user.split("--- ORIGINAL EMAIL ---", 1)[1]
        return user

    def _analyse(self, user: str) -> dict[str, Any]:
        text = self._body(user)
        subject = ""
        haystack = f"{subject}\n{text}"

        category = self._classify(haystack)
        priority = self._priority(haystack, category)
        actions = self._action_items(text)
        needs_reply = bool(actions) or any(word in haystack.lower() for word in QUESTION_WORDS)
        if category == Category.SPAM:
            needs_reply = False
        if category in (Category.NEWSLETTER, Category.PROMOTION):
            needs_reply = False

        return {
            "category": str(category),
            "priority": priority,
            "priority_reason": self._reason(priority),
            "summary": self._summary(haystack),
            "action_items": actions,
            "needs_reply": needs_reply,
            "sentiment": "neutral",
            "language": "en",
        }

    @staticmethod
    def _classify(text: str) -> Category:
        lowered = text.lower()
        if _contains(lowered, SPAM_WORDS):
            return Category.SPAM
        if _contains(lowered, PROMO_WORDS) and not _contains(lowered, WORK_WORDS):
            return Category.PROMOTION
        if _contains(lowered, NEWSLETTER_WORDS) and not _contains(lowered, WORK_WORDS):
            return Category.NEWSLETTER
        if _contains(lowered, PERSONAL_WORDS) and not _contains(lowered, WORK_WORDS):
            return Category.PERSONAL
        if _contains(lowered, WORK_WORDS):
            return Category.WORK
        if _contains(lowered, URGENT_WORDS):
            return Category.IMPORTANT
        return Category.PERSONAL

    @staticmethod
    def _priority(text: str, category: Category) -> int:
        lowered = text.lower()
        score = 3
        if _contains(lowered, URGENT_WORDS):
            score = 5
        elif "overdue" in lowered or "past due" in lowered:
            score = 4
        elif category in (Category.NEWSLETTER, Category.PROMOTION, Category.SPAM):
            score = 1
        elif category == Category.PERSONAL:
            score = 2
        return score

    @staticmethod
    def _reason(priority: int) -> str:
        return {
            5: "Contains an explicit urgency or deadline marker.",
            4: "Time-sensitive request that should be handled soon.",
            3: "Standard request with no explicit urgency.",
            2: "Low-impact message.",
            1: "Bulk or non-actionable message.",
        }.get(priority, "Standard priority.")

    @staticmethod
    def _summary(text: str) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        body = " ".join(sentences[1:3]) if len(sentences) > 1 else (sentences[0] if sentences else "")
        return body[:300] or "Message could not be summarised."

    @staticmethod
    def _action_items(text: str) -> list[str]:
        items: list[str] = []
        for pattern in ACTION_PATTERNS:
            for match in pattern.findall(text):
                cleaned = re.sub(r"^\s*(?:please\s+)?", "", match, flags=re.I).strip()
                cleaned = re.sub(r"\s+", " ", cleaned)[:160]
                if cleaned and cleaned.lower() not in {i.lower() for i in items}:
                    items.append(cleaned)
            if items:
                break
        return items[:5]

    def _reply(self, user: str) -> str:
        tone = "Professional"
        match = re.search(r"Tone:\s*(.+)", user)
        if match:
            tone = match.group(1).strip()
        name_match = re.search(r"Sign-off name:\s*(.+)", user)
        name = name_match.group(1).strip() if name_match else "[Your name]"
        opener = {
            "Friendly": "Hi,",
            "Concise": "Hi,",
            "Formal": "Dear Sir or Madam,",
            "Apologetic": "Hi,",
        }.get(tone, "Hi,")
        body = {
            "Friendly": "Thanks for reaching out - happy to help with this.",
            "Concise": "Thanks for your message. I will handle this.",
            "Formal": "Thank you for your correspondence. I acknowledge receipt of your message.",
            "Apologetic": "Thank you for your patience, and apologies for the delay.",
            "Professional": (
                "Thank you for your message. I have noted the details and will follow up shortly."
            ),
        }.get(tone, "Thank you for your message.")
        return f"{opener}\n\n{body}\n\nBest regards,\n{name}"

    @staticmethod
    def _render(system: str, payload: dict[str, Any]) -> str:
        if "triage engine" in system:
            return json.dumps(payload)
        return str(payload)


class ScriptedLLM:
    """Returns pre-baked responses in order; useful for exact assertions."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        self.calls.append((system, user))
        if not self.responses:
            return "{}"
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
