"""User preferences and the human-feedback learning loop.

The assistant stores every thumbs-up/down and correction in SQLite. Over time
these are folded back into a small "playbook" that is injected into reply
drafts: preferred tone, sign-off, phrases to avoid, and a few positive examples
per tone. Nothing here replaces a human reviewer; it only makes the drafts
match the user's style better.
"""

from __future__ import annotations

from typing import Any

from .schemas import Tone


class PreferenceStore:
    """Reads/writes the user playbook to the ``settings`` table."""

    def __init__(self, db: Any) -> None:
        self.db = db

    # ------------------------------------------------------------- drafting context
    def context_for_drafting(self) -> str:
        """A short block injected into every reply-draft prompt."""
        lines: list[str] = []
        tone = self.get_setting("preferred_tone")
        if tone:
            lines.append(f"User's preferred reply tone: {tone}")
        sign_off = self.get_setting("sign_off_name")
        if sign_off:
            lines.append(f"User's sign-off name: {sign_off}")
        signature = self.get_setting("signature_block")
        if signature:
            lines.append(f"User's signature block:\n{signature}")
        avoid = self.get_setting("phrases_to_avoid")
        if avoid:
            lines.append(f"Phrases to avoid in replies: {', '.join(avoid)}")
        hours = self.get_setting("working_hours")
        if hours:
            lines.append(f"User's working hours: {hours}")
        examples = self.positive_examples_summary()
        if examples:
            lines.append("Approved reply examples to mirror:")
            lines.extend(f"- {e}" for e in examples[:3])
        return "\n".join(lines)

    def positive_examples(self, tone: Tone) -> list[str]:
        """Approved draft bodies for a given tone, newest first."""
        key = f"examples:{str(tone)}"
        raw = self.get_setting(key, [])
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return []

    def positive_examples_summary(self) -> list[str]:
        """Shortened examples across tones, for the drafting context."""
        out: list[str] = []
        for tone in Tone:
            for example in self.positive_examples(tone)[:1]:
                out.append(f"[{str(tone)}] {example[:160]}")
        return out[:5]

    # ------------------------------------------------------------- learning
    def learn(self, feedback: Any, draft: Any) -> None:
        """Fold one piece of human feedback back into the playbook."""
        rating = int(feedback.rating or 0)
        tone = str(getattr(draft, "tone", "") or "")
        body = str(getattr(draft, "body", "") or "")
        correction = str(getattr(feedback, "correction", "") or "")

        if rating >= 4 and tone and body:
            self._append_example(tone, body)

        if rating <= 2 and correction and tone:
            key = f"corrections:{tone}"
            existing = self.get_setting(key, [])
            if not isinstance(existing, list):
                existing = []
            existing.append(correction[:400])
            self.db.set_setting(key, existing)

        # Track the tone the user rates highest as their preference.
        if rating >= 4 and tone:
            self.db.set_setting("preferred_tone", tone)

        if rating >= 4 and body and not self.get_setting("sign_off_name"):
            # Try to extract a sign-off line from an approved draft.
            for line in reversed(body.splitlines()):
                line = line.strip()
                if (
                    line
                    and len(line) < 60
                    and any(w in line.lower() for w in ("regards", "best", "thanks", "cheers", "sincerely"))
                ):
                    self.db.set_setting("sign_off_name", line)
                    break

    def average_rating(self) -> float:
        rows = self.db.query("SELECT AVG(rating) AS avg FROM feedback") if hasattr(self.db, "query") else []
        if not rows:
            return 0.0
        value = rows[0]["avg"]
        return round(float(value), 2) if value is not None else 0.0

    def learned_summary(self) -> dict[str, Any]:
        rows = (
            self.db.query("SELECT COUNT(*) AS n, AVG(rating) AS avg FROM feedback")
            if hasattr(self.db, "query")
            else []
        )
        n = int(rows[0]["n"]) if rows and rows[0]["n"] is not None else 0
        avg = float(rows[0]["avg"]) if rows and rows[0]["avg"] is not None else 0.0
        examples = {str(tone): len(self.positive_examples(tone)) for tone in Tone}
        return {
            "feedback_count": n,
            "average_rating": round(avg, 2),
            "examples_per_tone": examples,
            "preferred_tone": self.get_setting("preferred_tone"),
        }

    # ------------------------------------------------------------- helpers
    def _append_example(self, tone: str, body: str) -> None:
        key = f"examples:{tone}"
        existing = self.get_setting(key, [])
        if not isinstance(existing, list):
            existing = []
        short = " ".join(body.split())[:220]
        if short and short not in existing:
            existing.append(short)
            # Keep only the most recent handful per tone.
            existing = existing[-8:]
        self.db.set_setting(key, existing)

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.db.get_setting(key, default)
