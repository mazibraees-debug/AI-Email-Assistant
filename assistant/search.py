"""Keyword + TF-IDF search over the stored inbox.

Used by ``EmailAssistant.ask`` (RAG-lite "ask your inbox") and the
``/api/v1/search`` endpoint. Falls back to simple token scoring when
scikit-learn is unavailable or there are too few emails to vectorise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .schemas import Email

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "this",
    "that",
    "these",
    "those",
    "hi",
    "hello",
    "dear",
    "regards",
    "best",
    "thanks",
    "thank",
    "sorry",
    "please",
    "team",
    "myself",
    "our",
    "your",
    "their",
    "my",
    "me",
    "us",
    "them",
    "his",
    "her",
    "its",
}


@dataclass
class SearchHit:
    email: Email
    score: float


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOPWORDS and len(t) > 2]


def _simple_score(query_tokens: list[str], email: Email) -> float:
    haystack = " ".join(t for t in [email.subject, email.sender, email.body] if t).lower()
    if not query_tokens:
        return 0.0
    matches = sum(1 for t in query_tokens if t in haystack)
    return matches / len(query_tokens)


def search_emails(db: Any, query: str, top_k: int = 5) -> list[SearchHit]:
    """Return up to ``top_k`` emails most relevant to ``query``."""
    if not query or not query.strip():
        return []
    query_tokens = _tokens(query)

    rows = db.list_emails(limit=500)  # list_emails returns (email, analysis|None)
    emails = [e for e, _ in rows]
    if not emails:
        return []

    scored: list[tuple[float, Email]] = []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        docs = [f"{e.subject}\n{e.body}" for e in emails]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
        matrix = vectorizer.fit_transform(docs)
        qvec = vectorizer.transform([query])
        sims = cosine_similarity(qvec, matrix).ravel()
        for score, email in zip(sims, emails, strict=False):
            scored.append((float(score), email))
    except Exception:  # noqa: BLE001 - scikit-learn missing or too small
        for email in emails:
            scored.append((_simple_score(query_tokens, email), email))

    scored.sort(key=lambda pair: -pair[0])
    return [SearchHit(email=email, score=score) for score, email in scored[:top_k] if score > 0]
