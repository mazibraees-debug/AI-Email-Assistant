"""REST API for the AI Email Assistant.

Run with:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The service boots offline (using a deterministic heuristic stub) when no LLM
provider is configured, so it can be demonstrated and tested without a paid
model.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from assistant import EmailAssistant
from assistant.llm import LLMError
from assistant.mailer import configured as mailbox_configured
from assistant.observability import METRICS, get_logger
from assistant.schemas import HealthStatus
from assistant.storage import Database, get_db

log = get_logger("api")

API_KEY = os.getenv("API_KEY", "")

assistant: EmailAssistant | None = None
db: Database | None = None


# ------------------------------------------------------------------ models
class EmailIn(BaseModel):
    sender: str = ""
    subject: str = ""
    body: str = ""
    date: str | None = None
    message_id: str | None = None


class TriageBatchIn(BaseModel):
    emails: list[EmailIn] = Field(default_factory=list)


class DraftIn(BaseModel):
    tone: str = "Professional"
    instructions: str = ""


class FeedbackIn(BaseModel):
    draft_id: int = Field(ge=1)
    rating: int = Field(ge=1, le=5)
    correction: str = ""
    notes: str = ""


class SearchIn(BaseModel):
    query: str
    top_k: int = Field(ge=1, le=20, default=5)


class AskIn(BaseModel):
    question: str
    top_k: int = Field(ge=1, le=20, default=5)


class EvalRunIn(BaseModel):
    model: str | None = None
    dataset_path: str | None = None


# ------------------------------------------------------------------ auth
def require_api_key(request: Request) -> None:
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key header",
        )


# ------------------------------------------------------------------ lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - FastAPI lifecycle signature
    global assistant, db
    db = get_db()
    try:
        assistant = EmailAssistant(db=db)
    except LLMError as exc:
        log.warning(
            "no LLM provider configured - falling back to offline stub",
            extra={"context": {"error": str(exc)}},
        )
        from assistant.testing import HeuristicLLM

        assistant = EmailAssistant(llm=HeuristicLLM(), db=db)
    yield
    if db is not None:
        db.close()


app = FastAPI(
    title="AI Email Assistant API",
    version="1.0.0",
    description="Production-grade email triage, drafting and inbox intelligence API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):  # noqa: D401
    request.state.request_id = uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - keep every failure shaped as JSON
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = getattr(request.state, "request_id", "-")
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


# ------------------------------------------------------------------ endpoints
@app.get("/health", response_model=None)
async def health(_: None = Depends(require_api_key)):
    try:
        mailbox = "configured" if mailbox_configured() else "not_configured"
    except Exception:  # noqa: BLE001
        mailbox = "error"
    provider = "none"
    model = ""
    if assistant is not None and getattr(assistant, "llm", None) is not None:
        provider = getattr(assistant.llm, "provider", "unknown")
        model = getattr(assistant.llm, "model", "")
    return HealthStatus(
        status="ok",
        database="ok",
        llm_provider=provider,
        llm_model=str(model),
        mailbox=mailbox,
    ).model_dump(mode="json")


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(_: None = Depends(require_api_key)):
    return METRICS.prometheus()


@app.get("/api/v1/stats")
async def stats(_: None = Depends(require_api_key)):
    return {"stats": db.stats() if db else {}, "recent_runs": db.recent_runs(10) if db else []}


@app.post("/api/v1/emails")
async def triage_email(body: EmailIn, _: None = Depends(require_api_key)):
    from assistant.extractor import Email

    email = Email(
        sender=body.sender,
        subject=body.subject,
        body=body.body,
        date=body.date or "",
        message_id=body.message_id or "",
        source="api",
    )
    result = assistant.triage(email)  # type: ignore[union-attr]
    return result.to_dict()


@app.post("/api/v1/emails/triage")
async def triage_batch(body: TriageBatchIn, _: None = Depends(require_api_key)):
    from assistant.extractor import Email

    emails = [
        Email(
            sender=item.sender,
            subject=item.subject,
            body=item.body,
            date=item.date or "",
            message_id=item.message_id or "",
            source="api",
        )
        for item in body.emails
    ]
    results = assistant.process_inbox_detailed(emails)  # type: ignore[union-attr]
    return [r.to_dict() for r in results]


@app.get("/api/v1/emails")
async def list_emails(
    category: str | None = None,
    needs_reply: str | None = None,
    min_priority: int | None = None,
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _: None = Depends(require_api_key),
):
    rows = db.list_emails(  # type: ignore[union-attr]
        category=category,
        needs_reply=(needs_reply.lower() == "true") if needs_reply else None,
        min_priority=min_priority,
        search=q,
        limit=min(limit, 500),
        offset=offset,
    )
    return [
        {
            "email": e.model_dump(mode="json"),
            "analysis": a.to_dict() if a else None,
        }
        for e, a in rows
    ]


@app.get("/api/v1/emails/{email_id}")
async def get_email(email_id: int, _: None = Depends(require_api_key)):
    email = db.get_email(email_id)  # type: ignore[union-attr]
    if not email:
        raise HTTPException(status_code=404, detail="email not found")
    analysis = db.latest_analysis(email_id)
    drafts = db.drafts_for_email(email_id)
    feedback = []
    for draft in drafts:
        if draft.id is not None:
            feedback.extend(db.feedback_for_draft(draft.id))
    return {
        "email": email.model_dump(mode="json"),
        "analysis": analysis.to_dict() if analysis else None,
        "drafts": [d.model_dump(mode="json") for d in drafts],
        "feedback": [f.model_dump(mode="json") for f in feedback],
    }


@app.post("/api/v1/emails/{email_id}/draft")
async def create_draft(email_id: int, body: DraftIn, _: None = Depends(require_api_key)):
    email = db.get_email(email_id)  # type: ignore[union-attr]
    if not email:
        raise HTTPException(status_code=404, detail="email not found")
    analysis = db.latest_analysis(email_id)
    draft = assistant.generate_draft(  # type: ignore[union-attr]
        email=email,
        analysis=analysis,
        tone=body.tone,
        instructions=body.instructions,
    )
    return draft.model_dump(mode="json")


@app.post("/api/v1/drafts/{draft_id}/approve")
async def approve_draft(draft_id: int, _: None = Depends(require_api_key)):
    draft = db.get_draft(draft_id)  # type: ignore[union-attr]
    if not draft:
        raise HTTPException(status_code=404, detail="draft not found")
    db.execute("UPDATE drafts SET approved = 1 WHERE id = ?", (draft_id,))  # type: ignore[union-attr]
    METRICS.increment("draft.approved")
    return {"id": draft_id, "approved": True}


@app.post("/api/v1/drafts/{draft_id}/send")
async def send_draft(draft_id: int, _: None = Depends(require_api_key)):
    draft = db.get_draft(draft_id)  # type: ignore[union-attr]
    if not draft:
        raise HTTPException(status_code=404, detail="draft not found")
    if not mailbox_configured():
        raise HTTPException(status_code=409, detail="mailbox is not configured; cannot send")

    original = db.get_email(draft.email_id or 0)  # type: ignore[union-attr]
    if not original:
        raise HTTPException(status_code=404, detail="original email not found")
    try:
        from assistant.mailer import send_reply

        send_reply(original, draft.body)
        db.mark_draft_sent(draft_id)  # type: ignore[union-attr]
        METRICS.increment("draft.sent")
        return {"id": draft_id, "sent": True}
    except Exception as exc:  # noqa: BLE001 - surface mailer errors as JSON
        raise HTTPException(status_code=502, detail=f"send failed: {exc}") from exc


@app.post("/api/v1/feedback")
async def record_feedback(body: FeedbackIn, _: None = Depends(require_api_key)):
    draft = db.get_draft(body.draft_id)  # type: ignore[union-attr]
    if not draft:
        raise HTTPException(status_code=404, detail="draft not found")
    return assistant.record_feedback(  # type: ignore[union-attr]
        draft=draft,
        rating=body.rating,
        correction=body.correction,
        notes=body.notes,
    )


@app.post("/api/v1/search")
async def search(body: SearchIn, _: None = Depends(require_api_key)):
    try:
        from assistant.search import search_emails

        hits = search_emails(db, body.query, top_k=body.top_k)  # type: ignore[union-attr]
    except ImportError:
        hits = []
    return {
        "query": body.query,
        "results": [
            {
                "score": round(hit.score, 4),
                "email": hit.email.model_dump(mode="json"),
            }
            for hit in hits
        ],
    }


@app.post("/api/v1/ask")
async def ask(body: AskIn, _: None = Depends(require_api_key)):
    return assistant.ask(body.question, top_k=body.top_k)  # type: ignore[union-attr]


@app.get("/api/v1/digest")
async def get_digest(_: None = Depends(require_api_key)):
    return assistant.digest().model_dump(mode="json")  # type: ignore[union-attr]


@app.post("/api/v1/eval/run")
async def run_eval(body: EvalRunIn, _: None = Depends(require_api_key)):
    try:
        from assistant.evaluate import run_evaluation
    except ImportError as exc:  # pragma: no cover - harness always present
        raise HTTPException(status_code=503, detail="evaluation harness not available") from exc
    report = run_evaluation(assistant=assistant, dataset_path=body.dataset_path)
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return report.__dict__


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)  # noqa: S104


# keep exports stable for backwards-compatible imports
__all__ = ["app", "EmailIn", "DraftIn", "FeedbackIn", "SearchIn", "AskIn", "EvalRunIn"]
