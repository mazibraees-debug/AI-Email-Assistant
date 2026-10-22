"""Offline tests for the REST API. No network, no API key required."""

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"


def test_metrics(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")


def test_create_and_list_email(client: TestClient):
    r = client.post(
        "/api/v1/emails",
        json={
            "sender": "Sarah Khan <sarah@acmecorp.com>",
            "subject": "URGENT: Client demo moved to tomorrow 10am",
            "body": "Hi, can you confirm you are available and send the deck by 6pm today?",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "analysis" in data
    assert data["analysis"]["category"] in ("Important", "Work")
    assert data["analysis"]["priority"] >= 1

    r = client.get("/api/v1/emails")
    assert r.status_code == 200
    items = r.json()
    assert any(it["email"]["subject"].startswith("URGENT") for it in items)


def test_filter_by_category_and_priority(client: TestClient):
    client.post(
        "/api/v1/emails",
        json={
            "sender": "deals@megastore.com",
            "subject": "FLASH SALE",
            "body": "Only 3 hours left! Click here to claim your discount.",
        },
    )
    client.post(
        "/api/v1/emails",
        json={
            "sender": "boss@corp.com",
            "subject": "Meeting tomorrow",
            "body": "Are you free for a meeting tomorrow at 2pm?",
        },
    )
    r = client.get("/api/v1/emails", params={"category": "Promotion"})
    assert r.status_code == 200
    items = r.json()
    assert all(it["analysis"]["category"] == "Promotion" for it in items if it["analysis"])

    r = client.get("/api/v1/emails", params={"min_priority": 4})
    assert r.status_code == 200


def test_unknown_email_404(client: TestClient):
    r = client.get("/api/v1/emails/999999")
    assert r.status_code == 404


def test_draft_endpoint(client: TestClient):
    r = client.post(
        "/api/v1/emails",
        json={
            "sender": "boss@corp.com",
            "subject": "Meeting tomorrow",
            "body": "Are you free for a meeting tomorrow at 2pm?",
        },
    )
    assert r.status_code == 200
    r = client.get("/api/v1/emails")
    email_id = r.json()[0]["email"]["id"]
    r = client.post(f"/api/v1/emails/{email_id}/draft", json={"tone": "Concise"})
    assert r.status_code == 200
    assert r.json()["body"]


def test_feedback_endpoint(client: TestClient):
    r = client.post(
        "/api/v1/emails",
        json={
            "sender": "x@y.com",
            "subject": "Hi",
            "body": "Please confirm receipt.",
        },
    )
    r = client.get("/api/v1/emails")
    email_id = r.json()[0]["email"]["id"]
    r = client.post(f"/api/v1/emails/{email_id}/draft", json={"tone": "Professional"})
    draft_id = r.json()["id"]
    r = client.post("/api/v1/feedback", json={"draft_id": draft_id, "rating": 5, "correction": ""})
    assert r.status_code == 200


def test_draft_send_409_without_mailbox(client: TestClient):
    r = client.post("/api/v1/emails", json={"sender": "a@b.com", "subject": "s", "body": "body"})
    r = client.get("/api/v1/emails")
    email_id = r.json()[0]["email"]["id"]
    r = client.post(f"/api/v1/emails/{email_id}/draft", json={"tone": "Professional"})
    draft_id = r.json()["id"]
    r = client.post(f"/api/v1/drafts/{draft_id}/send")
    assert r.status_code == 409


def test_ask_returns_answer(client: TestClient):
    client.post(
        "/api/v1/emails",
        json={
            "sender": "Sarah Khan <sarah@acmecorp.com>",
            "subject": "demo moved",
            "body": "The client asked to move the demo to tomorrow at 10am.",
        },
    )
    r = client.post("/api/v1/ask", json={"question": "When is the demo?", "top_k": 3})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_digest(client: TestClient):
    client.post(
        "/api/v1/emails",
        json={
            "sender": "a@b.com",
            "subject": "urgent invoice",
            "body": "Invoice #4821 for $2,300 is overdue.",
        },
    )
    r = client.get("/api/v1/digest")
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    assert data["email_count"] >= 1
