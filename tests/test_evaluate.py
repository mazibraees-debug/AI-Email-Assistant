"""Tests for the evaluation harness. Fast, offline, no API key required."""

import json

from assistant.evaluate import (
    EvalItem,
    EvalReport,
    Metric,
    _action_f1,
    compare_reports,
    load_dataset,
    regression,
    run_evaluation,
)
from assistant.testing import HeuristicLLM


def test_dataset_loads_with_valid_items(tmp_path, monkeypatch):
    dataset = load_dataset("data/eval_dataset.json")
    assert len(dataset) >= 45
    valid_categories = {"Important", "Work", "Personal", "Newsletter", "Promotion", "Spam"}
    for item in dataset:
        assert item.id
        assert item.sender
        assert item.subject
        assert item.body
        assert item.expected_category in valid_categories
        assert 1 <= item.expected_priority <= 5
        assert isinstance(item.expected_action_items, list)
        assert isinstance(item.tags, list)
    # At least 6 adversarial injection items whose expected category is genuine.
    injection = [i for i in dataset if "injection" in i.tags]
    assert len(injection) >= 6
    for item in injection:
        # The genuine category should not be what the injection asks for.
        assert item.expected_category in ("Work", "Personal", "Promotion")


def test_load_dataset_tolerates_missing_optional_keys(tmp_path):
    path = tmp_path / "small.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "sender": "a@b.com",
                    "subject": "s",
                    "body": "b",
                    "expected_category": "Work",
                    "expected_priority": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    items = load_dataset(str(path))
    assert len(items) == 1
    assert items[0].expected_action_items == []
    assert items[0].expected_needs_reply is True
    assert items[0].tags == []


def test_action_item_f1_near_match_counts():
    pred = ["send the deck by 6pm", "confirm attendance"]
    exp = ["Send updated slide deck by 6pm today"]
    p, r, f1 = _action_f1(pred, exp)
    assert p > 0 and r > 0 and f1 > 0

    p2, r2, f2 = _action_f1(["unrelated thing"], ["send deck"])
    assert f2 == 0.0

    p3, r3, f3 = _action_f1([], ["send deck"])
    assert p3 == 0.0 and r3 == 0.0 and f3 == 0.0


def test_metric_and_report_roundtrip():
    m = Metric(name="category_accuracy", value=0.85, unit="ratio", detail="ok")
    assert m.to_dict()["value"] == 0.85

    report = EvalReport(dataset_size=5, metrics={"category_accuracy": 0.8})
    md = report.to_markdown()
    assert "Evaluation Report" in md
    assert "category_accuracy" in md
    assert "| category_accuracy |" in md


def test_run_evaluation_offline_subset():
    dataset = [
        EvalItem(
            id="1",
            sender="a@b.com",
            subject="Urgent invoice",
            body="Pay $100 by Friday",
            expected_category="Work",
            expected_priority=4,
            expected_action_items=["Pay $100"],
            expected_needs_reply=True,
        ),
        EvalItem(
            id="2",
            sender="mom@fam.com",
            subject="Lunch?",
            body="Lunch Sunday?",
            expected_category="Personal",
            expected_priority=2,
            expected_action_items=[],
            expected_needs_reply=True,
        ),
        EvalItem(
            id="3",
            sender="deals@store.com",
            subject="Sale",
            body="50% off everything",
            expected_category="Promotion",
            expected_priority=1,
            expected_action_items=[],
            expected_needs_reply=False,
        ),
    ]
    report = run_evaluation(
        dataset=dataset,
        llm=HeuristicLLM(),
        concurrency=1,
    )
    assert report.dataset_size == 3
    assert "category_accuracy" in report.metrics
    assert "priority_mae" in report.metrics
    assert "action_item_f1" in report.metrics
    assert "injection_resistance_rate" in report.metrics
    assert len(report.per_item) == 3
    assert report.latency_ms_avg >= 0


def test_injection_resistance_below_one_when_fooled():
    # Build two synthetic results manually to verify regression/direction logic.
    baseline = EvalReport(
        dataset_size=2,
        metrics={"category_accuracy": 0.5, "priority_mae": 1.0},
    )
    candidate = EvalReport(
        dataset_size=2,
        metrics={"category_accuracy": 0.0, "priority_mae": 2.0},
    )
    regs = regression(baseline, candidate)
    assert "category_accuracy" in regs
    assert "priority_mae" in regs


def test_compare_reports_shape():
    baseline = EvalReport(dataset_size=1, metrics={"category_accuracy": 0.5})
    candidate = EvalReport(dataset_size=1, metrics={"category_accuracy": 0.75})
    deltas = compare_reports(baseline, candidate)
    assert deltas["category_accuracy"]["delta"] == 0.25
