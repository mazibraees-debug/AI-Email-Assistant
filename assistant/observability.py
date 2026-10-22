"""Structured logging, tracing and lightweight metrics.

Design goals
------------
* One log line == one JSON object, so logs are queryable without a parser.
* Every LLM call is a span: it knows its model, tokens, latency and cost.
* Metrics are aggregatable in-process and exportable in Prometheus text format.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

LOG_FORMAT = "%(message)s"

# Approximate USD per 1M tokens. Used for cost attribution and budget alerts.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model id -> (prompt usd / 1M, completion usd / 1M)
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "anthropic/claude-3.5-sonnet": (3.0, 15.0),
    "anthropic/claude-3.5-haiku": (0.8, 4.0),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-4.1": (2.0, 8.0),
    "openai/gpt-4.1-mini": (0.4, 1.6),
    "google/gemini-2.0-flash-001": (0.1, 0.4),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.3),
    "mistralai/mistral-large": (2.0, 6.0),
    "google/gemma-3-27b-it": (0.1, 0.2),
    "anthropic/claude-sonnet-5.5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "gpt-4o-mini": (0.15, 0.6),
}
DEFAULT_PRICING = (1.0, 3.0)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort cost in USD for a completion."""
    key = (model or "").lower()
    pricing = DEFAULT_PRICING
    for candidate, value in MODEL_PRICING.items():
        if candidate in key:
            pricing = value
            break
    prompt_rate, completion_rate = pricing
    return (prompt_tokens / 1_000_000) * prompt_rate + (completion_tokens / 1_000_000) * completion_rate


@dataclass
class LLMUsage:
    """Accounting record for a single LLM call."""

    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


@dataclass
class Span:
    """A timed unit of work with structured attributes."""

    name: str
    trace_id: str
    span_id: str
    started_at: float = field(default_factory=time.time)
    duration_ms: int = 0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
        }


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in getattr(record, "context", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False
_lock = threading.Lock()


def configure_logging(level: str | None = None, json_output: bool = True) -> None:
    """Install the JSON handler on the root logger (idempotent)."""
    global _configured
    with _lock:
        root = logging.getLogger()
        lvl = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
        root.setLevel(lvl)
        if _configured:
            return
        handler = logging.StreamHandler()
        if json_output:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.handlers = [handler]
        _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


class Tracer:
    """Creates correlated spans and emits one log line per span."""

    def __init__(self, service: str = "ai-email-assistant"):
        self.service = service
        self.log = get_logger("tracer")

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex[:16]

    @contextmanager
    def span(self, name: str, trace_id: str | None = None, **attributes: Any) -> Iterator[Span]:
        trace_id = trace_id or self.new_trace_id()
        span = Span(name=name, trace_id=trace_id, span_id=uuid.uuid4().hex[:12], attributes=dict(attributes))
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.attributes["error"] = str(exc)
            raise
        finally:
            span.duration_ms = int((time.perf_counter() - start) * 1000)
            self.log.info("span", extra={"context": {"service": self.service, **span.to_dict()}})

    def event(self, name: str, **payload: Any) -> None:
        self.log.info(name, extra={"context": {"service": self.service, **payload}})


class MetricsRegistry:
    """Thread-safe counters and timers with Prometheus-style export."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._timings: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._timings.setdefault(name, []).append(value)

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timers = {
                name: {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "p95_ms": sorted(values)[min(len(values) - 1, int(len(values) * 0.95))],
                    "max_ms": max(values),
                }
                for name, values in self._timings.items()
                if values
            }
            return {"counters": dict(self._counters), "timings": timers, "gauges": dict(self._gauges)}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()
            self._gauges.clear()

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines: list[str] = []
        for name, value in snap["counters"].items():
            metric = _prom_name(name)
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
        for name, stats in snap["timings"].items():
            metric = _prom_name(name)
            lines.append(f"# TYPE {metric} summary")
            lines.append(f"{metric}_count {stats['count']}")
            lines.append(f"{metric}_avg {stats['avg_ms']:.3f}")
            lines.append(f"{metric}_p95 {stats['p95_ms']:.3f}")
            lines.append(f"{metric}_max {stats['max_ms']:.3f}")
        for name, value in snap["gauges"].items():
            metric = _prom_name(name)
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {value}")
        return "\n".join(lines) + "\n"


def _prom_name(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    return "aiea_" + "".join(out).strip("_").lower()


METRICS = MetricsRegistry()
TRACER = Tracer()


def jitter(base: float, spread: float = 0.3) -> float:
    """Exponential backoff with jitter to avoid thundering herds."""
    return base * (1 - spread) + random.random() * base * spread  # noqa: S311 - jitter, not cryptography
