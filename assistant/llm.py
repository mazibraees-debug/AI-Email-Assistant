"""LLM client with retries, cost accounting, caching and model fallback.

Supported providers (auto-selected, override with ``LLM_PROVIDER``):

* ``openrouter`` - any model behind the OpenRouter gateway
* ``openai``     - the OpenAI API directly
* ``anthropic``  - the Anthropic API directly
* ``local``      - any OpenAI-compatible server (Ollama, vLLM, LM Studio)

Every call returns usage metadata so cost and latency can be attributed per
email, per run and per tenant.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from .observability import METRICS, TRACER, LLMUsage, estimate_cost, get_logger, jitter

load_dotenv()

log = get_logger("llm")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    """A single completion plus its accounting metadata."""

    text: str = ""
    provider: str = ""
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: Any = None
    cached: bool = False
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "cached": self.cached,
            "attempts": self.attempts,
        }


class ResponseCache:
    """Small thread-safe TTL cache; collapses repeated prompts in evals."""

    def __init__(self, ttl_seconds: int = 900, max_entries: int = 512):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: dict[str, tuple[float, LLMResponse]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, system: str, user: str, max_tokens: int) -> str:
        raw = f"{model}|{max_tokens}|{system}|{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                self.misses += 1
                return None
            expires, response = entry
            if expires < time.time():
                self._data.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return response

    def put(self, key: str, response: LLMResponse) -> None:
        with self._lock:
            if len(self._data) >= self.max_entries:
                oldest = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                self._data.pop(oldest, None)
            self._data[key] = (time.time() + self.ttl, response)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


class LLMError(RuntimeError):
    """Raised when every provider/model attempt failed."""


class LLM:
    """Provider-agnostic chat completion client."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        use_cache: bool = True,
    ):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.local_base = os.getenv("LOCAL_LLM_BASE_URL", "").strip()

        provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
        if not provider:
            if self.openrouter_key:
                provider = "openrouter"
            elif self.openai_key:
                provider = "openai"
            elif self.anthropic_key:
                provider = "anthropic"
            elif self.local_base:
                provider = "local"
            else:
                raise LLMError(
                    "No LLM provider configured. Set OPENROUTER_API_KEY, OPENAI_API_KEY, "
                    "ANTHROPIC_API_KEY or LOCAL_LLM_BASE_URL in .env."
                )

        self.provider = provider
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache = ResponseCache(ttl_seconds=int(os.getenv("LLM_CACHE_TTL", "900")))
        self.use_cache = use_cache

        if provider == "openrouter":
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            self.api_key = self.openrouter_key
            self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        elif provider == "openai":
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            self.api_key = self.openai_key
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == "local":
            self.base_url = self.local_base.rstrip("/")
            self.api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
            self.model = model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
        else:
            self.base_url = ""
            self.api_key = self.anthropic_key
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

        env_fallback = os.getenv("LLM_FALLBACK_MODELS", "").strip()
        self.fallback_models = fallback_models or (
            [m.strip() for m in env_fallback.split(",") if m.strip()] if env_fallback else []
        )

        self._openai_client = None
        self._anthropic_client = None
        self._init_clients()

    # ------------------------------------------------------------------ init
    def _init_clients(self) -> None:
        if self.provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMError("anthropic package is not installed. Run: pip install anthropic") from exc
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        else:
            try:
                import openai
            except ImportError:
                self._openai_client = None
                return
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.provider == "openrouter":
                kwargs["base_url"] = self.base_url
                kwargs["default_headers"] = {
                    "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8501"),
                    "X-Title": os.getenv("APP_NAME", "AI Email Assistant"),
                }
            elif self.provider == "local":
                kwargs["base_url"] = self.base_url
            self._openai_client = openai.OpenAI(**kwargs)

    # -------------------------------------------------------------- public API
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Backwards-compatible text completion."""
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
        """Run a completion, retrying transient failures and falling back."""
        target_model = model or self.model
        cache_key = ResponseCache.key(target_model, system, user, max_tokens)
        if self.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                METRICS.increment("llm.cache_hit")
                cached.cached = True
                return cached

        attempts = 0
        models = [target_model] + [m for m in self.fallback_models if m != target_model]
        last_error: Exception | None = None

        for candidate in models:
            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                attempts += 1
                try:
                    response = self._call(system, user, max_tokens, temperature, json_mode, candidate)
                    response.attempts = attempts
                    if self.use_cache:
                        self.cache.put(cache_key, response)
                    METRICS.increment("llm.calls")
                    METRICS.observe("llm.latency_ms", response.usage.latency_ms)
                    METRICS.gauge(
                        "llm.cost_usd_total",
                        METRICS.snapshot()["gauges"].get("llm.cost_usd_total", 0.0) + response.usage.cost_usd,
                    )
                    return response
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    last_error = exc
                    status = getattr(exc, "status_code", None)
                    retryable = status in RETRYABLE_STATUS or status is None
                    log.warning(
                        "llm call failed",
                        extra={
                            "context": {
                                "provider": self.provider,
                                "model": candidate,
                                "attempt": attempt,
                                "status": status,
                                "error": str(exc)[:300],
                            }
                        },
                    )
                    if not retryable or attempt >= self.max_retries:
                        break
                    time.sleep(jitter(min(2.0 * (2 ** (attempt - 1)), 8.0)))
            if models.index(candidate) < len(models) - 1:
                log.warning("falling back to next model", extra={"context": {"from": candidate}})

        METRICS.increment("llm.errors")
        raise LLMError(f"All LLM attempts failed: {last_error}") from last_error

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        required_keys: list[str] | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Completion that is guaranteed to parse into a dict."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.complete_response(
                    system,
                    user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=True,
                )
                parsed = parse_json_object(response.text)
                missing = [k for k in (required_keys or []) if k not in parsed]
                if missing:
                    last_error = ValueError(f"missing keys: {missing}")
                    continue
                return parsed, response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(jitter(min(1.5 * (2**attempt), 6.0)))
        raise LLMError(f"Could not obtain valid JSON from the model: {last_error}") from last_error

    # ----------------------------------------------------------------- calls
    def _call(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None,
        json_mode: bool,
        model: str,
    ) -> LLMResponse:
        started = time.perf_counter()
        if self.provider == "anthropic":
            text, usage_in = self._call_anthropic(system, user, max_tokens, temperature, model)
        else:
            text, usage_in = self._call_openai_compatible(
                system, user, max_tokens, temperature, json_mode, model
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        prompt_tokens = int(usage_in.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_in.get("completion_tokens", 0) or 0)
        usage = LLMUsage(
            provider=self.provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            cost_usd=estimate_cost(model, prompt_tokens, completion_tokens),
        )
        TRACER.event(
            "llm.completion",
            provider=self.provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            cost_usd=round(usage.cost_usd, 6),
        )
        return LLMResponse(text=text, provider=self.provider, model=model, usage=usage)

    def _call_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float | None, model: str
    ) -> tuple[str, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._anthropic_client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        usage = getattr(resp, "usage", None)
        return text, {
            "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        }

    def _call_openai_compatible(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None,
        json_mode: bool,
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if self._openai_client is not None:
            resp = self._openai_client.chat.completions.create(**payload)
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            return text, {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            }

        return self._call_urllib(payload, model)

    def _call_urllib(self, payload: dict[str, Any], model: str) -> tuple[str, dict[str, Any]]:
        """stdlib fallback used when the openai package is unavailable."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(body).get("error", {}).get("message", body)
            except Exception:  # noqa: BLE001
                message = body
            error = LLMError(f"{self.provider} API error ({exc.code}): {message}")
            error.status_code = exc.code  # type: ignore[attr-defined]
            raise error from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"{self.provider} request failed: {exc.reason}") from exc

        choice = (result.get("choices") or [{}])[0]
        text = (choice.get("message", {}).get("content") or "").strip()
        usage = result.get("usage") or {}
        return text, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }


def parse_json_object(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Last resort: single quotes / trailing commas / unquoted keys.
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate if start != -1 else cleaned)
    repaired = re.sub(r"(?<![\w\"'])(\w+)\s*:", r'"\1":', repaired)
    repaired = repaired.replace("'", '"')
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    raise ValueError(f"Model did not return a JSON object: {text[:200]}")
