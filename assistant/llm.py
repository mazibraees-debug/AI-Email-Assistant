"""LLM abstraction supporting OpenRouter and Anthropic APIs."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self, model: str | None = None):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.openrouter_key and not self.anthropic_key:
            raise RuntimeError(
                "Neither OPENROUTER_API_KEY nor ANTHROPIC_API_KEY is set. "
                "Copy .env.example to .env and set your API key."
            )

        if self.openrouter_key:
            self.provider = "openrouter"
            self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5")
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            try:
                import openai
                self._openai_client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.openrouter_key,
                )
            except ImportError:
                self._openai_client = None
        else:
            self.provider = "anthropic"
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            except ImportError:
                raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        if self.provider == "openrouter":
            return self._complete_openrouter(system, user, max_tokens)
        else:
            return self._complete_anthropic(system, user, max_tokens)

    def _complete_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _complete_openrouter(self, system: str, user: str, max_tokens: int) -> str:
        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        # Fallback to standard library urllib if openai package is not installed
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Email Assistant",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("error", {}).get("message", err_body)
            except Exception:
                msg = err_body
            raise RuntimeError(f"OpenRouter API error ({e.code}): {msg}") from e
