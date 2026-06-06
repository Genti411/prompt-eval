"""Pluggable LLM backends. MockLLM (a responder function) is used by tests so the
framework is verified deterministically with no API cost; the real backends are
used to evaluate/optimize prompts against an actual model."""
from __future__ import annotations

import os
from typing import Callable


class MockLLM:
    def __init__(self, responder: Callable[[list], str]):
        self.responder = responder

    def complete(self, messages) -> str:
        return self.responder(messages)


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, messages) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers,
                          json={"model": self.model, "messages": messages, "temperature": 0},
                          timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicLLM:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def complete(self, messages) -> str:
        import requests
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = [m for m in messages if m["role"] != "system"]
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": self.model, "max_tokens": 256, "system": system,
                                "messages": conv}, timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


def from_env():
    backend = os.environ.get("LLM", "").lower()
    if backend == "anthropic":
        return AnthropicLLM(os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                            os.environ["ANTHROPIC_API_KEY"])
    if backend in ("openai", "ollama"):
        return OpenAICompatLLM(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                               os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                               os.environ.get("OPENAI_API_KEY"))
    raise RuntimeError("Set LLM=anthropic|openai|ollama (+ API env). "
                       "Free local: LLM=ollama OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_MODEL=llama3.1")
