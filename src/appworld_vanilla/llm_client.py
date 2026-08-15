"""Thin OpenAI-compatible chat client for a locally served model.

Works with vLLM (`vllm serve`), SGLang, TGI's OpenAI shim, llama.cpp server,
Ollama's /v1 endpoint - anything exposing POST {base_url}/chat/completions.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from .config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_s: float = 0.0


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], seed: int | None = None) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        )

    def _payload(self, messages, seed) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.stop:
            body["stop"] = self.cfg.stop
        if self.cfg.send_seed and seed is not None:
            body["seed"] = int(seed)
        body.update(self.cfg.extra_body or {})
        return body

    def chat(self, messages: list[dict[str, str]], seed: int | None = None) -> LLMResponse:
        payload = self._payload(messages, seed)
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            start = time.time()
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.cfg.timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return LLMResponse(
                    text=choice["message"].get("content") or "",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", ""),
                    latency_s=time.time() - start,
                )
            except Exception as exc:  # noqa: BLE001 - retry everything transient
                last_err = exc
                if attempt == self.cfg.max_retries:
                    break
                sleep = (self.cfg.retry_backoff ** attempt) * (1.0 + 0.3 * random.random())
                time.sleep(sleep)
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries + 1} attempts: {last_err}")

    def health_check(self) -> dict[str, Any]:
        models_url = self.cfg.base_url.rstrip("/") + "/models"
        info: dict[str, Any] = {"models_endpoint": models_url}
        try:
            r = self.session.get(models_url, timeout=30)
            info["models_status"] = r.status_code
            info["served_models"] = [m.get("id") for m in r.json().get("data", [])]
        except Exception as exc:  # noqa: BLE001
            info["models_error"] = str(exc)
        probe = self.chat([{"role": "user", "content": "Reply with the single word: ok"}], seed=0)
        info["probe_reply"] = probe.text.strip()[:200]
        info["probe_latency_s"] = round(probe.latency_s, 2)
        return info


class ScriptedClient:
    """Offline stand-in for smoke tests: replays a fixed list of outputs."""

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.i = 0

    def chat(self, messages, seed: int | None = None) -> LLMResponse:
        text = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        return LLMResponse(text=text, finish_reason="stop")


def build_client(cfg: LLMConfig) -> ChatClient:
    return OpenAICompatibleClient(cfg)
