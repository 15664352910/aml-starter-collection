from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleLLM:
    """Small OpenAI-compatible client for OpenRouter and equivalent gateways."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_tokens: int,
    ) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is required for agent retrieval")
        self.api_key = api_key
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]]) -> LLMReply:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/imlrz/ReFind",
            "X-Title": "ReFind Agent Memory",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                return LLMReply(
                    text=body["choices"][0]["message"].get("content") or "",
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                )
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"LLM request failed after 3 attempts: {last_error}"
        ) from last_error
