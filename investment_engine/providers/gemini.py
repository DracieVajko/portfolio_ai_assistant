from __future__ import annotations

import os
import requests

from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider – uses free-tier models like gemini-2.5-flash."""

    @property
    def name(self) -> str:
        return "Gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        max_output_tokens: int = 8192,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        if not self.api_key:
            return f"[{stage}] Gemini provider is not configured."

        model = (context or {}).get("model", self.model) if context else self.model
        # Gemini generateContent endpoint - OpenAI-compatible via ?key=
        # Use :generateContent
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.15,
                "maxOutputTokens": min(self.max_output_tokens, 8192),
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # Gemini response structure: candidates[0].content.parts[0].text
            candidates = data.get("candidates") or []
            if not candidates:
                # Check for error with free quota
                if "error" in data:
                    return f"[{stage}] Gemini error: {data['error']}"
                return f"[{stage}] Gemini returned no candidates."
            content = candidates[0].get("content", {})
            parts = content.get("parts") or []
            if not parts:
                return f"[{stage}] Gemini returned no content."
            text = parts[0].get("text", "")
            return str(text).strip() or f"[{stage}] Gemini returned empty response."
        except requests.exceptions.HTTPError as e:
            # Handle quota / rate limit – bubble as exception for fallback to trigger
            body = e.response.text[:500] if e.response is not None else str(e)
            raise RuntimeError(f"Gemini HTTP {e.response.status_code if e.response else ''}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}") from e
