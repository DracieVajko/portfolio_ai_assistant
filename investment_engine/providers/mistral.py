from __future__ import annotations

import os
import requests

from .base import BaseLLMProvider


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider – OpenAI-compatible, free tier models like open-mistral-nemo."""

    @property
    def name(self) -> str:
        return "Mistral"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "open-mistral-nemo",
        base_url: str = "https://api.mistral.ai/v1",
        max_output_tokens: int = 8192,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        if not self.api_key:
            return f"[{stage}] Mistral provider is not configured."

        model = (context or {}).get("model", self.model) if context else self.model

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"You are the {stage} stage of the investment engine."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "max_tokens": min(self.max_output_tokens, 8192),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return f"[{stage}] Mistral returned no choices."
            content = choices[0].get("message", {}).get("content", "")
            return str(content).strip() or f"[{stage}] Mistral returned empty response."
        except requests.exceptions.HTTPError as e:
            body = e.response.text[:500] if e.response is not None else str(e)
            raise RuntimeError(f"Mistral HTTP {e.response.status_code if e.response else ''}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Mistral error: {e}") from e
