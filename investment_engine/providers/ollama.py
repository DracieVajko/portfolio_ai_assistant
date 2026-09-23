from __future__ import annotations

from typing import Any

import requests

from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama-compatible provider for the modular investment engine."""

    @property
    def name(self) -> str:
        return "Ollama"

    def __init__(
        self,
        base_url: str = "http://100.101.20.64:1234",
        model: str = "gemma4:12b",
        max_context_tokens: int = 128000,
        max_output_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are the {stage} stage of the investment engine."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.15,
                "max_tokens": min(self.max_output_tokens, 4096),
            },
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                if content := data.get("message", {}).get("content"):
                    return str(content).strip()
                if output := data.get("output"):
                    if isinstance(output, list) and output:
                        first = output[0]
                        if isinstance(first, dict) and first.get("content"):
                            return str(first.get("content") or "").strip()
                        if isinstance(first, str):
                            return first.strip()
            return f"[{stage}] Ollama returned no content."
        except Exception as exc:  # pragma: no cover - defensive fallback
            return f"[{stage}] Ollama error: {exc}"

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
