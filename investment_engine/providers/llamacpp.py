from __future__ import annotations

from typing import Any

import requests

from .base import BaseLLMProvider


class LlamaCppProvider(BaseLLMProvider):
    """llama.cpp / OpenAI-compatible provider for the modular investment engine.

    Connects to a llama.cpp server running with OpenAI-compatible API (v1/chat/completions).
    """

    @property
    def name(self) -> str:
        return "llama.cpp"

    def __init__(
        self,
        base_url: str = "http://100.125.47.31:11435/v1",
        model: str = "qwen3-4b-pi",
        max_context_tokens: int = 32768,
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
            "temperature": 0.15,
            "max_tokens": min(self.max_output_tokens, 4096),
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    if content := msg.get("content"):
                        return str(content).strip()
            return f"[{stage}] llama.cpp returned no content."
        except Exception as exc:  # pragma: no cover - defensive fallback
            return f"[{stage}] llama.cpp error: {exc}"

    def is_available(self) -> bool:
        """Check availability via Tailscale IP."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=3)
            return response.status_code == 200
        except Exception:
            return False