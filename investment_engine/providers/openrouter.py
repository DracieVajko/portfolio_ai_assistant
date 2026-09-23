from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

from .base import BaseLLMProvider


def _clean_llm_output(content: str) -> str:
    """Clean LLM output: strip <answer> tags, markdown fences, chain-of-thought."""
    if not content:
        return ""
    content = str(content).strip()
    # Strip <answer>...</answer> tags
    if content.startswith("<answer>") and content.endswith("</answer>"):
        content = content[8:-9].strip()
    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = '\n'.join(lines).strip()
    # Strip chain-of-thought / thinking tags (some models use <think)
    content = re.sub(r'.*?', '', content, flags=re.DOTALL).strip()
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL).strip()
    return content


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter-compatible provider for the modular investment engine."""

    @property
    def name(self) -> str:
        return "OpenRouter"

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1", model: str = "openai/gpt-oss-20b", api_key: str | None = None, max_context_tokens: int = 62000, max_output_tokens: int = 8192) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens

    # Curated free models on OpenRouter (Sep 2026, verified live).
    # Will try in order if model == "free" or "openrouter/free".
    # "openrouter/free" router itself is first; explicit IDs below are fallbacks.
    FREE_MODELS = [
        "openrouter/free",
        "openai/gpt-oss-20b:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3.5-lightning:free",
        "cohere/north-mini-code:free",
        "z-ai/glm-5.2:free",
        "poolside/laguna-s-2.1:free",
        "poolside/laguna-xs-2.1:free",
        "liquid/lfm-2.5-2.6b:free",
    ]

    def _models_to_try(self) -> list[str]:
        m = (self.model or "").strip()
        if m.lower() in ("free", "openrouter/free", "openrouter/free:free", ""):
            return self.FREE_MODELS
        # If user set specific model, use ONLY that model - no fallback to free models
        return [m]

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        if not self.api_key:
            return f"[{stage}] OpenRouter provider is not configured."

        models = self._models_to_try()
        # When using a specific model (not free tier), don't fall back to other models
        use_single_model = len(models) == 1 and models[0] != "openrouter/free"
        
        for model in models:
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

            max_retries = 3
            base_delay = 2
            for attempt in range(max_retries):
                try:
                    response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=120)
                    if response.status_code == 429:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    if response.status_code in (400, 402, 404) and not use_single_model:
                        # Model not available / no credits – try next free model (only in free tier mode)
                        last_error = f"[{stage}] OpenRouter model {model} error {response.status_code}: {response.text[:300]}"
                        break
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        last_error = f"[{stage}] OpenRouter model {model} returned no choices."
                        break
                    message = choices[0].get("message", {})
                    content = message.get("content") or message.get("reasoning") or ""
                    cleaned = _clean_llm_output(content)
                    if cleaned:
                        return cleaned
                    last_error = f"[{stage}] OpenRouter model {model} returned an empty response."
                    break
                except requests.exceptions.Timeout as e:
                    last_error = f"[{stage}] OpenRouter timeout model {model}: {e}"
                    if attempt == max_retries - 1:
                        break
                    time.sleep(base_delay * (2 ** attempt))
                except Exception as exc:
                    last_error = f"[{stage}] OpenRouter error model {model}: {exc}"
                    if attempt == max_retries - 1:
                        break
                    time.sleep(base_delay * (2 ** attempt))
            # If using single model, don't try other models - return error
            if use_single_model:
                return last_error or f"[{stage}] OpenRouter failed after {max_retries} attempts"
            # If we are here and last_error is about model not available, try next model
            if last_error and ("error 400" in last_error or "error 404" in last_error or "402" in last_error or "returned no choices" in last_error):
                continue
            if last_error and "timeout" not in last_error.lower() and len(models) > 1:
                # Try next model for any error when multiple models available
                continue
            if last_error is None:
                continue

        return last_error or f"[{stage}] OpenRouter failed after {max_retries} attempts"
