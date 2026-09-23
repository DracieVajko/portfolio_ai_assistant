from __future__ import annotations

import logging
import re
import time
from typing import Any

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


# Per-run circuit breaker: unavailable providers are skipped for the rest of
# the run instead of retried at every stage (LM Studio/Ollama down, Gemini
# 429 rate-limit). No stack traces; one concise log line per trip.
_SKIP_REST_OF_RUN: set[str] = set()
_GEMINI_SKIP_UNTIL: float = 0.0


def _provider_model(provider: BaseLLMProvider) -> str:
    return str(getattr(provider, "model", "") or "")


def _is_rate_limited(result: str) -> bool:
    low = (result or "").lower()
    return "429" in low or "resource_exhausted" in low or "quota exceeded" in low


def _retry_after_seconds(result: str) -> float:
    m = re.search(r"retry in ([\d.]+)s", result or "")
    try:
        return float(m.group(1)) if m else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def reset_circuit_breaker() -> None:
    """Clear per-run skip state (called at engine start)."""
    _SKIP_REST_OF_RUN.clear()
    global _GEMINI_SKIP_UNTIL
    _GEMINI_SKIP_UNTIL = 0.0


def _is_error_result(result: str, stage: str) -> bool:
    """Detect provider error strings that should trigger fallback."""
    if not isinstance(result, str):
        return True
    r = result.strip()
    if not r:
        return True
    # Common error markers returned by providers (not exceptions)
    markers = [
        f"[{stage}]",
        "not configured",
        "returned no choices",
        "returned an empty response",
        "returned no content",
        "returned no candidates",
        "timeout after",
        "rate limited",
    ]
    # If starts with [stage] it's an error string
    if r.startswith(f"[{stage}]"):
        return True
    # Fallback on generic markers
    low = r.lower()
    for m in markers:
        if m.lower() in low and r.startswith("["):
            return True
    return False


class FallbackProvider(BaseLLMProvider):
    """Wrap a primary provider with a fallback provider."""

    def __init__(self, primary: BaseLLMProvider, fallback: BaseLLMProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def name(self) -> str:
        return f"Fallback({self.primary.name}/{self.fallback.name})"

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        # Use each provider's own default model - don't pass model from context to primary
        primary_context = {k: v for k, v in (context or {}).items() if k != "model"}
        fallback_context = {k: v for k, v in (context or {}).items() if k != "model"}
        
        try:
            primary_result = self.primary.generate(prompt, stage=stage, context=primary_context)
            if _is_error_result(primary_result, stage):
                logger.warning("FallbackProvider: primary (%s) returned error: %s. Trying fallback (%s)",
                               self.primary.name, primary_result[:200], self.fallback.name)
                raise RuntimeError(primary_result)
            _m = _provider_model(self.primary)
            logger.info("FallbackProvider: provider %s%s succeeded at stage %s",
                        self.primary.name, f" model={_m}" if _m else "", stage)
            return primary_result
        except Exception as exc:
            # Primary provider raised an exception - fall back
            logger.warning("FallbackProvider: primary (%s) failed: %s. Trying fallback (%s)", 
                          self.primary.name, exc, self.fallback.name)
            try:
                result = self.fallback.generate(prompt, stage=stage, context=fallback_context)
                if _is_error_result(result, stage):
                    # If fallback chain handles it, let it bubble; otherwise raise
                    # Check if fallback is also a chain – if it returns error, we still want to raise to trigger outer fallback
                    # For single fallback, return error string as final fallback
                    # Detect if fallback is chain by checking name
                    if "Fallback" in self.fallback.name and _is_error_result(result, stage):
                        raise RuntimeError(result)
                return result
            except Exception as fallback_exc:
                logger.error("FallbackProvider: fallback (%s) also failed: %s", 
                            self.fallback.name, fallback_exc)
                raise


class ChainedFallbackProvider(BaseLLMProvider):
    """Chain of providers – tries each in order until one succeeds (no error string)."""

    def __init__(self, providers: list[BaseLLMProvider]) -> None:
        if not providers:
            raise ValueError("ChainedFallbackProvider requires at least one provider")
        self.providers = providers

    @property
    def name(self) -> str:
        return "Chain(" + "->".join(p.name for p in self.providers) + ")"

    def _eligible(self) -> list[BaseLLMProvider]:
        now = time.time()
        out = []
        for p in self.providers:
            if p.name in _SKIP_REST_OF_RUN:
                continue
            if p.name == "Gemini" and now < _GEMINI_SKIP_UNTIL:
                continue
            out.append(p)
        return out or list(self.providers)

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        clean_context = {k: v for k, v in (context or {}).items() if k != "model"} if context else {}
        last_exc: Exception | None = None
        last_result: str | None = None
        for provider in self._eligible():
            try:
                result = provider.generate(prompt, stage=stage, context=clean_context)
                if _is_error_result(result, stage):
                    logger.warning("ChainedFallback: provider %s returned error, trying next: %s", provider.name, result[:200])
                    last_result = result
                    last_exc = RuntimeError(result)
                    self._trip_breaker(provider, result)
                    continue
                model = _provider_model(provider)
                logger.info("ChainedFallback: provider %s%s succeeded at stage %s",
                            provider.name, f" model={model}" if model else "", stage)
                return result
            except Exception as exc:
                logger.warning("ChainedFallback: provider %s failed: %s", provider.name, exc)
                last_exc = exc
                last_result = str(exc)
                self._trip_breaker(provider, str(exc))
                continue
        # All failed
        if last_exc:
            raise last_exc
        raise RuntimeError(last_result or "All providers in chain failed")

    @staticmethod
    def _trip_breaker(provider: BaseLLMProvider, result: str) -> None:
        """Skip dead providers for the rest of the run (concise, no traceback)."""
        global _GEMINI_SKIP_UNTIL
        name = provider.name
        if name in ("LM Studio", "Ollama"):
            if name not in _SKIP_REST_OF_RUN:
                _SKIP_REST_OF_RUN.add(name)
                logger.info("CircuitBreaker: skipping %s for the rest of the run", name)
        elif name == "Gemini" and _is_rate_limited(result):
            wait = _retry_after_seconds(result)
            _GEMINI_SKIP_UNTIL = time.time() + wait
            logger.info("CircuitBreaker: skipping Gemini for %s",
                        f"{wait:.0f}s" if wait != float("inf") else "the rest of the run")
