from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name used for routing and tracing."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        """Generate a response for the supplied stage and prompt."""
        raise NotImplementedError
