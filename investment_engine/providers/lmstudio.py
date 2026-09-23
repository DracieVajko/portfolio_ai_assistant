from __future__ import annotations

import logging
import threading
import time

import requests

from .base import BaseLLMProvider
from investment_engine.providers.presets import merge_sampling, preset_for

logger = logging.getLogger(__name__)

# Global lock for model warm-up
_warmup_lock = threading.Lock()
_model_warmed = set()
_model_failed = set()  # Cache models that failed to load


class LMStudioProvider(BaseLLMProvider):
    """LM Studio provider using the configured local endpoint (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str = "http://100.101.20.64:1234/v1",
        model: str = "oda-fin-rl-8b",
        max_context_tokens: int = 32000,
        max_output_tokens: int = 8192,
        timeout_seconds: int = 7200,  # 2 hours for long model loading/generation
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._connect_timeout = 60  # seconds to establish connection
        self._read_timeout = 7200    # seconds to read response (model generation) - 2 hours

    @property
    def name(self) -> str:
        return "LM Studio"

    def _warmup_model(self, model: str) -> None:
        """Warm up the model with a simple request to ensure it's loaded."""
        key = (self.base_url, model)
        if key in _model_warmed:
            return
        if key in _model_failed:
            raise RuntimeError(f"Model '{model}' previously failed to load")
        with _warmup_lock:
            if key in _model_warmed:
                return
            if key in _model_failed:
                raise RuntimeError(f"Model '{model}' previously failed to load")
            logger.info("Warming up model %s on %s...", model, self.base_url)
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                    "temperature": 0.1,
                    "stream": False,
                }
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=(5, 10),  # Fail fast: 5s connect, 10s read
                )
                if r.status_code == 200:
                    _model_warmed.add(key)
                    logger.info("Model %s warmed up successfully", model)
                elif r.status_code == 400 and "Failed to load model" in r.text:
                    # Model cannot be loaded - fail fast so fallback can trigger
                    _model_failed.add(key)
                    error_msg = f"Model '{model}' failed to load: {r.text[:500]}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    # Transient (cold load, 5xx, ...): do NOT cache — the model
                    # may become ready by the next stage. Only the definitive
                    # 400 "Failed to load model" above is cached permanently.
                    logger.warning("Model warmup returned %s: %s", r.status_code, r.text[:200])
            except RuntimeError:
                raise
            except Exception as e:
                # Timeouts / connection errors while the model is (cold) loading
                # must not poison the model for the whole run — next stage
                # retries the warmup instead of falling back immediately.
                logger.warning("Model warmup failed (transient, will retry): %s", e)

    def _server_root(self) -> str:
        """Server root without the /v1 suffix (for management endpoints)."""
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root.rstrip("/")

    def unload_model(self, model: str) -> bool:
        """Best-effort unload of one model to free VRAM. Never raises.

        Tries known LM Studio management endpoint variants; servers without
        unload support just log a warning (use the app's auto-evict setting
        as the reliable path there). Also clears local warmup caches.
        """
        key = (self.base_url, model)
        _model_warmed.discard(key)
        _model_failed.discard(key)
        return _unload_via_api(self.base_url, model)

    def generate(self, prompt: str, *, stage: str, context: dict | None = None) -> str:
        requested_output_tokens = (context or {}).get("max_output_tokens", self.max_output_tokens)
        requested_model = str((context or {}).get("model", self.model))
        
        # Warm up the model on first use
        self._warmup_model(requested_model)
        
        preset = preset_for(stage)
        sampling = merge_sampling(preset, context)
        max_tokens = min(int(requested_output_tokens), self.max_output_tokens)
        floor = int(preset.get("min_tokens_floor", 0) or 0)
        if floor > max_tokens:
            # Reasoning trace needs budget or the answer comes back empty.
            max_tokens = min(floor, self.max_output_tokens)
        payload = {
            "model": requested_model,
            "messages": [
                {"role": "system", "content": f"You are the {stage} stage of the investment engine."},
                {"role": "user", "content": prompt},
            ],
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "top_k": sampling["top_k"],
            "repeat_penalty": sampling["repeat_penalty"],
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            logger.info("LM Studio request: stage=%s model=%s", stage, requested_model)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=(30, 300),  # 30s connect, 5min read
            )
            if response.status_code == 400:
                # Model is likely (cold) loading after a switch — the server
                # answers 400 instead of queueing. Wait, re-warmup, retry once.
                logger.warning(
                    "LM Studio 400 at stage=%s (likely model loading), "
                    "waiting 30s and retrying once: %s",
                    stage, (response.text or "")[:200])
                time.sleep(30)
                try:
                    self._warmup_model(requested_model)
                except Exception:
                    pass
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=(30, 300),
                )
            logger.info("LM Studio response: stage=%s status=%s", stage, response.status_code)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return f"[{stage}] LM Studio returned no choices."
            message = choices[0].get("message", {})
            # Some models (e.g., qwen3) put response in reasoning_content instead of content
            content = message.get("content") or message.get("reasoning_content") or ""
            content = str(content or "").strip()
            # Strip <answer> tags if present (some models wrap JSON in <answer> tags)
            if content.startswith("<answer>") and content.endswith("</answer>"):
                content = content[8:-9].strip()
            # Strip markdown code fences if present (some models wrap JSON in ```json ... ```)
            if content.startswith("```"):
                lines = content.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()
            return content.strip() or f"[{stage}] LM Studio returned an empty response."
        except requests.exceptions.HTTPError as exc:
            logger.exception("LM Studio generation failed at stage=%s", stage)
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("LM Studio generation failed at stage=%s", stage)
            raise


def _unload_via_api(base_url: str, model: str) -> bool:
    """Try known unload endpoint variants. Never raises. Returns True on success."""
    base = (base_url or "").rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    root = root.rstrip("/")
    candidates = [
        ("POST", f"{root}/api/v0/models/unload", {"model": model}),
        ("POST", f"{base}/models/unload", {"model": model}),
        ("DELETE", f"{base}/models/{model}", None),
    ]
    for method, url, payload in candidates:
        try:
            if method == "DELETE":
                r = requests.delete(url, timeout=(5, 15))
            else:
                r = requests.post(url, json=payload, timeout=(5, 15))
            body = (r.text or "")[:200]
            if r.status_code == 200 and "Unexpected endpoint" not in body:
                logger.info("Unloaded model %s via %s %s", model, method, url)
                return True
            logger.debug("Unload variant %s %s unsupported: %s %s",
                         method, url, r.status_code, body)
        except Exception as e:
            logger.debug("Unload variant %s %s failed: %s", method, url, e)
    logger.warning(
        "Model %s NOT unloaded: server has no unload API. "
        "Enable auto-evict in LM Studio app settings to free VRAM.", model)
    return False


def unload_lmstudio_models(base_url: str, models: list[str]) -> dict[str, bool]:
    """Best-effort unload of several models. Never raises. Returns {model: ok}."""
    out: dict[str, bool] = {}
    for model in dict.fromkeys(models):  # dedupe, keep order
        try:
            out[model] = _unload_via_api(base_url, model)
        except Exception:
            out[model] = False
    return out