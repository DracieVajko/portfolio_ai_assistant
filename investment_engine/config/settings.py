from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env
    load_dotenv("api.env")  # Also load api.env for Trading212 credentials
except ImportError:
    pass


def _load_strategy_config() -> dict:
    """Load strategy configuration from JSON file."""
    config_path = Path(__file__).parent / "strategy.json"
    try:
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"strategy": "conservative", "profiles": {}, "overrides": {}}


_STRATEGY_CONFIG = _load_strategy_config()
_ACTIVE_STRATEGY = _STRATEGY_CONFIG.get("strategy", "conservative")
_ACTIVE_PROFILE = _STRATEGY_CONFIG.get("profiles", {}).get(_ACTIVE_STRATEGY, {})
_OVERRIDES = _STRATEGY_CONFIG.get("overrides", {})


def _env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").lower()
    return val in ("1", "true", "yes", "on") if val else default


@dataclass(slots=True)
class MarketRegimeSettings:
    """Configuration for EXI2 market regime detection."""

    enabled: bool = True
    symbol: str = "EXI2.DE"
    cache_ttl_seconds: int = 3600

    # Timeframe configurations
    timeframes: dict = field(default_factory=lambda: {
        "intraday_15m": {"interval": "15m", "period": "1d", "lookback_bars": 96, "enabled": True, "weight": 0.5},
        "intraday_1h": {"interval": "1h", "period": "5d", "lookback_bars": 120, "enabled": True, "weight": 0.7},
        "daily": {"interval": "1d", "period": "3mo", "lookback_bars": 63, "enabled": True, "weight": 1.5},
        "weekly": {"interval": "1wk", "period": "2y", "lookback_bars": 104, "enabled": True, "weight": 2.0},
        "monthly": {"interval": "1mo", "period": "max", "lookback_bars": 120, "enabled": True, "weight": 1.0},
    })

    # Peak/Valley detection
    peak_valley: dict = field(default_factory=lambda: {
        "prominence_pct": 2.0,
        "min_distance": 5,
        "cluster_threshold_pct": 1.0,
        "min_prominence_bars": 3,
    })

    # Regime classification thresholds
    thresholds: dict = field(default_factory=lambda: {
        # PEAK_HOLD
        "peak_hold_rsi_daily_min": 60,
        "peak_hold_rsi_weekly_min": 55,
        "peak_hold_price_above_sma20_daily": True,
        "peak_hold_price_above_sma50_weekly": True,
        "peak_hold_macd_hist_declining": True,
        "peak_hold_consolidation_days_min": 5,
        "peak_hold_consolidation_range_pct": 2.5,

        # DECLINING
        "declining_price_below_sma20_daily": True,
        "declining_price_below_sma50_daily": True,
        "declining_macd_bearish_cross": True,
        "declining_rsi_daily_max": 50,
        "declining_adx_min": 25,
        "declining_lower_highs": True,
        "declining_lower_lows": True,

        # MUST_BUY
        "must_buy_near_200dma_tolerance_pct": 2.0,
        "must_buy_near_52w_low_tolerance_pct": 5.0,
        "must_buy_rsi_daily_max": 35,
        "must_buy_rsi_weekly_max": 40,
        "must_buy_volume_spike_threshold": 2.0,
        "must_buy_supertrend_flip_bullish": True,
        "must_buy_macd_weekly_bullish": True,

        # Confidence
        "min_timeframes_agree": 2,
        "confidence_threshold": 0.75,
    })

    # News settings
    news: dict = field(default_factory=lambda: {
        "max_age_hours": 48,
        "min_relevance_score": 60,
        "max_articles_per_symbol": 5,
    })

    # FinViz enrichment
    finviz: dict = field(default_factory=lambda: {
        "enabled": True,
        "get_fundamentals": True,
        "get_sector_breadth": True,
        "sectors": [
            "Technology", "Healthcare", "Financial", "Consumer Cyclical",
            "Industrial", "Energy", "Utilities", "Real Estate",
            "Basic Materials", "Consumer Defensive", "Communication Services",
        ],
    })

    # Reporting
    reporting: dict = field(default_factory=lambda: {
        "include_in_main_report": True,
        "include_charts": False,
        "include_json": True,
    })


def _profile(key: str, default: Any) -> Any:
    """Get value from active strategy profile, with env override, then profile default, then hard default."""
    env_key = f"STRATEGY_{key.upper()}"
    if env_val := os.getenv(env_key):
        # Try to parse as int/float/bool
        if isinstance(default, bool):
            return env_val.lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            try:
                return int(env_val)
            except ValueError:
                pass
        if isinstance(default, float):
            try:
                return float(env_val)
            except ValueError:
                pass
        return env_val
    return _ACTIVE_PROFILE.get(key, default)


# Default llama.cpp endpoints (tailscale + docker internal)
_DEFAULT_LLAMACPP_URLS = [
    "http://100.125.47.31:11435/v1",      # Tailscale IP
    "http://host.docker.internal:11435/v1",  # Docker internal
]


@dataclass(slots=True)
class EngineSettings:
    """Configuration for the new modular investment engine."""

    reasoning_level: str = "structured"
    preferred_themes: list[str] = field(default_factory=lambda: ["long_term", "quality", "risk_control"])
    preferred_sectors: list[str] = field(default_factory=lambda: ["AI", "Semiconductors", "Cloud", "Data Centers", "Renewables"])
    portfolio_strategy: str = "long_term"
    investment_profile: str = _ACTIVE_STRATEGY  # conservative/balanced/aggressive/very_aggressive
    max_news_age_days: int = _profile("max_news_age_hours", 7 * 24) // 24
    ignore_duplicate_news: bool = True
    strict_fact_mode: bool = True
    confidence_threshold: float = _profile("confidence_threshold", 0.6)
    research_depth: str = "moderate"
    provider: str = _env("PROVIDER", "auto")
    
    # LM Studio (local GPU)
    lm_studio_base_url: str = _env("LM_STUDIO_BASE_URL", "http://100.101.20.64:1234/v1")
    lm_studio_model: str = _env("LM_STUDIO_MODEL", "qwen3.8-4b")
    lm_studio_context_tokens: int = _env_int("LM_STUDIO_CONTEXT_TOKENS", 32768)
    lm_studio_max_output_tokens: int = _env_int("LM_STUDIO_MAX_OUTPUT_TOKENS", 32768)
    
    request_timeout_seconds: int = _env_int("REQUEST_TIMEOUT_SECONDS", 1800)
    decision_model: str = _env("DECISION_MODEL", "qwen3.8-4b")
    writer_model: str = _env("WRITER_MODEL", "qwen3.8-4b")
    fallback_model: str = _env("FALLBACK_MODEL", "openrouter/free")
    
    # OpenRouter
    openrouter_base_url: str = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = _env("OPENROUTER_MODEL", "openrouter/free")
    openrouter_api_key: str = _env("OPENROUTER_API_KEY", "")
    openrouter_api_key_alt: str = _env("OPENROUTER_API_KEY_ALT", "")
    openrouter_grok_model: str = _env("OPENROUTER_GROK_MODEL", "x-ai/grok-3-mini")
    openrouter_context_tokens: int = _env_int("OPENROUTER_CONTEXT_TOKENS", 62000)
    openrouter_max_output_tokens: int = _env_int("OPENROUTER_MAX_OUTPUT_TOKENS", 8192)
    
    # Gemini (Google) – free tier available
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_base_url: str = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    
    # Mistral – free tier / trial
    mistral_api_key: str = _env("MISTRAL_API_KEY", "")
    mistral_model: str = _env("MISTRAL_MODEL", "open-mistral-nemo")
    mistral_base_url: str = _env("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    
    # OpenCode Zen
    opencode_zen_api_key: str = _env("OPENCODE_ZEN_API_KEY", "")
    opencode_zen_model: str = _env("OPENCODE_ZEN_MODEL", "opencode-zen-free")
    opencode_zen_base_url: str = _env("OPENCODE_ZEN_BASE_URL", "https://api.opencode.ai/v1")
    
    # llama.cpp (OpenAI-compatible API on Raspberry Pi 5, port 11435)
    # Tries tailscale IP first, then docker internal
    llamacpp_base_url: str = _env("LLAMACPP_BASE_URL", _DEFAULT_LLAMACPP_URLS[0])
    llamacpp_model: str = _env("LLAMACPP_MODEL", "qwen3.8-9b")
    llamacpp_context_tokens: int = _env_int("LLAMACPP_CONTEXT_TOKENS", 32768)
    
    # Ollama (separate from llama.cpp)
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://100.125.47.31:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "qwen3.8-9b-pi")
    ollama_context_tokens: int = _env_int("OLLAMA_CONTEXT_TOKENS", 128000)
    
    output_folder: str = _env("OUTPUT_FOLDER", "reports")
    archive_folder: str = _env("ARCHIVE_FOLDER", "reports/archive")
    prompt_dir: str = _env("PROMPT_DIR", "investment_engine/prompts")

    # AI aggressiveness profile from strategy
    ai_aggressiveness: str = _profile("ai_aggressiveness", "balanced")
    
    # Watchlist tickers to always monitor (even if not held)
    watchlist: list[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "GOOG",
        "NEE", "VWS.CO", "IBE.MC", "RWE.DE",
        "ASML.AS", "SAP.DE", "ASML", "TSM"
    ])
    
    # Playwright fallback for technical data when yfinance fails
    use_playwright_fallback: bool = _env_bool("USE_PLAYWRIGHT_FALLBACK", True)
    playwright_headless: bool = _env_bool("PLAYWRIGHT_HEADLESS", True)
    playwright_timeout_seconds: int = _env_int("PLAYWRIGHT_TIMEOUT_SECONDS", 30)
    
    # Market regime settings
    market_regime: MarketRegimeSettings = field(default_factory=MarketRegimeSettings)

    @classmethod
    def from_defaults(cls) -> "EngineSettings":
        """Create settings with strategy profile applied to market_regime."""
        # Create base instance
        instance = cls()
        # Apply strategy profile to market_regime.news (override defaults)
        instance.market_regime.news["max_age_hours"] = _profile("max_news_age_hours", 48)
        instance.market_regime.news["min_relevance_score"] = _profile("min_relevance_score", 30)
        instance.market_regime.news["max_articles_per_symbol"] = _profile("max_articles_per_symbol", 5)
        return instance

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "EngineSettings":
        """Create settings from the project's JSON config, ignoring unrelated keys."""
        values = values or {}
        aliases = {
            "model": "lm_studio_model",
            "num_ctx": "lm_studio_context_tokens",
            "lm_studio_timeout": "request_timeout_seconds",
        }
        allowed = set(cls.__dataclass_fields__)
        normalized = {
            aliases.get(key, key): value
            for key, value in values.items()
            if aliases.get(key, key) in allowed
        }
        # Handle nested market_regime settings
        if "market_regime" in values and isinstance(values["market_regime"], dict):
            normalized["market_regime"] = MarketRegimeSettings(**values["market_regime"])
        # Ensure secrets are loaded from environment if not in config
        for secret_field in (
            "openrouter_api_key",
            "openrouter_api_key_alt",
            "gemini_api_key",
            "mistral_api_key",
            "opencode_zen_api_key",
            "lm_studio_base_url",
            "openrouter_base_url",
            "ollama_base_url",
            "gemini_base_url",
            "mistral_base_url",
            "opencode_zen_base_url",
            "llamacpp_base_url",
        ):
            if secret_field not in normalized:
                normalized[secret_field] = _env(secret_field.upper(), "")
        
        # Apply strategy profile to market_regime news settings
        if "market_regime" not in normalized:
            normalized["market_regime"] = {}
        if isinstance(normalized["market_regime"], dict):
            mr = normalized["market_regime"]
            if "news" not in mr:
                mr["news"] = {}
            mr["news"].setdefault("max_age_hours", _profile("max_news_age_hours", 48))
            mr["news"].setdefault("min_relevance_score", _profile("min_relevance_score", 50))
            mr["news"].setdefault("max_articles_per_symbol", _profile("max_articles_per_symbol", 5))
        
        return cls(**normalized)

    def prompt_dir_path(self) -> Path:
        return Path(self.prompt_dir)