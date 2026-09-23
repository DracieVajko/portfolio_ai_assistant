from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from investment_engine.config.settings import EngineSettings
from investment_engine.providers.factory import ProviderFactory
from investment_engine.research.market_data import analyst_consensus, fetch_recent_headlines, recent_earnings_date
from investment_engine.research.market_regime import EXI2RegimeAnalyzer
from investment_engine.research.news_engine import StrictNewsFetcher, NewsItem, analyze_news_sentiment
from investment_engine.research.pies import PieLoader
from investment_engine.reporting.regime_report import (
    RegimeReportGenerator,
    AIContextReportBuilder,
)
from investment_engine.portfolio.broker_first import (
    ReconciliationResult,
    AccountSnapshot,
)
from investment_engine.schemas.ai_recommendations import parse_ai_recommendations, sanitize_ai_output


# Compact record of LLM fallback events for the current run. Rendered as a
# one-line provider note in the report (no stack traces, no secrets).
_FALLBACK_EVENTS: list[str] = []


def _record_fallback(stage: str, detail: str) -> None:
    label = f"{stage}: {detail}".strip()
    if label not in _FALLBACK_EVENTS:
        _FALLBACK_EVENTS.append(label)


def _note_unreachable_local_providers(settings) -> None:
    """Probe local LLM endpoints once; record a compact fallback event.

    Non-fatal by design: unreachable LM Studio/Ollama is reported in the
    report's provider note, never as a stack trace.
    """
    try:
        import requests as _rq
    except ImportError:
        return
    for label, url in (
        ("LM Studio", f"{str(getattr(settings, 'lm_studio_base_url', '')).rstrip('/')}/models"),
        ("Ollama", f"{str(getattr(settings, 'ollama_base_url', '')).rstrip('/')}/api/tags"),
    ):
        if not url or "://" not in url:
            continue
        try:
            _r = _rq.get(url, timeout=2)
            if _r.status_code != 200:
                _record_fallback("provider-chain", f"{label} unreachable")
        except Exception:
            _record_fallback("provider-chain", f"{label} unreachable")


def _provider_fallback_note() -> str:
    if not _FALLBACK_EVENTS:
        return ""
    stages = sorted({e.split(":")[0] for e in _FALLBACK_EVENTS})
    return (
        f"> ℹ️ **Provider note:** local LLM endpoints unavailable; "
        f"deterministic fallback used for: {', '.join(stages)}."
    )


def _symbol(asset: dict[str, Any]) -> str:
    return str(asset.get("broker_symbol") or asset.get("yahoo_symbol") or asset.get("name") or "UNKNOWN")


def _is_us_symbol(symbol: str) -> bool:
    """Check if symbol is likely US-listed (for FinViz compatibility)."""
    # EXI2.DE, .L, .F, .CO, .KS, .DE, .AS, etc. are non-US
    non_us_suffixes = ('.DE', '.L', '.F', '.CO', '.KS', '.AS', '.SW', '.MI', '.PA', '.BR', '.OL', '.HE', '.VI', '.ST', '.CO', '.IC')
    return not any(symbol.endswith(suf) for suf in non_us_suffixes)


def _fetch_t212_data(settings_dict: dict) -> dict[str, Any] | None:
    """Fetch Trading212 portfolio data if enabled, with graceful degradation."""
    if not settings_dict.get("use_trading212_api", True):
        return None
    try:
        from trading212_integration import create_integration
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        
        logger.info("Fetching T212 account summary...")
        t212 = create_integration(config_file="api.env")
        if not t212:
            logger.warning("T212 not configured")
            return {"status": "failed", "error": "T212 not configured"}
        
        logger.debug("T212 integration created, calling get_account_summary...")
        # Single call to get_account_summary gets everything (cash + positions)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(t212.get_account_summary)
            try:
                summary = future.result(timeout=20)
            except FuturesTimeoutError:
                logger.error("T212 summary timeout after 20s")
                return {"status": "failed", "error": "T212 summary timeout"}
        
        logger.info(f"T212 summary received: status={summary.get('status')}, keys={list(summary.keys()) if isinstance(summary, dict) else 'N/A'}")
        
        if summary.get("status") == "failed":
            logger.warning(f"T212 summary failed: {summary.get('error')}")
            return {"status": "failed", "error": summary.get("error")}
        
        logger.debug(f"T212 summary: total_equity={summary.get('total_equity')}, cash_free={summary.get('cash_free')}, positions={len(summary.get('positions', []))}, all_positions={len(summary.get('all_positions', []))}")

        # Extract everything from single response. `all_positions` is the
        # authoritative broker snapshot; `positions` is the non-pie subset view.
        return {
            "status": "ok",
            "account_summary": summary,
            "positions": summary.get("positions", []),
            "all_positions": summary.get("all_positions", []),
            "cash": {
                "free": summary.get("cash_free", 0),
                "invested": summary.get("invested", 0),
                "pie_cash": summary.get("cash_pie", 0),
                "total": summary.get("total_equity", 0),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"T212 fetch failed: {e}")
        return {"status": "failed", "error": str(e)}


def _fetch_all_news_parallel(settings_dict: dict, assets: List[Dict[str, Any]], max_workers: int = 8) -> tuple[dict, list[NewsItem]]:
    """Fetch news for assets in parallel - all enabled assets, not just top 10."""
    mr_news = settings_dict.get("market_regime", {}).get("news", {})
    fetcher = StrictNewsFetcher(
        max_age_hours=mr_news.get("max_age_hours", 48),
        min_relevance=mr_news.get("min_relevance_score", 30),  # Lower default for aggressive
    )

    # Fetch for ALL enabled assets (not just top 10)
    priority_assets = [a for a in assets if a.get("enabled", True)]

    def fetch_one(asset):
        sym = _symbol(asset)
        name = asset.get("name", sym)
        items = fetcher.fetch_for_symbol(
            sym, name,
            limit=mr_news.get("max_articles_per_symbol", 5)
        )
        return sym, [item.__dict__ for item in items], items

    news_by_symbol: dict[str, list[dict]] = {}
    all_news_items: list[NewsItem] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, asset): asset for asset in priority_assets}
        for future in as_completed(futures):
            try:
                sym, items_dict, items_obj = future.result(timeout=15)
                if items_dict:
                    news_by_symbol[sym] = items_dict
                    all_news_items.extend(items_obj)
            except Exception as e:
                asset = futures[future]
                print(f"News fetch failed for {asset.get('name', 'unknown')}: {e}")

    # Fetch macro news (SPY, BTC) + Trump policy watch in parallel
    # Always include macro per strategy overrides
    macro_queries = {
        "SPY": ("S&P 500", 3, None),
        "BTC": ("Bitcoin", 2, None),
        "TRUMP": ("Trump", 5, ["Trump tariffs trade policy", "Trump economy markets", "Trump regulation stocks"]),
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        macro_futures = {
            executor.submit(fetcher.fetch_for_symbol, sym, name, limit, extra): label
            for sym, (name, limit, extra) in macro_queries.items()
            for label in [sym]
        }
        for future in as_completed(macro_futures):
            try:
                label = macro_futures[future]
                items = future.result(timeout=15)
                if items:
                    news_by_symbol[label] = [item.__dict__ for item in items]
            except Exception:
                pass

    return news_by_symbol, []


def _yahoo_symbol(asset: dict[str, Any]) -> str:
    """Get the Yahoo Finance symbol for market data lookups."""
    return str(asset.get("yahoo_symbol") or asset.get("broker_symbol") or asset.get("name") or "UNKNOWN")


def _fetch_earnings_fast(assets: List[Dict[str, Any]], timeout_per_symbol: float = 3.0) -> dict[str, str]:
    """Fast earnings fetch with timeout per symbol, keyed by DISPLAY ticker.

    Yahoo lookups go through the explicit mapping layer only; company names
    and internal broker IDs never reach yfinance and never become keys.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    try:
        from investment_engine.portfolio.symbols import (
            is_earnings_supported,
            to_display_symbol as _td_e,
            to_yahoo_symbol as _ty_e,
        )
    except Exception:
        _td_e = lambda s, *_: str(s or "").strip().upper().split("_")[0]
        _ty_e = lambda i, d, a: d
        is_earnings_supported = lambda y: bool(y)

    def fetch_one(asset):
        explicit_yahoo = (asset or {}).get("yahoo_symbol")
        ref = explicit_yahoo or (asset or {}).get("ticker") or _symbol(asset)
        try:
            disp = _td_e(ref)
        except Exception:
            disp = str(ref or "UNKNOWN").strip().upper()
        try:
            yahoo = explicit_yahoo or _ty_e(ref, disp, None)
            if not yahoo or not is_earnings_supported(yahoo):
                return disp, "unavailable"
            date = recent_earnings_date(yahoo)
            return disp, date if date else "no_data"
        except Exception as e:
            return disp, f"error: {str(e)[:50]}"

    def _disp_of(asset) -> str:
        ref = (asset or {}).get("yahoo_symbol") or (asset or {}).get("ticker") or _symbol(asset)
        try:
            return _td_e(ref)
        except Exception:
            return str(ref or "UNKNOWN").strip().upper()

    earnings: dict[str, str] = {}
    # Limit to 15 assets; keys are DISPLAY tickers.
    priority_assets = assets[:15]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, asset): asset for asset in priority_assets}
        for future in as_completed(futures):
            try:
                sym, date = future.result(timeout=timeout_per_symbol)
                earnings[sym] = date
            except FuturesTimeoutError:
                asset = futures[future]
                earnings[_disp_of(asset)] = "timeout"
            except Exception as e:
                asset = futures[future]
                earnings[_disp_of(asset)] = f"error: {str(e)[:50]}"

    return earnings


def _build_regime_context(regime_result) -> str:
    """Build concise regime context for AI prompts."""
    if not regime_result:
        return "Regime analysis unavailable"
    impl = regime_result.implications
    return (
        f"EXI2 Regime: {regime_result.regime} ({regime_result.confidence:.0%} confidence)\n"
        f"Current Price: €{regime_result.timeframes.get('daily', {}).get('indicators', {}).get('CLOSE', 0):.2f}\n"
        f"RSI Daily: {regime_result.timeframes.get('daily', {}).get('indicators', {}).get('RSI_14', 0):.1f} | "
        f"RSI Weekly: {regime_result.timeframes.get('weekly', {}).get('indicators', {}).get('RSI_14', 0):.1f}\n"
        f"Trend Daily: {regime_result.timeframes.get('daily', {}).get('trend', 'UNKNOWN')} | "
        f"Trend Weekly: {regime_result.timeframes.get('weekly', {}).get('trend', 'UNKNOWN')}\n"
        f"Portfolio Action: {impl.get('portfolio_action', 'UNKNOWN')}\n"
        f"Tech Allocation: {impl.get('tech_allocation', 'NORMAL_DCA')}\n"
        f"Crypto Allocation: {impl.get('crypto_allocation', 'NORMAL_DCA')}\n"
        f"DCA Multiplier: {impl.get('dca_multiplier', 1.0)}x\n"
        f"Cash Target: {impl.get('cash_target_pct', 10)}%\n"
        f"Guidance: {impl.get('message', '')}\n"
        f"Warnings: {'; '.join(regime_result.warnings) if regime_result.warnings else 'None'}"
    )


def _fetch_finviz_safe(symbol: str) -> dict:
    """Safe FinViz fetch — disabled gracefully when incompatible/unavailable.

    FinViz covers US listings only; anything else returns {} without a call.
    Import paths are probed defensively so an installed-library API change
    degrades to {} (logged at debug) instead of raising.
    """
    if not symbol or not _is_us_symbol(symbol):
        return {}
    try:
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(symbol)
        fund = stock.ticker_fundament()
        if fund is None or (hasattr(fund, "empty") and fund.empty):
            return {}
    except Exception as exc:
        logger.debug("FinViz quote unavailable for %s: %s", symbol, exc)
        return {}

    breadth = {}
    try:
        from finvizfinance.group.overview import Overview as FinvizGroupOverview
        go = FinvizGroupOverview()
        sectors = ["Technology", "Healthcare", "Financial", "Consumer Cyclical",
                   "Industrial", "Energy", "Utilities", "Real Estate",
                   "Basic Materials", "Consumer Defensive", "Communication Services"]
        for sector in sectors:
            try:
                go.set_filter(filters_dict={"Sector": sector})
                df = go.screener_view()
                if df is not None and not df.empty and "Change" in df.columns:
                    changes = df["Change"].astype(str).str.rstrip("%").astype(float)
                    breadth[sector] = {"count": len(df), "avg_change": float(changes.mean())}
            except Exception as exc:
                logger.debug("FinViz sector %s unavailable: %s", sector, exc)
    except Exception as exc:
        logger.debug("FinViz group overview unavailable: %s", exc)

    result: dict = {"fundamentals": fund}
    if breadth:
        result["sector_breadth"] = breadth
    return result


def _is_us_symbol(symbol: str) -> bool:
    """Check if symbol is likely US-listed (for FinViz compatibility)."""
    non_us_suffixes = ('.DE', '.L', '.F', '.CO', '.KS', '.AS', '.SW', '.MI', '.PA', '.BR', '.OL', '.HE', '.VI', '.ST', '.CO', '.IC')
    return not any(symbol.endswith(suf) for suf in non_us_suffixes)


def _symbol(asset: dict[str, Any]) -> str:
    return str(asset.get("broker_symbol") or asset.get("yahoo_symbol") or asset.get("name") or "UNKNOWN")


def run_engine(assets: List[Dict[str, Any]], portfolio_context: Dict[str, Any] | None = None, settings: EngineSettings | None = None) -> Dict[str, Any]:
    """Create a compact, live-data report with short model calls per active category."""
    settings = settings or EngineSettings.from_defaults()
    # Reduce third-party library debug noise (yfinance/peewee/urllib3); engine
    # messages stay at the configured level.
    for _noisy in ("yfinance", "peewee", "urllib3.connectionpool", "urllib3"):
        try:
            logging.getLogger(_noisy).setLevel(logging.WARNING)
        except Exception:
            pass
    _FALLBACK_EVENTS.clear()
    try:
        from investment_engine.providers.fallback import reset_circuit_breaker
        reset_circuit_breaker()
    except Exception:
        pass
    try:
        from investment_engine.research.market_data import reset_market_data_registry
        reset_market_data_registry()
    except Exception:
        pass
    language = str((portfolio_context or {}).get("language") or "English")
    _note_unreachable_local_providers(settings)
    by_symbol = {_symbol(asset): asset for asset in assets}
    pies = PieLoader().load_all()
    tech_pies = [pie for pie in pies if "tech" in pie["name"].lower()]
    renewable_pies = [pie for pie in pies if "renew" in pie["name"].lower()]
    passive_pies = [pie for pie in pies if pie not in tech_pies and pie not in renewable_pies]
    crypto_assets = [asset for asset in assets if str(asset.get("group", "")).upper() == "CRYPTO"]

    def entries_from_pies(group: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [{"ticker": holding["slice"], "name": holding["name"]} for pie in group for holding in pie["holdings"]]

    tech = entries_from_pies(tech_pies)
    renewable = entries_from_pies(renewable_pies)
    existing = {item["ticker"] for item in tech + renewable}
    other = [{"ticker": _symbol(asset), "name": str(asset.get("name") or _symbol(asset))} for asset in assets if str(asset.get("group", "")).upper() in {"TECH_PIE", "SHORT_TERM_TRADING"} and _symbol(asset) not in existing]
    crypto = [{"ticker": _symbol(asset), "name": str(asset.get("name") or _symbol(asset))} for asset in crypto_assets]

    all_active_assets = tech + renewable + other + crypto

    # Known clean-ticker universe + names (single mapping layer input).
    # Built from pie CSV slices and config assets; broker names merged later.
    try:
        from investment_engine.portfolio.symbols import build_known_maps as _build_known_maps_early
        _early_slices = [
            (h.get("slice"), h.get("name"))
            for pie in (tech_pies + renewable_pies + passive_pies)
            for h in (pie.get("holdings", []) if isinstance(pie, dict) else [])
        ]
        known_clean, known_names_base = _build_known_maps_early(_early_slices, assets)
    except Exception:
        known_clean, known_names_base = set(), {}
    
    # Include pie constituents for individual analysis (user wants detail on PIE holdings too)
    active_positions = all_active_assets  # Don't filter out pie constituents

    # --- FETCH DATA IN PARALLEL ---
    import time
    start_time = time.time()

    # 1. Trading212 (with timeout protection)
    settings_dict = asdict(settings) if hasattr(settings, '__dataclass_fields__') else settings.__dict__
    t212_data = _fetch_t212_data(settings_dict)

    # Diagnostic raw endpoint dump (no state change, no trading)
    if t212_data and t212_data.get("status") == "ok":
        try:
            from trading212_integration import create_integration
            from trading212_auth import dump_raw_t212_diagnostics
            # Use the run_id from the portfolio_ai_assistant.py to ensure consistency
            _run_id = portfolio_context.get("run_id") if isinstance(portfolio_context, dict) else str(uuid.uuid4())[:8]
            _diag_client = create_integration(config_file="api.env")
            if _diag_client and _diag_client.monitor and _diag_client.monitor.client:
                _diag_result = dump_raw_t212_diagnostics(_diag_client.monitor.client, run_id=_run_id)
                if _diag_result:
                    t212_data["raw_endpoint_diagnostic"] = _diag_result
        except Exception as _diag_e:
            logger.debug(f"Raw endpoint diagnostic skipped: {_diag_e}")

    # Also include T212 positions if available (auto-discovered). The
    # authoritative snapshot is `all_positions`; only standalone (non-pie)
    # entries extend the asset universe for news context.
    if t212_data and t212_data.get("status") == "ok":
        _summary = t212_data.get("account_summary", {}) or {}
        _auth_positions = _summary.get("all_positions") or t212_data.get("all_positions") or _summary.get("positions", [])
        for pos in _auth_positions:
            if not isinstance(pos, dict) or pos.get("is_pie_constituent"):
                continue
            sym = pos.get("symbol")
            if sym and not any(a.get("ticker") == sym for a in all_active_assets):
                entry: dict[str, Any] = {"ticker": sym}
                # Only keep real instrument names; never echo the broker ID.
                if pos.get("name") and str(pos["name"]).strip().upper() != str(sym).strip().upper():
                    entry["name"] = pos["name"]
                all_active_assets.append(entry)
        active_positions = all_active_assets  # Update active_positions

    # Reconciliation guard: validate T212 data before proceeding
    reconciliation_ok = True
    if t212_data and t212_data.get("status") == "ok":
        summary = t212_data.get("account_summary", {})
        recon_status = summary.get("reconciliation_status", "UNKNOWN")
        recon_diff = summary.get("reconciliation_difference", 0)
        recon_threshold = summary.get("reconciliation_threshold", 0)
        total_equity = summary.get("total_equity", 0)
        derived_holdings = summary.get("derived_holdings_plus_available_cash", 0)
        
        logger.info(f"T212 Reconciliation: {recon_status} | Equity: €{total_equity:,.2f} | Derived: €{derived_holdings:,.2f} | Diff: €{recon_diff:,.2f} (threshold: €{recon_threshold:,.2f})")
        
        if recon_status == "FAIL":
            reconciliation_ok = False
            logger.warning("T212 reconciliation FAILED - suppressing account-level P&L and derived return from report and LLM")
            # Mark t212_data as reconciled=false so downstream code knows
            t212_data["reconciliation_ok"] = False
        else:
            t212_data["reconciliation_ok"] = True
    else:
        t212_data = t212_data or {"status": "failed", "reconciliation_ok": False}

    # 2. News (parallel, top 10 holdings)
    news_by_symbol, all_news_items = _fetch_all_news_parallel(settings_dict, all_active_assets)

    # 3. Earnings (parallel, fast)
    earnings = _fetch_earnings_fast(all_active_assets)

    # 4. Market Regime
    regime_result = None
    regime_context = ""
    regime_markdown = ""
    if settings.market_regime.enabled:
        try:
            regime_analyzer = EXI2RegimeAnalyzer(
                symbol=settings.market_regime.symbol,
                cache_ttl_seconds=settings.market_regime.cache_ttl_seconds,
            )
            regime_result = regime_analyzer.analyze()
            regime_context = _build_regime_context(regime_result)
        except Exception as e:
            regime_context = f"Regime analysis failed: {e}"

    # 5. AI Recommendations
    ai_recs = {}
    t212_ok = t212_data and t212_data.get("status") == "ok"
    reconciliation_ok = t212_data and t212_data.get("reconciliation_ok", False)
    # Allow AI recommendations even with reconciliation failure, just suppress account-level P&L
    if regime_result and t212_ok:
        provider = ProviderFactory.create(settings)
        ai_recs = _build_ai_recommendations(regime_result, t212_data, all_active_assets, provider, settings_dict, language)

    # 6. Regime + context report rendering happens at the end (needs all AI
    # sections, technicals, news and earnings ready). Placeholders for now.
    regime_markdown = ""
    context_file: Path | None = None

    # 7. LLM Decisions for active positions
    provider = ProviderFactory.create(settings)
    decision_sections: list[str] = []  # curated, rendered in the human report
    verbose_sections: list[str] = []  # full per-ticker analysis → AI context file only
    # ONE canonical signal mapping, built once and shared by every section
    # (T212 Portfolio, Priority Actions, AI Recommendations, PIE evaluation,
    # KPI text). Keyed by DISPLAY symbol; internal IDs alias to the same entry.
    try:
        from investment_engine.reporting.regime_report import build_canonical_signals as _build_canon
        from investment_engine.reporting.regime_report import ensure_canonical_defaults as _ensure_defaults
        from investment_engine.portfolio.symbols import to_display_symbol as _td_canon
        canonical_map: dict = _build_canon(ai_recs, known_clean)
        # Pre-fill HOLD defaults for every authoritative holding so the PIE
        # prompt and every downstream section share one complete mapping.
        _auth = ((t212_data or {}).get("account_summary", {}) or {}).get("all_positions") or (t212_data or {}).get("all_positions") or []
        _ensure_defaults(canonical_map, [_td_canon(str((p or {}).get("symbol", "")), known_clean) for p in _auth if isinstance(p, dict)])
    except Exception:
        canonical_map = {}
    canonical = {k: (v.get("signal") if isinstance(v, dict) else v) for k, v in (canonical_map or {}).items()}
    # Deduplicate display keys for the prompt note.
    _seen_disp: dict = {}
    for _k, _v in (canonical_map or {}).items():
        if isinstance(_v, dict) and _v.get("display") and _v["display"] not in _seen_disp:
            _seen_disp[_v["display"]] = _v.get("signal")
    canonical_display_note = (
        ", ".join(f"{_d}={_s}" for _d, _s in sorted(_seen_disp.items())) or "none yet"
    )
    canonical_note = (
        f"\nCANONICAL DECISIONS (final, from structured recommendations – do NOT contradict, do NOT repeat rationale, one line each): {canonical}"
        if canonical else ""
    )

    # Prepare news for LLM
    news_for_llm = news_by_symbol

    for title, positions in (("Crypto — decision only", crypto), ("Active Positions — Tech", tech + other), ("Active Positions — Renewables", renewable)):
        if not positions:
            continue
        try:
            from investment_engine.portfolio.symbols import to_display_symbol as _td_scope
            position_symbols = {_td_scope(str(item.get("ticker", ""))) for item in positions}
        except Exception:
            position_symbols = {item["ticker"] for item in positions}
        scoped_news = {symbol: entries for symbol, entries in news_for_llm.items() if symbol in position_symbols or (title.startswith("Crypto") and symbol in ["CRYPTO-MACRO", "MACRO"])}
        scoped_earnings = {symbol: date for symbol, date in earnings.items() if symbol in position_symbols}
        
        # Fetch technical data for each position (resolved Yahoo tickers only;
        # company names and internal IDs never reach yfinance).
        tech_data = {}
        for pos in positions:
            sym = pos["ticker"]
            try:
                from investment_engine.portfolio.symbols import to_display_symbol as _td_v, to_yahoo_symbol as _ty_v
                from investment_engine.research.market_data import fetch_technical_indicators
                _disp_v = _td_v(sym)
                _yahoo_v = _ty_v(sym, _disp_v, None)
                if not _yahoo_v:
                    continue
                indicators = fetch_technical_indicators(_yahoo_v, period="3mo", interval="1d")
                if indicators:
                    tech_data[sym] = indicators
            except Exception:
                pass
        
        tech_summary = ""
        if tech_data:
            for sym, ind in tech_data.items():
                tech_summary += _tech_summary_line(sym, ind) + "\n"
        
        # Special handling for crypto - no earnings dates, simpler format
        if title.startswith("Crypto"):
            prompt = f"""Write only the '{title}' section of a concise portfolio report in {language}. This is research, not financial advice.
Return at most 300 tokens. Use one Markdown bullet for every supplied ticker, with exactly: TICKER — BUY/SELL/HOLD/WAIT/WATCH — horizon (swing/long-term) — concise catalyst or risk — key level (support/resistance).
Do not invent prices, results, dates, news, or analyst opinions. Use the supplied headlines only as current evidence.

MARKET REGIME CONTEXT:
{regime_context}

TECHNICAL DATA:
{tech_summary or 'None available'}

POSITIONS: {positions}
RECENT HEADLINES: {[f"{sym}: {e.get('title','')[:120]} ({e.get('source','')}, {e.get('published','')})" for sym, entries in scoped_news.items() for e in entries] or 'None'}"""
        else:
            prompt = f"""Write only the '{title}' section of a concise portfolio report in {language}. This is research, not financial advice.
Return at most 500 tokens. Use one Markdown bullet for every supplied ticker, with exactly: TICKER — BUY/SELL/HOLD/WAIT/WATCH — horizon (intraday/swing/long-term) — concise catalyst or risk — key level (support/resistance) — earnings date/status.
Do not invent prices, results, dates, news, or analyst opinions. Use the supplied headlines only as current evidence. Aggressive short or intraday ideas must be marked HIGH RISK.

MARKET REGIME CONTEXT:
{regime_context}

TECHNICAL DATA:
{tech_summary or 'None available'}

POSITIONS: {positions}
EARNINGS: {scoped_earnings}
RECENT HEADLINES: {[f"{sym}: {e.get('title','')[:120]} ({e.get('source','')}, {e.get('published','')})" for sym, entries in scoped_news.items() for e in entries] or 'None'}"""
        
        section = _generate_with_fallback(provider, prompt, settings, tokens=500, stage="decision")
        verbose_sections.append(f"## {title}\n{section}")

    # 7b. T212 standalone holdings (outside PIEs) – per-stock data like legacy build:
    # price/RSI/SMA/support/resistance + 48h news + P&L, with AI BUY/SELL/HOLD per ticker.
    # 7c. PIE deep-dive – aggregate per PIE + detailed sample of constituents + sector view.
    # Both go through _generate_with_fallback, i.e. the full chained provider
    # (local → OpenRouter free → Gemini → Mistral → OpenCodeZen).
    standalone: list[dict] = []
    pie_blocks: list[dict] = []
    extra_tech: dict[str, dict] = {}
    extra_news: dict[str, list[dict]] = {}
    extra_earn: dict[str, str] = {}
    # display -> yahoo for every enrichment symbol (single mapping layer).
    yahoo_by_display: dict[str, str | None] = {}
    # Known clean tickers + names from pie CSVs and config assets.
    known_clean: set = set()
    try:
        from investment_engine.portfolio.pie_metadata import _normalize_base_symbol
        from investment_engine.portfolio.symbols import (
            build_known_maps as _build_known_maps,
            is_earnings_supported as _earn_supported,
            to_display_symbol as _to_display,
            to_yahoo_symbol as _to_yahoo,
        )
        aliases = _load_symbol_aliases(portfolio_context)
        _summ = (t212_data or {}).get("account_summary", {}) or {}
        # Authoritative broker snapshot; `positions` is fallback only and is
        # never merged into `all_positions`.
        t212_positions_all = list(_summ.get("all_positions") or [])
        if not t212_positions_all:
            t212_positions_all = list((t212_data or {}).get("all_positions") or [])
        if not t212_positions_all:
            t212_positions_all = list((t212_data or {}).get("positions", []) or [])

        # Known universe for display stripping + name resolution.
        _pie_slices = [
            (h.get("slice"), h.get("name"))
            for pie in (tech_pies + renewable_pies + passive_pies)
            for h in (pie.get("holdings", []) if isinstance(pie, dict) else [])
        ]
        known_clean, _known_names = _build_known_maps(_pie_slices, assets)
        for _k, _v in (aliases or {}).items():
            known_clean.add(str(_k).strip().upper())

        standalone = []
        for pos in t212_positions_all:
            if not isinstance(pos, dict) or pos.get("is_pie_constituent"):
                continue
            qty = pos.get("quantity") or 0
            if qty <= 0:
                continue
            t_sym = str(pos.get("symbol", ""))
            disp = _to_display(t_sym, known_clean)
            yahoo = _to_yahoo(t_sym, disp, aliases)
            yahoo_by_display[disp] = yahoo
            standalone.append({
                "t212": t_sym,
                "display": disp,
                "yahoo": yahoo,
                "qty": qty,
                "value_eur": pos.get("value_eur", 0) or 0,
                "pnl_eur": pos.get("pnl_eur", 0) or 0,
                "pnl_pct": pos.get("pnl_pct", 0) or 0,
            })
        standalone = sorted(standalone, key=lambda p: p["value_eur"], reverse=True)[:20]

        # Complete display -> Yahoo map for every authoritative position
        # (no network here; verification happens before each request).
        for pos in t212_positions_all:
            if isinstance(pos, dict) and pos.get("symbol"):
                try:
                    _d = _to_display(str(pos["symbol"]), known_clean)
                    yahoo_by_display.setdefault(_d, _to_yahoo(str(pos["symbol"]), _d, aliases))
                except Exception:
                    pass

        # PIE aggregates: match T212 positions to pie slices by DISPLAY symbol
        # (single mapping layer, so C7A0d_EQ matches the C7A0 pie slice).
        pos_by_display: dict[str, dict] = {}
        for pos in t212_positions_all:
            if isinstance(pos, dict) and pos.get("symbol"):
                pos_by_display[_to_display(str(pos.get("symbol")), known_clean)] = pos
        pie_blocks = []
        for pie in (tech_pies + renewable_pies + passive_pies):
            holdings = pie.get("holdings", []) if isinstance(pie, dict) else []
            enriched = []
            for h in holdings:
                sl = str(h.get("slice", "")).strip().upper()
                if not sl:
                    continue
                match = pos_by_display.get(_to_display(sl, known_clean))
                enriched.append({
                    "slice": sl,
                    "name": h.get("name", sl),
                    "value_eur": (match or {}).get("value_eur", 0) or 0,
                    "pnl_pct": (match or {}).get("pnl_pct", 0) or 0,
                    "matched": match is not None,
                })
            matched = [e for e in enriched if e["matched"]]
            total_v = sum(e["value_eur"] for e in enriched)
            sample = sorted(matched, key=lambda e: e["value_eur"], reverse=True)[:4]
            if len(sample) < 4:
                seen = {e["slice"] for e in sample}
                for e in enriched:
                    if e["slice"] not in seen:
                        sample.append(e)
                    if len(sample) >= 4:
                        break
            pie_blocks.append({
                "name": pie.get("name", "PIE") if isinstance(pie, dict) else "PIE",
                "count": len(enriched),
                "matched_count": len(matched),
                "value_eur": total_v,
                "sample": sample,
            })

        # Technicals + news + earnings for standalone + sampled pie slices.
        # Everything is keyed by DISPLAY symbol; Yahoo tickers come only from
        # the explicit mapping layer (never company names).
        for b in pie_blocks:
            for s in b["sample"]:
                disp = _to_display(s["slice"], known_clean)
                s["display"] = disp
                if disp not in yahoo_by_display:
                    yahoo_by_display[disp] = _to_yahoo(s["slice"], disp, aliases)
        # Yahoo lookup pairs, skipping unsupported instruments cleanly.
        yahoo_pairs = [(disp, y) for disp, y in yahoo_by_display.items() if y]
        extra_tech = _fetch_technicals_parallel(yahoo_pairs)
        extra_news = _fetch_news_for_extra_tickers(
            settings_dict,
            [(p["display"], p["yahoo"] or p["display"], p["display"]) for p in standalone if p.get("yahoo")]
            + [(s["display"], yahoo_by_display.get(s["display"]) or s["display"], s["name"]) for b in pie_blocks for s in b["sample"] if yahoo_by_display.get(s["display"])],
        )
        extra_earn = _fetch_earnings_for_symbols(yahoo_by_display)
        # Technicals keyed by display symbol for row enrichment.
        unified_technicals_seed = dict(extra_tech or {})

        # NOTE (refactor): the per-position holdings live ONLY in the unified
        # "T212 Portfolio" table (regime_report._t212_portfolio). The legacy
        # standalone LLM section "T212 Holdings — individual (outside PIEs)"
        # was a duplicate verbose breakdown and is intentionally NOT rendered.
        # Standalone data collected above is reused to enrich that single table.
        # (No decision_sections entry for individual holdings by design.)

        if any(b["sample"] for b in pie_blocks):
            pie_parts = []
            for b in pie_blocks:
                det = []
                for s in b["sample"]:
                    disp = s.get("display") or _to_display(s["slice"], known_clean)
                    det.append(f"  - {disp} ({s['name']}) | value=€{s['value_eur']:,.2f} | P&L={s['pnl_pct']:+.1f}% | {_tech_summary_line(disp, extra_tech.get(disp))} | earnings={extra_earn.get(disp, 'unavailable')}")
                pie_parts.append(
                    f"PIE {b['name']}: {b['count']} holdings, {b['matched_count']} matched in broker, matched value=€{b['value_eur']:,.2f}\n"
                    + "\n".join(det)
                )
            pie_prompt = f"""Write only the 'PIEs — evaluation' section of a concise portfolio report in {language}. This is research, not financial advice.
For EACH pie below output:
1. Verdict line: PIE NAME — HOLD / ACCUMULATE / TRIM — one-line reason (aggregate P&L + sampled technicals + sector view).
2. Sector view (1-2 sentences): how is the sector of its holdings doing, based on the technicals and headlines supplied.
3. One bullet per sampled stock, using EXACTLY this format with the DISPLAY ticker and the CANONICAL signal shown (do NOT change the signal; keep the technical observation after the em dash):
TICKER — SIGNAL — one-line technical observation — support/resistance.
CANONICAL DISPLAY TICKERS AND SIGNALS (final — use verbatim): {canonical_display_note}.
Do not invent prices, results, dates, news, or analyst opinions. Sampled stocks are a detail sample; the verdict is for the whole PIE.{canonical_note}

MARKET REGIME CONTEXT:
{regime_context}

PIES:
{chr(10).join(pie_parts)}

RECENT HEADLINES (48h, sampled constituents):
{_headlines_block({k: v for k, v in extra_news.items() if any(k == (s.get('display') or s['slice']) for b in pie_blocks for s in b['sample'])})}"""
            pie_section = _generate_with_fallback(provider, pie_prompt, settings, tokens=1200, stage="decision")
            pie_section = _enforce_canonical_in_pie_section(pie_section, canonical_map, known_clean)
            decision_sections.append(f"## PIEs — evaluation\n{pie_section}")
    except Exception as exc:
        logger.warning("T212/PIE deep-dive sections failed: %s", exc)

    # Candidate section
    discovery_prompt = f"""Write a Markdown section called '## Potential New Ideas' in {language}.
Using only these recent headlines, name at most three ticker candidates as WATCH, never BUY. Explain in one short line why each needs further verification. If no ticker is clearly identifiable and supported, write 'None supported by current evidence.'"""
    candidate_section = _generate_with_fallback(provider, discovery_prompt, settings, tokens=400, stage="discovery")

    # Summary (LLM) + Priority Actions (deterministic, unified).
    # Priority Actions are NEVER taken verbatim from the LLM: they are built
    # from the final unified T212 Portfolio rows so every ticker exists in
    # the portfolio table and every signal matches the normalized signal.
    # The LLM only writes the short ## Summary bullets.
    allowed_for_priority = _portfolio_tickers_for_priority(t212_data, known_clean)
    summary_prompt = f"""Write ONLY the '## Summary' Markdown section in {language}, based strictly on the supplied decisions.
At most 5 concise bullets: portfolio state, EXI2 regime, top risks and opportunities.
Do NOT write a Priority Actions table (it is generated deterministically).
Only mention these owned tickers when naming actions: {', '.join(sorted(allowed_for_priority)) or 'none'}.
Respect the canonical decisions below – same ticker keeps the same action everywhere.
Do not invent price, news or analyst facts. This is research, not financial advice.
CANONICAL: {canonical if canonical else 'none yet'}.

DECISIONS:
{chr(10).join(decision_sections + verbose_sections + [candidate_section])}"""
    summary_llm = _generate_with_fallback(provider, summary_prompt, settings, tokens=600, stage="summary")
    summary_section = _strip_priority_actions_from_summary(summary_llm)

    # Unified enrichment maps for portfolio table / KPIs / radar.
    # Names: display symbol -> company name (pie slices + config assets +
    # broker names + overrides). known_clean feeds display stripping.
    portfolio_names = _build_portfolio_names(all_active_assets, pie_blocks, t212_data, known_names_base)
    try:
        from investment_engine.reporting.regime_report import build_unified_portfolio_rows as _build_rows
        portfolio_rows = _build_rows(
            t212_data, ai_recs, unified_technicals_seed, portfolio_names,
            known_clean, canonical_map, yahoo_by_display,
        )
    except Exception:
        portfolio_rows = []
    unified_technicals = dict(extra_tech or {})

    # Earnings radar keyed by DISPLAY symbol with inline certainty tags.
    # Config-asset earnings remapped to display; T212 enrichment earnings
    # mapped via the Yahoo layer; unsupported instruments stay 'unavailable'.
    import re as _re
    from investment_engine.research.market_data import classify_earnings_status

    def _add_radar(disp: str, d, status: str | None = None) -> None:
        if isinstance(d, str) and _re.match(r"^\d{4}-\d{2}-\d{2}", d):
            st = status or classify_earnings_status(str(disp), d[:10])
            prev = earnings_status.get(str(disp))
            # A confirmed date always wins over an estimated one.
            if prev is None or (prev[1] == "estimated" and st == "confirmed"):
                earnings_status[str(disp)] = (d[:10], st)

    earnings_status: dict[str, tuple[str, str]] = {}
    unavailable_earnings: list[str] = []
    try:
        from investment_engine.portfolio.symbols import to_display_symbol as _td
        for disp, d in (earnings or {}).items():
            _add_radar(_td(str(disp), known_clean), d)
        for p in standalone:
            disp = p.get("display") or _td(p.get("t212", ""), known_clean)
            val = extra_earn.get(disp, extra_earn.get(p.get("yahoo", ""), ""))
            if isinstance(val, str) and _re.match(r"^\d{4}-\d{2}-\d{2}", val):
                _add_radar(disp, val)
            elif val in ("unavailable", "no_data", ""):
                if disp not in unavailable_earnings:
                    unavailable_earnings.append(disp)
        for b in pie_blocks:
            for s in b.get("sample", []) or []:
                disp = s.get("display") or _td(s.get("slice", ""), known_clean)
                val = extra_earn.get(disp, extra_earn.get(s.get("slice", ""), ""))
                if isinstance(val, str) and _re.match(r"^\d{4}-\d{2}-\d{2}", val):
                    _add_radar(disp, val)
                elif val in ("unavailable", "no_data", ""):
                    if disp not in unavailable_earnings:
                        unavailable_earnings.append(disp)
    except Exception:
        pass
    earnings_radar = _format_earnings_radar(earnings_status, portfolio_names, unavailable_earnings)
    # Trade safety: reconcile-gated BUY demotion applied once to the shared
    # canonical mapping AND rows, before priority/KPI/PIE/AI rendering.
    try:
        from investment_engine.reporting.regime_report import (
            apply_trade_safety as _safety,
            compute_reconciliation as _recon_fn,
        )
        _recon = _recon_fn(t212_data, portfolio_rows)
        # Sanitized observability record: numerics + endpoint names only.
        logger.info(
            "Reconciliation diagnostic | endpoints=/equity/account/cash,/equity/portfolio"
            " | total=%.2f positions=%.2f implied=%.2f reported=%.2f delta=%+.2f threshold=%.2f status=%s",
            _recon.get("total_equity", 0), _recon.get("positions_value", 0),
            _recon.get("implied_cash", 0), _recon.get("reported_cash", 0),
            _recon.get("cash_delta", 0), _recon.get("threshold", 0), _recon.get("status"),
        )
        ai_recs = _apply_fail_guards(ai_recs, _recon["status"])
        _safety(canonical_map, portfolio_rows, unified_technicals, _recon["status"], earnings_status)
    except Exception as exc:
        logger.debug("Trade safety pass skipped: %s", exc)
    priority_table = _build_priority_actions_table(
        portfolio_rows, earnings_status, unified_technicals, canonical_map=canonical_map
    )
    # Broker-first reconciliation diagnostic (CSV + Markdown, best-effort).
    # Never blocks the report; never claims a loss.
    try:
        from investment_engine.portfolio.broker_first import (
            build_account_snapshot as _build_snap,
        )
        from investment_engine.portfolio.broker_first import (
            build_diagnostic_rows as _diag_rows,
        )
        from investment_engine.portfolio.broker_first import (
            reconcile_snapshot as _reconcile_snap,
        )
        from investment_engine.portfolio.broker_first import (
            write_diagnostic_csv as _write_dcsv,
        )
        from investment_engine.portfolio.broker_first import (
            write_diagnostic_markdown as _write_dmd,
        )
        try:
            from trading212_portfolio import get_live_fx_rates as _live_fx
            _diag_fx = (_live_fx().get("rates") or {})
        except Exception:
            _diag_fx = {}
        try:
            from investment_engine.portfolio.broker_first import get_cached_catalog as _get_cat
            _diag_catalog = _get_cat()
        except Exception:
            _diag_catalog = {}
        _dsum = (t212_data.get("account_summary", {}) or {}) if isinstance(t212_data, dict) else {}
        # Raw broker rows first (audit-grade); parsed rows are NOT re-parsed.
        _dpos = _dsum.get("raw_positions") or []
        if not _dpos:
            _dpos = _dsum.get("all_positions") or (t212_data.get("all_positions", []) if isinstance(t212_data, dict) else [])
        _snap = _build_snap(
            {"free": _dsum.get("cash_free", 0), "pieCash": _dsum.get("cash_pie", 0),
             "blocked": _dsum.get("cash_blocked", 0), "invested": _dsum.get("invested", 0),
             "total": _dsum.get("total_equity", 0), "ppl": _dsum.get("unrealized_pnl_eur", 0)},
            _dpos, None, known_clean=known_clean, live_rates=_diag_fx or None,
            catalog=_diag_catalog or None,
        )
        _diag_recon = _reconcile_snap(_snap)
        _drows, _dsummary = _diag_rows(_snap)
        _dsummary["reconciliation_status"] = _diag_recon.reconciliation_status
        _dsummary["reconciliation_delta_eur"] = _diag_recon.reconciliation_delta_eur
        _dsummary["expected_open_positions_value_eur"] = _diag_recon.expected_open_positions_value_eur
        _dsummary["sum_deduplicated_broker_position_values_eur"] = \
            _diag_recon.sum_deduplicated_broker_position_values_eur
        _dsummary["sum_raw_broker_position_values_eur"] = \
            _diag_recon.sum_raw_broker_position_values_eur
        _dout = Path("reports")
        _dout.mkdir(parents=True, exist_ok=True)
        _write_dcsv(_drows, _dout / "reconciliation_diagnostic.csv")
        _write_dmd(_drows, _dsummary, _dout / "reconciliation_diagnostic.md")
        logger.info("Reconciliation diagnostic written: %s rows=%d status=%s",
                    _dout, len(_drows), _dsummary.get("reconciliation_status"))
    except Exception as exc:
        logger.debug(f"Diagnostic report skipped: {type(exc).__name__}")
    if summary_section and "## Summary" not in summary_section:
        summary_section = "## Summary\n" + summary_section.strip()
    summary_section = (summary_section or "## Summary").rstrip() + "\n\n" + priority_table
    try:
        _recon_status = _recon.get("status") if isinstance(_recon, dict) else None
    except NameError:
        _recon_status = None
    if not _recon_status:
        try:
            from investment_engine.reporting.regime_report import compute_reconciliation as _recon_fn3
            _recon_status = _recon_fn3(t212_data, portfolio_rows).get("status")
        except Exception:
            _recon_status = "UNKNOWN"
    if str(_recon_status or "").upper() == "FAIL":
        summary_section, decision_sections = _apply_fail_watchlist_sweep(
            summary_section, decision_sections, portfolio_rows, earnings_status, unified_technicals)

    # 6. Regime report parts (dashboard order: portfolio → AI recs → EXI2).
    # The unified T212 Portfolio table receives the same rows enrichment
    # (technicals + names + canonical signals) as the deterministic actions.
    regime_parts: dict[str, str] = {}
    if settings.market_regime.enabled and regime_result:
        report_gen = RegimeReportGenerator(
            include_charts=settings.market_regime.reporting.get("include_charts", False),
        )
        regime_parts = report_gen.generate_parts(
            regime_result, t212_data, ai_recs, None, unified_technicals,
            portfolio_names, known_clean, canonical_map, yahoo_by_display,
        )

    # AI context file – full archive for machines: verbose per-ticker analysis,
    # all technicals and all headlines (human report stays curated).
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tech_dump = "\n".join(
            _tech_summary_line(sym, ind) for sym, ind in sorted(extra_tech.items())
        )
        all_headlines = [e for entries in list(news_by_symbol.values()) + list(extra_news.values()) for e in entries]
        try:
            from investment_engine.reporting.regime_report import unresolved_market_data as _unresolved_md
            _unresolved_syms = _unresolved_md(portfolio_rows)
        except Exception:
            _unresolved_syms = []
        _unresolved_note = (
            "\n\n## Market-data coverage\n"
            + ("Market-data mapping unavailable; broker valuation retained for: "
               + ", ".join(_unresolved_syms) + "." if _unresolved_syms
               else "All holdings have verified market-data mappings.")
        )
        # Build reconciliation result for AI context
        try:
            from investment_engine.reporting.regime_report import compute_reconciliation
            _recon_for_context = compute_reconciliation(t212_data, portfolio_rows)
            from investment_engine.portfolio.broker_first import ReconciliationResult
            # expected_open_positions_value_eur = total_equity - available_cash (not implied_cash!)
            _total_equity = _recon_for_context.get("total_equity", 0.0)
            _available_cash = _recon_for_context.get("available_cash", 0.0)
            _expected_positions = _total_equity - _available_cash
            reconciliation = ReconciliationResult(
                broker_total_equity_eur=_total_equity,
                reported_free_cash_eur=_recon_for_context.get("free_cash", 0.0),
                pie_cash_eur=_recon_for_context.get("pie_cash", 0.0),
                total_reported_cash_eur=_available_cash,
                pending_cash_adjustments_eur=0.0,
                expected_open_positions_value_eur=round(_expected_positions, 2),
                sum_raw_broker_position_values_eur=_recon_for_context.get("positions_value", 0.0),
                sum_deduplicated_broker_position_values_eur=_recon_for_context.get("positions_value", 0.0),
                sum_externally_computed_position_values_eur=0.0,
                sum_excluded_values_eur=0.0,
                reconciliation_delta_eur=_recon_for_context.get("cash_delta", 0.0),
                tolerance_eur=_recon_for_context.get("threshold", 0.0),
                reconciliation_status=_recon_for_context.get("status", "UNKNOWN"),
                data_quality="OK" if _recon_for_context.get("status") == "PASS" else "DATA_QUALITY_FAIL",
                notes=[],
            )
        except Exception:
            reconciliation = None

        # Build account snapshot for AI context
        try:
            from investment_engine.portfolio.broker_first import build_account_snapshot as _build_snap
            _dsum = (t212_data.get("account_summary", {}) or {}) if isinstance(t212_data, dict) else {}
            _dpos = _dsum.get("raw_positions") or _dsum.get("all_positions") or (t212_data.get("all_positions", []) if isinstance(t212_data, dict) else [])
            from investment_engine.portfolio.broker_first import get_cached_catalog as _get_cat
            _diag_catalog = _get_cat()
            from trading212_portfolio import get_live_fx_rates as _live_fx
            _diag_fx = (_live_fx().get("rates") or {})
            account_snapshot = _build_snap(
                {"free": _dsum.get("cash_free", 0), "pieCash": _dsum.get("cash_pie", 0),
                 "blocked": _dsum.get("cash_blocked", 0), "invested": _dsum.get("invested", 0),
                 "total": _dsum.get("total_equity", 0), "ppl": _dsum.get("unrealized_pnl_eur", 0)},
                _dpos, None, known_clean=known_clean, live_rates=_diag_fx or None,
                catalog=_diag_catalog or None,
            )
        except Exception:
            account_snapshot = None

        # Build FX info
        try:
            from trading212_portfolio import get_live_fx_rates as _live_fx2
            fx_info = _live_fx2()
        except Exception:
            fx_info = {}

        # Build catalog meta
        try:
            from investment_engine.portfolio.broker_first import get_cached_catalog as _get_cat2
            import json as _json
            from pathlib import Path as _Path
            _cat_path = _Path("data/cache/t212_instruments.json")
            if _cat_path.exists():
                _cat_data = _json.loads(_cat_path.read_text(encoding="utf-8"))
                catalog_meta = {
                    "fetched_at": _cat_data.get("fetched_at", ""),
                    "age_days": 0.0,
                    "count": _cat_data.get("count", 0),
                }
            else:
                catalog_meta = {}
        except Exception:
            catalog_meta = {}

        # Build external research (signals from AI recs that are not broker positions)
        external_research = {}
        watchlist = []
        if ai_recs:
            # Extract watchlist ideas from candidate section
            try:
                from investment_engine.reporting.documents import extract_validated_ideas as _extract_ideas
                from investment_engine.portfolio.symbols import support_state as _ssw
                from investment_engine.portfolio.symbols import to_yahoo_symbol as _tyw
                def _idea_supported(ticker: str) -> bool:
                    try:
                        return _ssw(_tyw(ticker, ticker, {}), "market_data") == "SUPPORTED"
                    except Exception:
                        return False
                _ideas = _extract_ideas(candidate_section, news_by_symbol, _idea_supported)
                for idea in _ideas:
                    watchlist.append({
                        "ticker": idea.get("ticker", ""),
                        "signal": "WATCH",
                        "horizon": "research",
                        "catalyst": idea.get("reason", ""),
                        "key_level": "",
                    })
            except Exception:
                pass

        import uuid as _uuid
        run_id = str(_uuid.uuid4())[:8]

        # Collect model info for attribution
        model_info = {
            "decision_model": settings.decision_model,
            "summary_model": settings.writer_model,
            "discovery_model": settings.decision_model,
            "decision_detail_model": settings.decision_model,
            "ai_aggressiveness": getattr(settings, "ai_aggressiveness", "balanced"),
        }

        context_file = _save_ai_context(
            regime_result=regime_result,
            t212_data=t212_data,
            portfolio_rows=portfolio_rows,
            technicals=unified_technicals,
            yahoo_map=yahoo_by_display,
            external_research=external_research,
            watchlist=watchlist,
            reconciliation=reconciliation,
            account_snapshot=account_snapshot,
            fx_info=fx_info,
            catalog_meta=catalog_meta,
            output_dir=output_dir,
            run_id=run_id,
            model_info=model_info,
        )
    except Exception as exc:
        logger.warning("AI context save failed: %s", exc)
        # Fallback minimal context
        import uuid as _uuid
        context_file = _save_ai_context(
            regime_result=regime_result,
            t212_data=t212_data,
            portfolio_rows=portfolio_rows,
            technicals=unified_technicals,
            yahoo_map=yahoo_by_display,
            external_research={},
            watchlist=[],
            reconciliation=None,
            account_snapshot=None,
            fx_info={},
            catalog_meta={},
            output_dir=output_dir,
            run_id=str(_uuid.uuid4())[:8],
            model_info=model_info,
        )

    # --- Two-document reporting: decision brief (concise) + snapshot (full) ---
    from investment_engine.reporting.documents import (
        build_monitoring_items as _mon_items,
    )
    from investment_engine.reporting.documents import (
        extract_validated_ideas as _extract_ideas,
    )
    from investment_engine.reporting.documents import (
        render_brief as _render_brief,
    )
    from investment_engine.reporting.documents import (
        render_snapshot as _render_snapshot,
    )
    from investment_engine.research.market_data import filter_earnings_window as _filter_7d
    from investment_engine.research.market_data import report_now_bta as _now_bta
    from investment_engine.research.news_engine import filter_decision_news as _filter_news

    report_dt = _now_bta()
    try:
        report_date = report_dt.date()
    except Exception:
        from datetime import date as _d
        report_date = _d.today()

    # Explicit PIE-level warnings: pie aggregate (matched sample) down <= -5%.
    pie_warnings: dict[str, str] = {}
    try:
        for b in pie_blocks or []:
            matched = [s for s in (b.get("sample", []) or []) if s.get("matched")]
            if not matched:
                continue
            avg = sum(float(s.get("pnl_pct", 0) or 0) for s in matched) / len(matched)
            if avg <= -5.0:
                for s in matched:
                    try:
                        from investment_engine.portfolio.symbols import to_display_symbol as _tdw
                        pie_warnings[_tdw(s.get("slice", ""), known_clean)] = (
                            f"PIE {b.get('name')} aggregate {avg:+.1f}%")
                    except Exception:
                        pass
    except Exception:
        pie_warnings = {}

    # Watchlist ideas first (validated ticker + evidence + URL + supported data).
    try:
        from investment_engine.portfolio.symbols import support_state as _ssw
        from investment_engine.portfolio.symbols import to_yahoo_symbol as _tyw
        def _idea_supported(ticker: str) -> bool:
            try:
                return _ssw(_tyw(ticker, ticker, aliases), "market_data") == "SUPPORTED"
            except Exception:
                return False
    except ImportError:
        def _idea_supported(ticker: str) -> bool:
            return True
    ideas = _extract_ideas(candidate_section, news_by_symbol, _idea_supported)

    row_displays = {str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
                    for r in (portfolio_rows or []) if isinstance(r, dict)}
    pie_sample_displays: set[str] = set()
    try:
        for b in pie_blocks or []:
            for s in (b.get("sample", []) or []):
                d = s.get("display")
                if d:
                    pie_sample_displays.add(str(d).strip().upper())
    except Exception:
        pass
    idea_tickers = {str(i.get("ticker", "") or "").strip().upper() for i in ideas}
    earnings_allowed = set(row_displays) | set(pie_sample_displays) | set(idea_tickers)
    earnings_events: list[dict] = []
    for disp, val in (earnings_status or {}).items():
        d = val[0][:10] if isinstance(val, (tuple, list)) and val and isinstance(val[0], str) else None
        if not d:
            continue
        key = str(disp).strip().upper()
        if key in earnings_allowed:
            earnings_events.append({"display": key, "company": portfolio_names.get(key, key),
                                    "date": d, "source_type": None})
    earnings_7d = _filter_7d(earnings_events, report_dt)
    earnings_7d_displays = {e["display"] for e in earnings_7d.get("in_window", [])}

    monitoring_items = _mon_items(
        portfolio_rows, earnings_status=earnings_status, technicals=unified_technicals,
        pie_warnings=pie_warnings, recon_status=(_recon.get("status") if isinstance(_recon, dict) else "UNKNOWN"),
        report_date=report_date, canonical_map=canonical_map,
    )
    monitoring_by_display = {it["display"]: it["presentation"] for it in monitoring_items}
    for r in portfolio_rows or []:
        if isinstance(r, dict):
            d = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
            monitoring_by_display.setdefault(d, "HOLD")

    try:
        news_threshold = int(((settings_dict.get("market_regime", {}) or {}).get("news", {}) or {}).get(
            "min_relevance_score", 30))
    except (TypeError, ValueError):
        news_threshold = 30
    decision_news = _filter_news(
        news_by_symbol, owned=row_displays, constituents=pie_sample_displays,
        watchlist=idea_tickers, earnings=earnings_7d_displays, names=portfolio_names,
        relevance_min=news_threshold, now=report_dt,
        include_macro_unlinked=True,
        min_tier_for_unlinked=1,
    )

    if str((_recon.get("status") if isinstance(_recon, dict) else "")).upper() == "FAIL":
        cash_line = (f"€{_recon.get('reported_cash', 0):,.2f} reported cash — "
                     f"deployment withheld until account reconciliation passes.")
    else:
        cash_line = _format_cash_deployment(regime_result, t212_data, ai_recs, portfolio_rows)

    # Account performance: broker equity vs verified net cash flows. Shown
    # even when position reconciliation FAILs; FAIL only blocks new
    # BUY/DCA/deployment instructions (enforced upstream).
    try:
        account_performance, cashflow_metadata = _resolve_account_performance(
            t212_data, portfolio_context)
    except Exception as exc:
        logger.debug("Account performance unavailable: %s", type(exc).__name__)
        account_performance, cashflow_metadata = {"performance_status": "Unavailable"}, {}

    brief_md = _render_brief(
        generated_at=report_dt.strftime("%Y-%m-%d %H:%M %Z"),
        regime_result=regime_result, recon=_recon if isinstance(_recon, dict) else {},
        rows=portfolio_rows, monitoring_items=monitoring_items, earnings_7d=earnings_7d,
        decision_news=decision_news, ideas=ideas, cash_line=cash_line,
        performance=account_performance,
    )
    snapshot_md = _render_snapshot(
        generated_at=report_dt.strftime("%Y-%m-%d %H:%M %Z"),
        recon=_recon if isinstance(_recon, dict) else {},
        cash=(t212_data.get("cash", {}) if isinstance(t212_data, dict) else {}),
        rows=portfolio_rows, monitoring_by_display=monitoring_by_display,
        earnings_status=earnings_status, t212_data=t212_data,
        performance=account_performance, ledger_metadata=cashflow_metadata,
    )

    elapsed = time.time() - start_time
    print(f"Report generated in {elapsed:.1f}s")

    # Sanitized reconciliation diagnostic object for the JSON output
    # (numerics + status only; no raw payloads, no secrets).
    try:
        _diag_src = _recon if isinstance(_recon, dict) else {}
    except NameError:
        _diag_src = {}
    reconciliation_diagnostic = {
        key: _diag_src.get(key)
        for key in ("total_equity", "positions_value", "implied_cash",
                    "reported_cash", "cash_delta", "threshold", "status")
    }

    # Failed-ticker ledger: UNRESOLVED mappings + market-data fetch misses.
    # Never breaks the run; empty list when everything resolved.
    try:
        from investment_engine.reporting.failed_tickers import (
            collect_failed_tickers as _collect_failed,
        )
        from investment_engine.research.market_data import (
            _FETCH_FAILED as _fetch_failed_set,
        )
        failed_tickers = _collect_failed(
            yahoo_by_display if "yahoo_by_display" in dir() else {},
            standalone=standalone if "standalone" in dir() else [],
            fetch_failed=set(_fetch_failed_set),
        )
    except Exception:
        failed_tickers = []

    return {
        "markdown": brief_md,
        "brief_markdown": brief_md,
        "snapshot_markdown": snapshot_md,
        "brief_metadata": {
            "generated_at": report_dt.isoformat(),
            "timezone": "Europe/Bratislava",
            "recon_status": reconciliation_diagnostic.get("status"),
            "stance": next((l for l in brief_md.splitlines() if l.startswith("- Portfolio stance:")), ""),
            "monitoring_items": len(monitoring_items),
            "earnings_7d": len(earnings_7d.get("in_window", [])),
            "news_items": len(decision_news),
            "watchlist": len(ideas),
        },
        "snapshot_metadata": {
            "generated_at": report_dt.isoformat(),
            "timezone": "Europe/Bratislava",
            "recon_status": reconciliation_diagnostic.get("status"),
            "positions": len(portfolio_rows or []),
        },
        "context_file": str(context_file),
        "tech_tickers": tech + other,
        "renewable_tickers": renewable,
        "crypto": crypto,
        "passive_pies": [{"name": pie["name"], "tickers": [holding["slice"] for holding in pie["holdings"]]} for pie in passive_pies],
        "recent_news": news_by_symbol,
        "trump_watch": {},
        "earnings_dates": earnings,
        "settings": _sanitized_settings(settings),
        "report_mode": "decision_brief_and_snapshot",
        "market_regime": regime_result.to_dict() if regime_result else None,
        "ai_recommendations": ai_recs,
        "t212_data": _sanitized_t212_data(t212_data),
        "reconciliation": reconciliation_diagnostic,
        "account_performance": account_performance,
        "cash_flow_ledger_metadata": cashflow_metadata,
        # Human-brief contract: keys consumed by generate_human_brief()
        # via portfolio_ai_assistant.py wrapper (all computed above).
        "portfolio_rows": portfolio_rows or [],
        "monitoring_items": monitoring_items or [],
        "regime_result": regime_result,
        "decision_news": decision_news or [],
        "earnings_7d": earnings_7d or {},
        "ideas": ideas or [],
        "portfolio_names": portfolio_names or {},
        "failed_tickers": failed_tickers,
    }


def _build_ai_recommendations(
    regime_result, 
    t212_data: dict, 
    assets: List[Dict[str, Any]],
    provider,
    settings_dict: dict,
    language: str
) -> dict:
    """Generate AI-powered trade recommendations based on regime, portfolio, and free cash."""
    
    if not regime_result or not t212_data or t212_data.get("status") != "ok":
        return {"error": "Missing regime or T212 data"}
    
    # Reconciliation guard: account-level derived totals are unreliable when
    # reconciliation fails – but per-position broker data (qty/price/P&L) and cash
    # are still real. Run in degraded mode with a warning note for the LLM
    # instead of skipping AI recommendations entirely.
    reconciliation_ok = t212_data.get("reconciliation_ok", False)
    recon_note = ""
    if not reconciliation_ok:
        logger.warning("T212 reconciliation FAILED - AI recommendations in degraded mode (position-level data only, no derived account returns)")
        recon_note = "\nWARNING: Account-level derived totals are UNRECONCILED – do NOT use account return figures. Base decisions ONLY on per-position data, regime and news below."
    
    summary = t212_data.get("account_summary", {})
    positions = t212_data.get("positions", [])
    cash = t212_data.get("cash", {})
    free_cash = cash.get("free", 0)
    total_equity = summary.get("total_equity", 0)
    
    def _canonical_position(pos: dict) -> dict:
        """Extract canonical normalized position data for LLM prompts."""
        return {
            "symbol": pos.get("symbol"),
            "qty": pos.get("quantity"),
            "value_eur": pos.get("value_eur"),
            "pnl_pct": pos.get("pnl_pct"),
            "average_price_eur": pos.get("average_price_eur"),
            "current_price_eur": pos.get("current_price_eur"),
            "quote_currency": pos.get("quote_currency"),
            "price_unit": pos.get("price_unit"),
            "unrealized_pnl_eur": pos.get("pnl_eur"),
        }


    pos_summary = []
    for pos in positions:
        canonical = _canonical_position(pos)
        # Validation: quantity > 0 but value_eur is 0 or missing
        if canonical["qty"] and canonical["qty"] > 0 and (canonical["value_eur"] is None or canonical["value_eur"] == 0):
            logger.warning(f"Position validation FAIL: {canonical['symbol']} qty={canonical['qty']} value_eur={canonical['value_eur']}")
            canonical["validation_status"] = "FAIL"
        else:
            canonical["validation_status"] = "PASS"
        pos_summary.append(canonical)
    
    regime_summary = {
        "regime": regime_result.regime,
        "confidence": regime_result.confidence,
        "action": regime_result.implications.get("portfolio_action"),
        "cash_target_pct": regime_result.implications.get("cash_target_pct"),
        "dca_multiplier": regime_result.implications.get("dca_multiplier"),
        "tech_allocation": regime_result.implications.get("tech_allocation"),
        "crypto_allocation": regime_result.implications.get("crypto_allocation"),
    }
    
    key_levels = {
        "support": regime_result.price_structure.nearest_support,
        "resistance": regime_result.price_structure.nearest_resistance,
    }
    
    support_str = f"{key_levels['support']:.2f}" if key_levels['support'] else 'N/A'
    resistance_str = f"{key_levels['resistance']:.2f}" if key_levels['resistance'] else 'N/A'
    
    prompt = f"""You are a portfolio manager. Output ONLY valid JSON with specific trade actions.

PORTFOLIO:{recon_note}
- Equity: {total_equity:,.0f} EUR | Cash: {free_cash:,.0f} EUR | Cash Target: {regime_summary['cash_target_pct']}%
- Regime: {regime_summary['regime']} ({regime_summary['confidence']:.0%}) | Action: {regime_summary['action']}
- DCA Multiplier: {regime_summary['dca_multiplier']}x
- Tech Allocation: {regime_summary['tech_allocation']} | Crypto: {regime_summary['crypto_allocation']}
- Support: {support_str} | Resistance: {resistance_str}

POSITIONS: {pos_summary}

REGIME GUIDANCE: {regime_summary['action']} - Tech: {regime_summary['tech_allocation']}, Crypto: {regime_summary['crypto_allocation']}

OUTPUT ONLY VALID JSON:
{{
  "portfolio_actions": [{{"action": "hold", "asset": "SYMBOL", "reason": "reason based on regime/indicators/news"}}],
  "trades": [{{"side": "BUY", "asset": "SYMBOL", "qty": 0.0, "price": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "risk_pct": 0.0, "reason": "specific reason: regime/indicators/news"}}],
  "position_updates": [{{"asset": "SYMBOL", "stop_loss": 0.0, "new_stop": 0.0, "take_profit": 0.0, "new_tp": 0.0, "reason": "reason"}}],
  "cash_deployment": {{"free_cash": 0.0, "deploy_amount": 0.0, "deploy_pct": 0.0, "reserve": 0.0, "targets": [{{"asset": "SYMBOL", "amount": 0.0, "price": 0.0, "max_risk": 0.0}}]}}
}}

RULES:
- Max 2% risk per trade, min 1:2 R:R, mandatory stops, 50/50 scale-out
- PEAK_HOLD = prioritize SELLs, reduce new BUYs
- MUST_BUY = prioritize BUYs, deploy cash aggressively
- DECLINING = prepare watchlist, raise cash, no new positions
- NEUTRAL = normal DCA
- Use free cash for deployment, respect cash target
- Base decisions on: regime signals, technical indicators (RSI, MACD, moving averages), news sentiment, earnings
- Be concise, specific, actionable. No hedging language.
- Output ONLY valid JSON, no markdown, no extra text."""

    try:
        _rec_ctx: dict = {"max_output_tokens": 1024}
        _rec_model = ((settings_dict or {}).get("decision_model")
                      if isinstance(settings_dict, dict)
                      else getattr(settings_dict, "decision_model", None))
        if _rec_model:
            _rec_ctx["model"] = _rec_model
        result = provider.generate(prompt, stage="ai_recommendations", context=_rec_ctx)
        # Clean up response - remove any markdown code fences or extra text
        cleaned = result.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        # Use the new schema validation
        from investment_engine.schemas.ai_recommendations import parse_ai_recommendations
        return parse_ai_recommendations(cleaned)
    except Exception as e:
        return {"error": f"AI recommendation failed: {str(e)}", "raw_response": result[:500] if 'result' in locals() else "no response"}


def _save_ai_context(
    regime_result: RegimeResult,
    t212_data: dict | None,
    portfolio_rows: list[dict] | None,
    technicals: dict | None,
    yahoo_map: dict | None,
    external_research: dict | None,
    watchlist: list[dict] | None,
    reconciliation: ReconciliationResult | None,
    account_snapshot: AccountSnapshot | None,
    fx_info: dict | None,
    catalog_meta: dict | None,
    output_dir: Path,
    run_id: str = "",
    model_info: dict | None = None,
) -> Path:
    """Save machine-safe AI context report using broker-first model."""
    builder = AIContextReportBuilder()
    context = builder.build_report(
        regime_result=regime_result,
        t212_data=t212_data,
        portfolio_rows=portfolio_rows,
        technicals=technicals,
        yahoo_map=yahoo_map,
        external_research=external_research,
        watchlist=watchlist,
        reconciliation=reconciliation,
        account_snapshot=account_snapshot,
        fx_info=fx_info,
        catalog_meta=catalog_meta,
        run_id=run_id,
        model_info=model_info,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    context_file = output_dir / f"ai_context_{timestamp}.md"
    context_file.write_text(context, encoding="utf-8")
    return context_file


from investment_engine.providers.presets import WRITER_STAGES as _WRITER_STAGES


def _model_for_stage(settings, stage: str) -> str:
    """Route stages to models: writer (fast) for summaries, reasoning otherwise."""
    if stage in _WRITER_STAGES:
        return getattr(settings, "writer_model", None) or settings.decision_model
    return settings.decision_model


def _generate_with_fallback(provider, prompt, settings, tokens=800, stage="decision") -> str:
    """Generate with fallback - let the provider handle fallback internally."""
    import time as _time
    start = _time.time()
    model = _model_for_stage(settings, stage)

    def _log_attempt(stage_name, model_name, success, duration, content_len, error=None):
        if success:
            logger.info("LLM stage=%s model=%s duration=%.1fs content_len=%d OK", stage_name, model_name, duration, content_len)
        else:
            logger.warning("LLM stage=%s model=%s duration=%.1fs FAILED: %s", stage_name, model_name, duration, error)

    # Single call - let the provider (FallbackProvider) handle fallback internally
    try:
        result = provider.generate(prompt, stage=stage, context={"model": model, "max_output_tokens": tokens})
        duration = _time.time() - start
        if result and not result.startswith(f"[{stage}]") and "error" not in result.lower():
            _log_attempt(stage, model, True, duration, len(result))
            return result
        else:
            _log_attempt(stage, model, False, duration, len(result) if result else 0, result)
            _record_fallback(stage, getattr(provider, "name", "provider") + " returned no usable content")
    except Exception as exc:
        duration = _time.time() - start
        _log_attempt(stage, model, False, duration, 0, str(exc))
        _record_fallback(stage, "local endpoints unavailable")
    
    # Deterministic fallback - configurable based on AI aggressiveness
    return _deterministic_fallback(stage, language=str((settings.__dict__ if hasattr(settings, '__dict__') else {}).get("language", "English")), aggressiveness=str((settings.__dict__ if hasattr(settings, '__dict__') else {}).get("ai_aggressiveness", "balanced")))


def _deterministic_fallback(stage: str, language: str = "English", aggressiveness: str = "balanced") -> str:
    """Return a deterministic fallback response for a given stage.
    
    aggressiveness: "conservative" | "balanced" | "aggressive" | "very_aggressive"
    """
    # Base conservative fallbacks
    conservative = {
        "decision": "HOLD — insufficient data for actionable decision",
        "discovery": "None supported by current evidence.",
        "summary": "## Summary\n- Unable to generate AI summary due to model unavailability.\n- Regime-based DCA schedule continues per configuration.\n- Reconciliation status determines data reliability.\n\n_Deterministic fallback in use — Priority Actions are generated from canonical portfolio rows below._",
        "ai_recommendations": '{"portfolio_actions": [], "trades": [], "position_updates": [], "cash_deployment": {"free_cash": 0, "deploy_amount": 0, "deploy_pct": 0, "reserve": 0, "targets": []}}',
    }
    
    # Aggressive fallbacks - more willing to act on limited data
    aggressive = {
        "decision": "HOLD — limited data; regime suggests cautious accumulation on dips",
        "discovery": "Scan watchlist for oversold quality names; prioritize high-conviction sectors.",
        "summary": "## Summary\n- AI model unavailable; using regime-based heuristic.\n- Regime signals primary driver; DCA continues per config.\n- Monitor watchlist for entry signals on volatility spikes.\n\n_Deterministic fallback (aggressive) — Priority Actions from canonical rows._",
        "ai_recommendations": '{"portfolio_actions": [{"action": "hold", "asset": "WATCHLIST", "reason": "monitor for entry signals"}], "trades": [], "position_updates": [], "cash_deployment": {"free_cash": 0, "deploy_amount": 0, "deploy_pct": 0, "reserve": 0, "targets": []}}',
    }
    
    # Very aggressive - willing to act on regime signals alone
    very_aggressive = {
        "decision": "BUY on dips / SELL on rips per regime — limited data but regime is clear",
        "discovery": "Aggressive scan: high-momentum watchlist names + oversold quality.",
        "summary": "## Summary\n- AI offline; aggressive regime-based heuristic active.\n- Deploying cash per regime signals; tight stops on all entries.\n- Watchlist prioritized for gap-up/breakout entries.\n\n_Deterministic fallback (very_aggressive) — Regime-driven actions._",
        "ai_recommendations": '{"portfolio_actions": [{"action": "buy", "asset": "WATCHLIST", "reason": "regime-driven accumulation"}], "trades": [], "position_updates": [], "cash_deployment": {"free_cash": 0, "deploy_amount": 0, "deploy_pct": 0, "reserve": 0, "targets": []}}',
    }
    
    profiles = {
        "conservative": conservative,
        "balanced": conservative,  # default safe
        "aggressive": aggressive,
        "very_aggressive": very_aggressive,
    }
    
    profile = profiles.get(aggressiveness.lower(), conservative)
    return profile.get(stage, "Data unavailable - model error")


def _passive_pie_lines(pies: list[dict[str, Any]]) -> str:
    lines = ["## Passive Pies — pie-level decision only"]
    for pie in pies:
        lines.append(f"- **{pie['name']}** — HOLD / regular add — long-term. Decisions are intentionally at Pie level; individual constituents are not analysed.")
    return "\n".join(lines)


def _load_symbol_aliases(portfolio_context: Dict[str, Any] | None) -> Dict[str, str]:
    """Load T212 -> Yahoo symbol aliases from the JSON config (symbol_aliases)."""
    try:
        cfg_path = (portfolio_context or {}).get("config_path")
        if not cfg_path:
            return {}
        import json as _json
        cfg = _json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        aliases = cfg.get("symbol_aliases", {}) or {}
        return {str(k).strip().upper(): str(v).strip() for k, v in aliases.items() if v}
    except Exception:
        return {}


def _t212_yahoo_symbol(t212_symbol: str, aliases: Dict[str, str]) -> str | None:
    """Map a Trading212 internal ID to a Yahoo Finance ticker (or None).

    Single mapping layer (see investment_engine.portfolio.symbols). Never
    returns company names. Kept for backward compatibility.
    """
    try:
        from investment_engine.portfolio.symbols import to_display_symbol, to_yahoo_symbol
        disp = to_display_symbol(t212_symbol)
        return to_yahoo_symbol(t212_symbol, disp, aliases)
    except Exception:
        return None


def _tech_summary_line(display: str, ind: dict | None) -> str:
    """One-line technical summary for LLM prompts (legacy-style per-asset block)."""
    if not ind:
        return f"{display}: no technical data"
    def _f(key, fmt="{:.2f}"):
        v = ind.get(key)
        try:
            return fmt.format(float(v)) if v is not None else "N/A"
        except (TypeError, ValueError):
            return "N/A"
    return (
        f"{display}: Price={_f('price')} | RSI={_f('RSI_14', '{:.1f}')} | "
        f"SMA20={_f('SMA_20')} | SMA50={_f('SMA_50')} | MACD={_f('MACD', '{:.4f}')} | "
        f"Support={_f('Support')} | Resistance={_f('Resistance')}"
    )


def _fetch_technicals_parallel(symbols: list) -> dict[str, dict]:
    """Fetch technical indicators, keyed by DISPLAY symbol.

    Accepts [(display, yahoo)] pairs (preferred) or plain Yahoo strings
    (backward compatible; keyed as given). Yahoo lookups only — never
    company names.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from investment_engine.research.market_data import fetch_technical_indicators
    pairs: list[tuple[str, str]] = []
    for item in dict.fromkeys(map(lambda x: x if isinstance(x, tuple) else (x, x), symbols or [])):
        disp, yahoo = item
        if yahoo and yahoo != "UNKNOWN":
            pairs.append((disp, yahoo))
    if not pairs:
        return {}
    out: dict[str, dict] = {}
    def _one(pair):
        disp, yahoo = pair
        try:
            from investment_engine.portfolio.symbols import support_state as _ss
            if _ss(yahoo, "market_data") != "SUPPORTED":
                return disp, None
            return disp, fetch_technical_indicators(yahoo, period="3mo", interval="1d")
        except Exception:
            return disp, None
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_one, p): p[0] for p in pairs}
        for f in as_completed(futs):
            try:
                disp, ind = f.result(timeout=60)
                if ind:
                    out[disp] = ind
            except Exception:
                pass
    return out


def _fetch_news_for_extra_tickers(settings_dict: dict, items: list[tuple]) -> dict[str, list[dict]]:
    """Fetch 48h news for extra tickers. items = [(key, yahoo_symbol, name)], limit 2 each."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out: dict[str, list[dict]] = {}
    items = [(k, s, n) for k, s, n in items if s and s != "UNKNOWN"]
    if not items:
        return out
    fetcher = StrictNewsFetcher(
        max_age_hours=settings_dict.get("market_regime", {}).get("news", {}).get("max_age_hours", 48),
        min_relevance=settings_dict.get("market_regime", {}).get("news", {}).get("min_relevance_score", 60),
    )
    def _one(item):
        key, sym, name = item
        try:
            return key, [i.__dict__ for i in fetcher.fetch_for_symbol(sym, name, limit=2)]
        except Exception:
            return key, []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_one, it): it[0] for it in items}
        for f in as_completed(futs):
            try:
                key, entries = f.result(timeout=60)
                if entries:
                    out[key] = entries
            except Exception:
                pass
    return out


def _fetch_earnings_for_symbols(yahoo_by_display: dict[str, str | None]) -> dict[str, str]:
    """Earnings dates keyed by DISPLAY symbol.

    Input maps display -> Yahoo ticker (or None when unsupported). Unsupported
    instruments resolve to 'unavailable' without any yfinance call; failures
    resolve to 'no_data'. Never queries company names.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from investment_engine.research.market_data import recent_earnings_date
    items = list((yahoo_by_display or {}).items())
    if not items:
        return {}
    out: dict[str, str] = {}
    for disp, yahoo in items:
        if not yahoo:
            out[disp] = "unavailable"
    todo = []
    try:
        from investment_engine.portfolio.symbols import support_state as _ss2
        for disp, yahoo in items:
            if not yahoo or disp in out:
                continue
            if _ss2(yahoo, "earnings") != "SUPPORTED":
                out[disp] = "unavailable"
            else:
                todo.append((disp, yahoo))
    except ImportError:
        todo = [(disp, yahoo) for disp, yahoo in items if yahoo and disp not in out]
    if not todo:
        return out
    def _one(pair):
        disp, yahoo = pair
        try:
            d = recent_earnings_date(yahoo)
            return disp, d if d else "no_data"
        except Exception:
            return disp, "no_data"
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_one, p): p[0] for p in todo}
        for f in as_completed(futs):
            try:
                disp, date = f.result(timeout=20)
                out[disp] = date
            except Exception:
                pass
    return out


def _headlines_block(news: dict[str, list[dict]]) -> str:
    """Compact headline list for prompts: SYM: title (source)."""
    parts = []
    for sym, entries in news.items():
        for e in entries[:2]:
            parts.append(f"{sym}: {e.get('title','')[:120]} ({e.get('source','')})")
    return "\n".join(parts) if parts else "None"


def _canonical_decisions(ai_recs: dict | None) -> dict[str, str]:
    """Canonical display-ticker → action map (see regime_report.build_canonical_signals).

    Single source of truth against duplicated/conflicting verdicts across
    report sections. Values are normalized BUY / SELL / HOLD.
    """
    try:
        from investment_engine.reporting.regime_report import build_canonical_signals
        canon_full = build_canonical_signals(ai_recs)
        out: dict[str, str] = {}
        for k, v in canon_full.items():
            disp = (v.get("display") if isinstance(v, dict) else None) or str(k).split("_")[0]
            sig = (v.get("signal") if isinstance(v, dict) else v) or "HOLD"
            out.setdefault(str(disp).strip().upper(), str(sig))
        return out
    except Exception:
        pass
    canon: dict[str, str] = {}
    if not isinstance(ai_recs, dict):
        return canon
    for t in ai_recs.get("trades", []) or []:
        if isinstance(t, dict) and t.get("asset"):
            canon[str(t["asset"]).strip().upper()] = str(t.get("side", "")).strip().upper() or "HOLD"
    for a in ai_recs.get("portfolio_actions", []) or []:
        if isinstance(a, dict) and a.get("asset"):
            canon.setdefault(str(a["asset"]).strip().upper(), str(a.get("action", "")).strip().upper() or "HOLD")
    return canon


def _portfolio_tickers_for_priority(t212_data: dict | None, known_clean: set | None = None) -> set[str]:
    """Allowed tickers for Priority Actions: display symbols of the unified set."""
    try:
        from investment_engine.portfolio.symbols import to_display_symbol as _td
    except Exception:
        _td = lambda s, *_: str(s or "").strip().upper().split("_")[0]
    tickers: set[str] = set()
    if not isinstance(t212_data, dict):
        return tickers
    summary = t212_data.get("account_summary", {}) or {}
    # Authoritative first; fallback only when unavailable; never merged.
    cands: list[dict] = list(summary.get("all_positions") or [])
    if not cands:
        cands = list(t212_data.get("all_positions") or [])
    if not cands:
        cands = list(t212_data.get("positions", []) or [])
        if not cands and isinstance(summary.get("positions"), list):
            cands = list(summary["positions"])
    for p in cands:
        if isinstance(p, dict) and p.get("symbol"):
            try:
                tickers.add(_td(str(p["symbol"]), known_clean))
            except Exception:
                tickers.add(str(p["symbol"]).strip().upper())
    return tickers


def _build_portfolio_names(all_active_assets, pie_blocks, t212_data, known_names_base: dict | None = None) -> dict[str, str]:
    """Display symbol -> company name (pie slices + config assets + overrides).

    Never maps a name to an internal broker ID, and never uses an internal
    ID as a name fallback.
    """
    names: dict[str, str] = dict(known_names_base or {})
    try:
        from investment_engine.portfolio.symbols import COMPANY_OVERRIDES, to_display_symbol as _td2
    except Exception:
        COMPANY_OVERRIDES = {}
        _td2 = lambda s, *_: str(s or "").strip().upper().split("_")[0]
    try:
        for item in all_active_assets or []:
            if isinstance(item, dict) and item.get("ticker") and item.get("name"):
                t, n = str(item["ticker"]), str(item["name"])
                if n.strip().upper() == t.strip().upper():
                    continue  # broker ID echoed as name — not a real company name
                disp = _td2(t)
                names.setdefault(disp, n)
    except Exception:
        pass
    try:
        for b in pie_blocks or []:
            for s in (b or {}).get("sample", []) or []:
                if isinstance(s, dict) and s.get("slice"):
                    disp = s.get("display") or _td2(str(s["slice"]))
                    names.setdefault(disp, str(s.get("name") or disp))
                if isinstance(s, dict) and s.get("name") and s.get("slice"):
                    names.setdefault(_td2(str(s["slice"])), str(s["name"]))
    except Exception:
        pass
    try:
        pools: list[dict] = []
        if isinstance(t212_data, dict):
            _s = t212_data.get("account_summary", {}) or {}
            pools.extend(_s.get("all_positions", []) or [])
            pools.extend(t212_data.get("all_positions", []) or [])
            pools.extend(t212_data.get("positions", []) or [])
        for p in pools:
            if isinstance(p, dict) and p.get("symbol") and p.get("name"):
                disp = _td2(str(p["symbol"]))
                nm = str(p["name"]).strip()
                if nm and nm.upper() != str(p["symbol"]).strip().upper():
                    names.setdefault(disp, nm)
    except Exception:
        pass
    for _k, _v in (COMPANY_OVERRIDES or {}).items():
        names.setdefault(str(_k).strip().upper(), str(_v))
    return names


def _validate_new_ideas(candidate_section: str | None, news_by_symbol: dict | None) -> str:
    """Render only validated structured ideas; never raw LLM prose.

    Shared extractor (see reporting.documents); this wrapper preserves the
    legacy "Potential New Ideas" section shape.
    """
    from investment_engine.reporting.documents import extract_validated_ideas
    FALLBACK = "No new ideas met the current evidence and validation threshold."
    ideas = extract_validated_ideas(candidate_section, news_by_symbol)
    lines = ["## Potential New Ideas", ""]
    if not ideas:
        lines.append(FALLBACK)
        return "\n".join(lines)
    lines.append("| Ticker | Evidence | Source | Confidence |")
    lines.append("|--------|----------|--------|------------|")
    for idea in ideas:
        lines.append(f"| **{idea['ticker']}** | {idea['evidence']} | {idea['url']} | {idea['confidence']} |")
    return "\n".join(lines)


def _build_monitoring_block(
    portfolio_rows: list[dict] | None,
    earnings_status: dict | None = None,
    technicals: dict | None = None,
) -> str:
    """Deterministic Portfolio Monitoring bullet for FAIL state.

    Owned HOLD positions with a priority trigger render REVIEW as a
    presentation-only status; everything else stays HOLD. Never WATCH.
    """
    earn_by_ticker: dict[str, str] = {}
    try:
        for disp, val in (earnings_status or {}).items():
            d = val[0][:10] if isinstance(val, (tuple, list)) and val and isinstance(val[0], str) else None
            if d:
                earn_by_ticker[str(disp).strip().upper()] = d
    except Exception:
        pass
    items: list[str] = []
    for r in sorted(portfolio_rows or [], key=lambda x: float((x or {}).get("market_value", 0) or 0), reverse=True):
        if not isinstance(r, dict):
            continue
        disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
        if not disp or str(r.get("signal", "HOLD")).strip().upper() != "HOLD":
            continue
        tech = (technicals or {}).get(disp) if isinstance(technicals, dict) else None
        trig = _hold_trigger_reason(r, earn_by_ticker.get(disp, ""), tech if isinstance(tech, dict) else None)
        if trig:
            items.append(f"**{disp}** — REVIEW: {trig}; canonical holding action remains HOLD. No transaction authorized.")
    body = "; ".join(items) if items else ""
    if body:
        body += "; "
    return f"- **Portfolio Monitoring:** {body}rest **HOLD**."


def _apply_fail_watchlist_sweep(
    summary_md: str,
    decision_sections: list,
    portfolio_rows: list[dict] | None = None,
    earnings_status: dict | None = None,
    technicals: dict | None = None,
) -> tuple[str, list]:
    """FAIL-state terminology sweep (see _build_monitoring_block).

    - "Actionable Tickers" line is replaced by the deterministic Portfolio
      Monitoring bullet (owned HOLDs stay HOLD; triggers render REVIEW).
    - Remaining BUY action labels become WATCH (research-only, non-owned).
    - Appends the visible research-only disclaimer.
    """
    import re as _re7
    monitoring = _build_monitoring_block(portfolio_rows, earnings_status, technicals)
    new_md, n_sub = _re7.subn(
        r"^-\s*\*\*Actionable Tickers:?\*\*:?.*$",
        monitoring,
        summary_md,
        flags=_re7.MULTILINE,
    )
    summary_md = new_md if n_sub else summary_md.rstrip() + f"\n\n{monitoring}"
    # Owned HOLD positions must never read as WATCH/BUY in FAIL prose.
    try:
        owned = sorted(
            {str((r or {}).get("display_symbol") or (r or {}).get("ticker") or "").strip().upper()
             for r in (portfolio_rows or []) if isinstance(r, dict)},
            key=len, reverse=True,
        )
        for disp in owned:
            if disp and disp != "UNKNOWN":
                summary_md = summary_md.replace(f"**{disp}** (WATCH)", f"**{disp}** (HOLD)")
                summary_md = _re7.sub(
                    r"\*\*" + _re7.escape(disp) + r"\s*\(WATCH\)(\*\*)?",
                    f"**{disp}** (HOLD)", summary_md)
        summary_md = _re7.sub(r"\bBUY\b(?=\*{0,2}\s+signals?)", "HOLD", summary_md)
    except Exception:
        pass
    summary_md = _re7.sub(r"\*\*Top Opportunities:?\*\*:?", "**Watchlist Candidates — WATCH:**", summary_md)
    summary_md = _re7.sub(r"\*\*BUY\*\*", "**WATCH**", summary_md)
    summary_md = _re7.sub(r"\(BUY(?=[,)])", "(WATCH", summary_md)
    disclaimer = ("Research-only watchlist; no deployment is authorized "
                  "while account reconciliation is failing.")
    if disclaimer not in summary_md:
        summary_md = summary_md.rstrip() + f"\n\n> {disclaimer}"
    return summary_md, decision_sections


def _strip_priority_actions_from_summary(summary_md: str) -> str:
    """Remove any LLM-generated Priority Actions table; it is rebuilt deterministically."""
    import re as _re3
    if not summary_md:
        return "## Summary\n- Portfolio review completed; see deterministic Priority Actions below."
    # Strip code fences the model sometimes wraps around the section.
    fence_stripped = _re3.sub(r"^\s*```(?:markdown|md)?\s*\n", "", summary_md.strip())
    fence_stripped = _re3.sub(r"\n?\s*```\s*$", "", fence_stripped).strip()
    summary_md = fence_stripped or summary_md
    # Cut from '## Priority Actions' (or '# Priority Actions') to end/next top header.
    cut = _re3.split(r"^#{1,3}\s*Priority Actions.*$", summary_md, flags=_re3.MULTILINE | _re3.IGNORECASE)
    base = cut[0].rstrip() if cut else summary_md
    # Also drop stray markdown tables that look like action tables (Ticker|Action).
    lines = []
    skip_table = False
    for ln in base.splitlines():
        if _re3.match(r"^\s*\|.*Ticker.*Action.*\|\s*$", ln, _re3.IGNORECASE):
            skip_table = True
            continue
        if skip_table:
            if _re3.match(r"^\s*\|", ln) or _re3.match(r"^\s*\|?[\s:\-|]+\|?\s*$", ln):
                continue
            if not ln.strip():
                continue
            skip_table = False
        lines.append(ln)
    out = "\n".join(lines).strip()
    return out or "## Summary\n- Portfolio review completed; see deterministic Priority Actions below."


def _apply_fail_guards(ai_recs: dict | None, recon_status: str) -> dict:
    """Deterministic FAIL-state guards (delegates to regime_report)."""
    try:
        from investment_engine.reporting.regime_report import apply_fail_guards as _guards
        return _guards(ai_recs, recon_status)
    except ImportError:
        return ai_recs or {}


def _resolve_account_performance(t212_data: dict | None, portfolio_context: dict | None) -> tuple[dict, dict]:
    """Resolve net deposits (verified API history > manual baseline) and build
    broker-equity-based account performance. Returns (performance, ledger_metadata).
    Never logs payloads, references, or credentials — counts and statuses only.
    """
    from investment_engine.accounting import cashflows as _cf
    summary = ((t212_data or {}).get("account_summary", {}) or {})
    cash = ((t212_data or {}).get("cash", {}) or {})
    equity = summary.get("total_equity")
    free_cash = cash.get("free", summary.get("cash_free"))
    pie_cash = cash.get("pie_cash", summary.get("cash_pie"))
    blocked_cash = summary.get("cash_blocked", cash.get("blocked", 0))

    api_result: dict | None = None
    if isinstance(t212_data, dict) and t212_data.get("status") == "ok":
        try:
            from trading212_integration import create_integration
            _t212 = create_integration(config_file="api.env")
            _client = getattr(getattr(_t212, "monitor", None), "client", None)
            if _client is not None:
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(_client.get_all_transaction_history)
                    try:
                        _pages = _fut.result(timeout=120)
                    except FuturesTimeoutError:
                        _pages = {"pages": [], "exhausted": False, "timeout": True}
                api_result = _cf.ingest_history_pages(
                    (_pages or {}).get("pages", []), bool((_pages or {}).get("exhausted", False)))
                try:
                    if api_result.get("status") == "VERIFIED":
                        _cf.save_history_cache(api_result)
                except Exception:
                    pass
                logger.info("Cash-flow history: status=%s items=%d pages=%d exhausted=%s",
                            api_result.get("status"), api_result.get("item_count", 0),
                            api_result.get("pages", 0), api_result.get("exhausted", False))
        except Exception as exc:
            logger.debug("Cash-flow history unavailable: %s", type(exc).__name__)

    try:
        _cfg_path = (portfolio_context or {}).get("config_path")
        _root = Path(_cfg_path).parent if _cfg_path else Path.cwd()
    except Exception:
        _root = Path.cwd()
    manual = _cf.load_manual_baseline(_cf.default_baseline_path(_root))
    # Coverage anchor: earliest known account activity (position first fills).
    # An API window starting later cannot be the full capital history.
    _all_pos = ((t212_data or {}).get("account_summary", {}) or {}).get("all_positions", []) or []
    _starts = [str(p.get("initial_fill", ""))[:10] for p in _all_pos
               if isinstance(p, dict) and str(p.get("initial_fill", ""))[:10]]
    _account_start = min(_starts) if _starts else None
    if isinstance(api_result, dict):
        api_result = _cf.enforce_history_coverage(api_result, _account_start)
    resolved = _cf.resolve_net_deposits(api_result, manual, _account_start)

    api_summary = (api_result or {}).get("summary", {}) if isinstance(api_result, dict) else {}
    metadata = {
        "api_status": (api_result or {}).get("status", "UNAVAILABLE"),
        "api_items": (api_result or {}).get("item_count", 0),
        "api_pages": (api_result or {}).get("pages", 0),
        "api_exhausted": bool((api_result or {}).get("exhausted", False)),
        "api_unresolved_fx": api_summary.get("unresolved_count", 0) if api_summary else 0,
        "manual_present": manual is not None,
        "manual_path": "data/portfolio_performance.toml",
        "selected_source": resolved.get("net_deposits_source"),
        "selected_status": resolved.get("net_deposits_status"),
        "fx_complete": bool(resolved.get("net_deposits_status") == "VERIFIED"
                            and (api_summary.get("unresolved_count", 0) if api_summary else 0) == 0)
                        or resolved.get("net_deposits_status") == "MANUAL",
    }
    performance = _cf.build_account_performance(
        equity, free_cash, pie_cash, blocked_cash,
        resolved.get("net_deposits_eur"), resolved.get("net_deposits_source"),
        resolved.get("net_deposits_status"),
        fees_eur=resolved.get("fees_eur"), interest_eur=resolved.get("interest_eur"),
        total_deposits_eur=resolved.get("total_deposits_eur"),
        total_withdrawals_eur=resolved.get("total_withdrawals_eur"),
    )
    return performance, metadata


def _sanitized_settings(settings) -> dict:
    """Settings safe for JSON reports: secret-bearing fields omitted entirely."""
    import re as _re_san
    try:
        raw = asdict(settings)
    except Exception:
        return {}
    clean: dict = {}
    for key, value in raw.items():
        if _re_san.search(r"key|secret|token|password", str(key), _re_san.IGNORECASE):
            continue
        elif isinstance(value, dict):
            clean[key] = {k: v for k, v in value.items()
                          if not _re_san.search(r"key|secret|token|password", str(k), _re_san.IGNORECASE)}
        else:
            clean[key] = value
    return clean


def _sanitized_t212_data(t212_data: dict | None) -> dict:
    """Broker snapshot safe for JSON reports: account identifiers removed."""
    import copy as _copy
    try:
        clean = _copy.deepcopy(t212_data) if isinstance(t212_data, dict) else {}
    except Exception:
        return {}
    try:
        summary = clean.get("account_summary")
        if isinstance(summary, dict):
            summary.pop("account_id", None)
    except Exception:
        pass
    return clean


def _hold_trigger_reason(row: dict, earnings_date: str = "", tech: dict | None = None) -> str:
    """Concrete trigger for surfacing a HOLD row, or '' when there is none.

    Triggers: earnings within 14 days, weight >= 5%, abs(P&L%) >= 10%,
    RSI <= 25 or >= 75, or price within 2% of a key level.
    """
    try:
        if earnings_date:
            from datetime import date as _d
            delta = (_d.fromisoformat(earnings_date[:10]) - _d.today()).days
            if 0 <= delta <= 14:
                return f"earnings in {delta}d"
    except (ValueError, TypeError):
        pass
    try:
        if float(row.get("weight", 0) or 0) >= 5.0:
            return "weight >= 5%"
    except (TypeError, ValueError):
        pass
    try:
        if abs(float(row.get("pnl_pct", 0) or 0)) >= 10.0:
            return "P&L move >= 10%"
    except (TypeError, ValueError):
        pass
    rsi = (tech or {}).get("RSI_14") if isinstance(tech, dict) else None
    try:
        if rsi is not None and (float(rsi) <= 25 or float(rsi) >= 75):
            return f"RSI {float(rsi):.1f} extreme"
    except (TypeError, ValueError):
        pass
    try:
        px = float(row.get("current_price", 0) or 0)
        for lvl in (row.get("support"), row.get("resistance")):
            if lvl is not None and px > 0 and abs(px - float(lvl)) / px <= 0.02:
                return "price near key level"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return ""


def _safe_pct(v) -> str:
    try:
        return f"{float(v or 0):+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _safe_money(v) -> str:
    try:
        return f"€{float(v or 0):+,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _safe_w(v) -> str:
    try:
        return f"{float(v or 0):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _build_priority_actions_table(
    portfolio_rows: list[dict] | None,
    earnings_status: dict | None = None,
    technicals: dict | None = None,
    max_rows: int = 10,
    canonical_map: dict | None = None,
) -> str:
    """Deterministic Priority Actions from canonical portfolio rows only.

    Every ticker exists in the T212 Portfolio table and every signal equals
    the canonical signal. SELL/BUY actions first; HOLD rows appear only with
    a concrete trigger (see _hold_trigger_reason). The table is never padded
    artificially.
    """
    import re as _re4
    rows = [r for r in (portfolio_rows or []) if isinstance(r, dict) and (r.get("display_symbol") or r.get("ticker"))]
    if not rows:
        return "### Priority Actions\n\nNo actionable signals — all holdings neutral on current data."
    for r in rows:
        disp = r.get("display_symbol") or r.get("ticker")
        sig = r.get("signal", "HOLD")
        if isinstance(canonical_map, dict):
            entry = canonical_map.get(str(disp).strip().upper())
            if isinstance(entry, dict) and entry.get("signal"):
                sig = entry["signal"]
        r["_norm"] = str(sig).strip().upper() if str(sig).strip().upper() in ("BUY", "SELL") else "HOLD"
        r["_disp"] = disp
    sells = sorted([r for r in rows if r["_norm"] == "SELL"], key=lambda r: float(r.get("market_value", 0) or 0), reverse=True)
    buys = sorted([r for r in rows if r["_norm"] == "BUY"], key=lambda r: float(r.get("market_value", 0) or 0), reverse=True)
    earn_by_ticker: dict[str, str] = {}
    try:
        for disp, val in (earnings_status or {}).items():
            d = val[0][:10] if isinstance(val, (tuple, list)) and val and isinstance(val[0], str) else None
            if d:
                earn_by_ticker[str(disp).strip().upper()] = d
    except Exception:
        pass
    holds: list[dict] = []
    for r in sorted([r for r in rows if r["_norm"] == "HOLD"], key=lambda r: float(r.get("market_value", 0) or 0), reverse=True):
        tech = (technicals or {}).get(r["_disp"])
        trig = _hold_trigger_reason(r, earn_by_ticker.get(str(r["_disp"]).strip().upper(), ""), tech if isinstance(tech, dict) else None)
        if trig:
            r["_trigger"] = trig
            holds.append(r)
    picked = (sells + buys + holds)[:max_rows]
    if not picked:
        return "### Priority Actions\n\nNo action triggers — all holdings neutral on current data."
    lines = ["### Priority Actions", "",
             "| Ticker | Action | Why | Horizon | Catalyst/Key Level |",
             "|--------|--------|-----|---------|---------------------|"]
    for r in picked:
        ticker = str(r["_disp"])
        sig = r["_norm"]
        emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL"}.get(sig, "🟡 HOLD")
        sup = r.get("support")
        res = r.get("resistance")
        try:
            sup_s = f"{float(sup):.2f}" if sup is not None else "n/a"
        except (TypeError, ValueError):
            sup_s = "n/a"
        try:
            res_s = f"{float(res):.2f}" if res is not None else "n/a"
        except (TypeError, ValueError):
            res_s = "n/a"
        tech = (technicals or {}).get(ticker)
        rsi_s = ""
        try:
            if isinstance(tech, dict) and tech.get("RSI_14") is not None:
                rsi_s = f"RSI {float(tech['RSI_14']):.1f}; "
        except (TypeError, ValueError):
            pass
        earn = earn_by_ticker.get(ticker.strip().upper(), "")
        earn_s = f"earnings {earn}; " if earn else ""
        trig_s = f"trigger: {r['_trigger']}; " if r.get("_trigger") else ""
        why = f"{trig_s}{rsi_s}{earn_s}P&L {_safe_pct(r.get('pnl_pct'))} ({_safe_money(r.get('unrealized_pnl'))}), weight {_safe_w(r.get('weight'))}".strip()
        horizon = "Swing" if sig == "SELL" else ("Long" if sig == "BUY" else "long-term")
        catalyst = f"Support {sup_s} / Resistance {res_s}"
        # Keep cells single-line.
        why = _re4.sub(r"\s*\|\s*", "; ", why)
        lines.append(f"| **{ticker}** | {emoji} | {why} | {horizon} | {catalyst} |")
    return "\n".join(lines)


def _enforce_canonical_in_pie_section(
    pie_md: str,
    canonical_map: dict | None,
    known_clean: set | None = None,
) -> str:
    """Rewrite PIE sample bullets to the canonical signal (display symbols).

    The technical observation is preserved after the em dash, e.g.
    ``NEE — HOLD — improving setup; watch €86.87 resistance before upgrade.``
    Internal broker IDs are replaced with display symbols.
    """
    if not pie_md or not isinstance(canonical_map, dict) or not canonical_map:
        return pie_md
    try:
        from investment_engine.portfolio.symbols import to_display_symbol as _td3
    except Exception:
        _td3 = lambda s, *_: str(s or "").strip().upper().split("_")[0]
    import re as _re5

    # Normalize any internal broker IDs in the prose to display symbols first.
    def _disp_sub(m):
        return _td3(m.group(0), known_clean)
    text = _re5.sub(r"\b[A-Z0-9]+_(?:US_EQ|DE_EQ|IM_EQ|EQ)\b", _disp_sub, pie_md, flags=_re5.IGNORECASE)

    def _repl(m):
        raw_ticker = m.group("ticker")
        raw_sig = m.group("sig")
        disp = _td3(raw_ticker, known_clean)
        entry = canonical_map.get(disp.strip().upper())
        if not isinstance(entry, dict) or not entry.get("signal"):
            # No canonical entry: normalize the ticker spelling only.
            if disp != raw_ticker:
                return f"{m.group('prefix')}{disp} — **{raw_sig.strip().upper()}**{m.group('rest')}"
            return m.group(0)
        want = entry["signal"]
        rest = m.group("rest") or ""
        # Preserve the observation; drop a contradictory trailing marker.
        rest = _re5.sub(r"\s*\(normalized to [^)]*\)\s*$", "", rest)
        marker = ""
        if want == "HOLD" and "HOLD —" not in rest:
            # Keep any existing observation text as-is.
            marker = ""
        return f"{m.group('prefix')}{disp} — **{want}**{rest}{marker}"

    # Matches bullets like: - **TICKER** — **SELL** — ... (display or internal IDs)
    pat = _re5.compile(
        r"(?P<prefix>(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?)(?P<ticker>[A-Za-z0-9.]{1,12}(?:_(?:US_EQ|DE_EQ|IM_EQ|EQ))?)(?:\*\*)?\s*—\s*(?:\*\*)?(?P<sig>BUY|SELL|HOLD|ACCUMULATE|ACCUMLATE|TRIM|REDUCE|WATCH|WAIT)(?:\*\*)?(?P<rest>[^\n]*)",
        _re5.IGNORECASE,
    )
    return pat.sub(_repl, text)


def _kpi_dashboard(
    regime_result,
    t212_data: dict | None,
    ai_recs: dict | None,
    portfolio_rows: list[dict] | None = None,
    earnings_status: dict | None = None,
    technicals: dict | None = None,
) -> str:
    """Executive KPI block: regime, action counts, top risks, cash.

    Top Risks and Cash Deployment are always populated — never '-', '—',
    blank, null, N/A. Risks are derived from actual holdings (concentration,
    drawdown, earnings proximity, technical extremes). Cash states a concrete
    allocation/action, with exactly one allowed fallback when cash is unknown.
    """
    if regime_result is None:
        regime_line = "Regime unavailable"
    else:
        try:
            close = regime_result.timeframes.get("daily", {}).get("indicators", {}).get("CLOSE", 0)
            rsi = regime_result.timeframes.get("daily", {}).get("indicators", {}).get("RSI_14", 0)
            regime_line = f"**{regime_result.regime}** (EXI2 €{close:,.2f}, RSI {rsi:.0f})"
        except Exception:
            regime_line = f"**{getattr(regime_result, 'regime', 'UNKNOWN')}**"
    risks = _derive_top_risks(regime_result, t212_data, portfolio_rows, earnings_status, technicals)
    # Action counts from normalized portfolio signals (single source of truth).
    buys = sells = 0
    try:
        from investment_engine.reporting.regime_report import normalize_signal as _norm
        if portfolio_rows:
            for r in portfolio_rows:
                s = _norm(r.get("signal"))
                if s == "BUY":
                    buys += 1
                elif s == "SELL":
                    sells += 1
        elif isinstance(ai_recs, dict):
            for t in ai_recs.get("trades", []) or []:
                s = _norm((t or {}).get("side", ""))
                if s == "BUY":
                    buys += 1
                elif s == "SELL":
                    sells += 1
    except Exception:
        pass
    total = buys + sells
    if total:
        actions_line = f"{total} actions ({buys} BUY / {sells} SELL)"
    else:
        n_hold = len(portfolio_rows or [])
        actions_line = f"0 BUY/SELL actions; {n_hold} positions on HOLD/neutral" if n_hold else "No portfolio actions; holdings neutral"
    cash_line = _format_cash_deployment(regime_result, t212_data, ai_recs, portfolio_rows)
    return "\n".join([
        "## Executive Dashboard",
        "",
        "| KPI | Value |",
        "|-----|-------|",
        f"| Market Regime | {regime_line} |",
        f"| Portfolio Actions | {actions_line} |",
        f"| Top Risks | {risks} |",
        f"| Cash & Deployment | {cash_line} |",
    ])


def _derive_top_risks(regime_result, t212_data, portfolio_rows, earnings_status, technicals) -> str:
    """Derive 2-3 concise risk items from actual holdings. Never blank/dash."""
    items: list[str] = []
    rows = list(portfolio_rows or [])
    # 1. Concentration: largest portfolio weight.
    try:
        if rows:
            top = max(rows, key=lambda r: float(r.get("weight", 0) or 0))
            w = float(top.get("weight", 0) or 0)
            if w > 0:
                items.append(f"Concentration: {top.get('ticker')} {w:.1f}% of equity (€{float(top.get('market_value', 0)):,.2f})")
    except Exception:
        pass
    # 2. Drawdown: worst unrealized P&L %.
    try:
        if rows:
            worst = min(rows, key=lambda r: float(r.get("pnl_pct", 0) or 0))
            wp = float(worst.get("pnl_pct", 0) or 0)
            if wp < 0:
                items.append(f"Drawdown: {worst.get('ticker')} {wp:+.1f}% (€{float(worst.get('unrealized_pnl', 0)):+,.2f})")
            elif worst.get("ticker"):
                items.append(f"P&L laggard: {worst.get('ticker')} {wp:+.1f}% — limited cushion on pullback")
    except Exception:
        pass
    # 3. Earnings proximity: nearest upcoming earnings among holdings.
    try:
        tickers = {str(r.get("ticker", "")).strip().upper() for r in rows}
        bases = {t.split("_")[0] for t in tickers}
        upcoming: list[tuple[str, str]] = []
        for disp, val in (earnings_status or {}).items():
            d = val[0][:10] if isinstance(val, (tuple, list)) and val else (val[:10] if isinstance(val, str) else None)
            if not d:
                continue
            key = str(disp).strip().upper()
            if key in tickers or key.split("_")[0] in bases:
                upcoming.append((d, str(disp)))
        upcoming.sort()
        if upcoming:
            nxt = "; ".join(f"{t} {d}" for d, t in upcoming[:2])
            items.append(f"Earnings event risk: {nxt}")
    except Exception:
        pass
    # 4. Technical extremes (RSI overbought/oversold) from available technicals.
    try:
        extremes: list[str] = []
        for sym, ind in (technicals or {}).items():
            if not isinstance(ind, dict):
                continue
            rsi = ind.get("RSI_14")
            if rsi is None:
                continue
            try:
                r = float(rsi)
            except (TypeError, ValueError):
                continue
            if r >= 70:
                extremes.append(f"{sym} overbought RSI {r:.1f}")
            elif r <= 30:
                extremes.append(f"{sym} oversold RSI {r:.1f}")
        if extremes:
            items.append("Technicals: " + "; ".join(sorted(set(extremes))[:2]))
    except Exception:
        pass
    # 5. Regime fallback (only to fill up to 2 items, still holding-aware).
    if len(items) < 2 and regime_result is not None:
        try:
            reg = getattr(regime_result, "regime", "NEUTRAL")
            if reg in ("PEAK_HOLD", "DECLINING"):
                items.append(f"Regime {reg}: reduce new exposure, prefer cash buffer")
            else:
                n = len(rows)
                items.append(f"Market {reg}: {n} holdings exposed to broad pullback; no cash hedge beyond reserve")
        except Exception:
            pass
    if not items:
        # Last-resort holding-aware fallback (still never blank/dash).
        n = len(rows)
        items.append(f"{n} holdings monitored; single-name and earnings risks under review")
    return "; ".join(items[:3])


def _format_cash_deployment(regime_result, t212_data, ai_recs, portfolio_rows) -> str:
    """Concrete cash allocation/action. Single allowed fallback when unknown."""
    # FAIL-state guard: deployment withheld until reconciliation passes.
    try:
        from investment_engine.reporting.regime_report import compute_reconciliation as _recon_fn2
        _recon2 = _recon_fn2(t212_data, portfolio_rows)
        if _recon2.get("status") == "FAIL":
            return (
                f"€{_recon2.get('reported_cash', 0):,.2f} reported cash — "
                f"deployment withheld until account reconciliation passes."
            )
    except Exception:
        pass
    free_cash = None
    deploy = None
    reserve = None
    targets: list[dict] = []
    if isinstance(ai_recs, dict) and isinstance(ai_recs.get("cash_deployment"), dict):
        cd = ai_recs["cash_deployment"]
        free_cash, deploy, reserve = cd.get("free_cash"), cd.get("deploy_amount"), cd.get("reserve")
        if isinstance(cd.get("targets"), list):
            targets = [t for t in cd["targets"] if isinstance(t, dict)]
    if free_cash is None and isinstance(t212_data, dict):
        free_cash = (t212_data.get("cash") or {}).get("free")
        if free_cash is None:
            free_cash = (t212_data.get("account_summary") or {}).get("cash_free")
    if free_cash is None:
        return "Cash balance unavailable — no allocation recommendation generated."
    try:
        fc = float(free_cash or 0)
    except (TypeError, ValueError):
        return "Cash balance unavailable — no allocation recommendation generated."
    # Cash target from regime (default 10%).
    cash_target_pct = 10.0
    try:
        cash_target_pct = float((regime_result.implications or {}).get("cash_target_pct", 10.0) or 10.0)
    except Exception:
        pass
    total_equity = 0.0
    try:
        total_equity = float(((t212_data or {}).get("account_summary") or {}).get("total_equity", 0) or 0)
    except Exception:
        pass
    # Explicit AI targets win.
    if targets and deploy:
        try:
            from investment_engine.portfolio.symbols import to_display_symbol as _td4
            _disp_map = {}
            for _r in portfolio_rows or []:
                if isinstance(_r, dict) and (_r.get("internal_id") or _r.get("ticker")):
                    _disp_map[str(_r.get("internal_id") or _r.get("ticker")).strip().upper()] = str(
                        _r.get("display_symbol") or _r.get("ticker"))
            def _tgt_disp(a):
                au = str(a or "?").strip().upper()
                if au in _disp_map:
                    return _disp_map[au]
                return _td4(str(a or "?"))
            tstr = ", ".join(f"{_tgt_disp(t.get('asset'))} €{float(t.get('amount', 0)):,.2f}" for t in targets[:3])
            return f"€{fc:,.2f} free — Deploy €{float(deploy):,.2f} to {tstr}; keep €{float(reserve or 0):,.2f} reserve (cash target {cash_target_pct:.0f}%)"
        except Exception:
            pass
    if deploy:
        try:
            return f"€{fc:,.2f} free — Deploy €{float(deploy):,.2f}; keep €{float(reserve if reserve is not None else fc - float(deploy)):,.2f} reserve (cash target {cash_target_pct:.0f}%)"
        except Exception:
            pass
    # Derive recommendation: BUY signals available?
    n_buy = 0
    try:
        from investment_engine.reporting.regime_report import normalize_signal as _norm2
        n_buy = sum(1 for r in (portfolio_rows or []) if _norm2(r.get("signal")) == "BUY")
    except Exception:
        pass
    fc_pct = (fc / total_equity * 100.0) if total_equity > 0 else 0.0
    if fc <= 0.01:
        return f"€{fc:,.2f} free ({fc_pct:.1f}% of equity) — No deployment; raise cash buffer toward {cash_target_pct:.0f}% target via planned SELLs"
    if n_buy > 0 and fc > 0:
        return f"€{fc:,.2f} free ({fc_pct:.1f}% of equity) — Hold reserve per {(getattr(regime_result, 'regime', 'NEUTRAL') if regime_result else 'NEUTRAL')} regime (cash target {cash_target_pct:.0f}%); deploy selectively to {n_buy} BUY signal(s) on confirmation"
    return f"€{fc:,.2f} free ({fc_pct:.1f}% of equity) — Hold as reserve; no new deployment (cash target {cash_target_pct:.0f}%, no validated BUY signals)"


def _format_earnings_radar(
    earnings_status: dict[str, tuple[str, str]],
    names: dict | None = None,
    unavailable: list[str] | None = None,
) -> str:
    """Chronological earnings timeline grouped by month, inline tags only.

    Strict format:
        ## Earnings Radar
        ## October 2026
        - Oct 14 — ASML, Company — Q3 earnings (confirmed)
    Within each month events are date-sorted ascending. Certainty appears only
    inline as (confirmed) or (estimated). No status column, no duplicates,
    no prose summaries. Instruments without coverage are listed once as
    unavailable (never reported as delisted).
    """
    try:
        from investment_engine.reporting.regime_report import _normalize_earnings_rows, _render_earnings_radar
        rows = _normalize_earnings_rows(earnings_status, names)
        if not rows:
            base = "## Earnings Radar\n\n*No upcoming earnings found in Yahoo Finance.*"
        else:
            base = _render_earnings_radar(rows)
        unav = sorted({str(u).strip().upper() for u in (unavailable or []) if u})
        dated = {r[1].strip().upper() for r in rows}
        missing = [u for u in unav if u not in dated]
        if missing:
            base += "\n\n*Earnings unavailable: " + ", ".join(missing[:12]) + ".*"
        return base.rstrip()
    except Exception:
        pass
    # Minimal deterministic fallback (same format, no imports).
    from datetime import date as _date
    import re as _re2
    norm: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    today = _date.today().isoformat()
    name_lookup = {str(k).strip().upper(): str(v) for k, v in (names or {}).items()}
    for disp, val in (earnings_status or {}).items():
        d = s = None
        if isinstance(val, (tuple, list)) and val:
            d = str(val[0])[:10]
            s = str(val[1]).lower().strip("() ") if len(val) > 1 and val[1] else "estimated"
        elif isinstance(val, str) and _re2.match(r"^\d{4}-\d{2}-\d{2}", val):
            d, s = val[:10], "estimated"
        if not d or d < today or (d, str(disp)) in seen:
            continue
        seen.add((d, str(disp)))
        st = s if s in ("confirmed", "estimated") else "estimated"
        company = name_lookup.get(str(disp).strip().upper(), str(disp).split("_")[0])
        norm.append((d, str(disp), company, st))
    if not norm:
        return "## Earnings Radar\n\n*No upcoming earnings found in Yahoo Finance.*"
    norm.sort(key=lambda r: (r[0], r[1]))
    months: dict[str, list] = {}
    order: list[str] = []
    for d, ticker, company, st in norm:
        try:
            header = f"## {_date.fromisoformat(d).strftime('%B %Y')}"
        except ValueError:
            continue
        if header not in months:
            months[header] = []
            order.append(header)
        months[header].append((d, ticker, company, st))
    lines = ["## Earnings Radar", ""]
    for header in order:
        lines.append(header)
        lines.append("")
        for d, ticker, company, st in sorted(months[header]):
            try:
                day = _date.fromisoformat(d).strftime("%b %d")
            except ValueError:
                day = d
            m = int(d[5:7])
            q = {1: "Q4", 2: "Q4", 3: "Q4", 4: "Q1", 5: "Q1", 6: "Q1", 7: "Q2", 8: "Q2", 9: "Q2", 10: "Q3", 11: "Q3", 12: "Q3"}.get(m, "")
            qlabel = f"{q} earnings" if q else "earnings"
            lines.append(f"- {day} — {ticker}, {company} — {qlabel} ({st})")
        lines.append("")
    return "\n".join(lines).rstrip()


def _methodology_block(t212_data: dict | None, provider) -> str:
    """Methodology & data-quality footer."""
    recon = "N/A"
    if isinstance(t212_data, dict):
        s = t212_data.get("account_summary", {}) or {}
        recon = f"{s.get('reconciliation_status', 'UNKNOWN')} (diff €{s.get('reconciliation_difference', 0):,.2f})"
    chain = getattr(provider, "name", "LLM chain")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "## Methodology & Data Quality",
        "",
        f"- Generated: {ts} | Sources: Yahoo Finance, Google News RSS (48h), Trading212 read-only API",
        f"- AI chain: {chain} (first success wins; local-first, public API fallback)",
        f"- Reconciliation: {recon} – per-position P&L is broker data; derived account totals withheld on FAIL",
        "- Full per-ticker analysis, technicals and headlines live in the AI context file.",
    ]
    note = _provider_fallback_note()
    if note:
        lines.append(note)
    return "\n".join(lines)


def _format_trump_watch(trump_watch: list[dict]) -> str:
    if not trump_watch:
        return ""
    lines = ["## Trump Watch (last 48h)"]
    for item in trump_watch:
        lines.append(f"- **{item.get('title','')}** — {item.get('source','')} · {item.get('published','')}")
    return "\n".join(lines)


def _format_news(news_by_symbol: dict[str, list[dict]]) -> str:
    if not news_by_symbol:
        return ""
    lines = ["## News (last 48h)"]
    shown = 0
    for symbol, entries in news_by_symbol.items():
        if symbol == "TRUMP":
            continue  # has its own Trump Watch section
        if not entries:
            continue
        shown += 1
        lines.append(f"### {symbol}")
        for item in entries[:3]:
            title = item.get('title', '')
            url = item.get('url', '')
            source = item.get('source', '')
            pub = item.get('published', '')
            lines.append(f"- [{title}]({url}) — {source} · {pub}")
    return "\n".join(lines) if shown else ""