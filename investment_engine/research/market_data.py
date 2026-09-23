from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import requests


logger = logging.getLogger(__name__)


BRATISLAVA_TZ_NAME = "Europe/Bratislava"


def report_now_bta():
    """Timezone-aware report timestamp in Europe/Bratislava."""
    from datetime import timezone as _tz
    from zoneinfo import ZoneInfo
    return datetime.now(_tz.utc).astimezone(ZoneInfo(BRATISLAVA_TZ_NAME))


def to_local_date(value, tz_name: str = BRATISLAVA_TZ_NAME):
    """Normalize a date/datetime/ISO string to a local calendar date.

    Aware datetimes convert to the target tz (handles midnight rollover);
    naive datetimes/dates/ISO dates are taken as local calendar dates.
    Returns None when unparseable.
    """
    from datetime import date as _date
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    try:
        if isinstance(value, _dt):
            if value.tzinfo is None:
                return value.date()
            return value.astimezone(ZoneInfo(tz_name)).date()
        if isinstance(value, _date):
            return value
        s = str(value or "").strip()
        if not s:
            return None
        if "T" in s or " " in s.strip() and any(c in s for c in ("+", "Z")):
            parsed = _dt.fromisoformat(s.replace("Z", "+00:00"))
            return to_local_date(parsed, tz_name)
        return _date.fromisoformat(s[:10])
    except (ValueError, TypeError, AttributeError):
        return None


def filter_earnings_window(
    events,
    report_dt=None,
    days: int = 7,
    tz_name: str = BRATISLAVA_TZ_NAME,
) -> dict:
    """Central seven-day earnings filter (inclusive, local calendar dates).

    events: iterable of dicts {display, company, date, status?, source_type?}.
    status is re-derived: (confirmed) only for source_type official_ir /
    exchange_notice, else (estimated). Duplicates by (display, date) collapse.
    Returns {"in_window": [...sorted by (date, display)...],
             "next_after": {...} | None}.
    """
    from datetime import date as _date
    from datetime import timedelta as _td
    if report_dt is None:
        report_dt = report_now_bta()
    report_date = to_local_date(report_dt, tz_name)
    if report_date is None:
        return {"in_window": [], "next_after": None}
    end_date = report_date + _td(days=days)
    seen: set[tuple[str, str]] = set()
    in_window: list[dict] = []
    future: list[dict] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        disp = str(ev.get("display", "") or "").strip().upper()
        d = to_local_date(ev.get("date"), tz_name)
        if not disp or d is None:
            continue
        key = (disp, d.isoformat())
        if key in seen:
            continue
        seen.add(key)
        norm = {"display": disp,
                "company": str(ev.get("company", "") or disp),
                "date": d.isoformat(),
                "status": classify_earnings_status(disp, d.isoformat(), ev.get("source_type"))}
        if report_date <= d <= end_date:
            in_window.append(norm)
        elif d > end_date:
            future.append(norm)
    in_window.sort(key=lambda e: (e["date"], e["display"]))
    future.sort(key=lambda e: (e["date"], e["display"]))
    return {"in_window": in_window, "next_after": future[0] if future else None}


# Central Yahoo gate: every yfinance call in this module passes through it.
# Company names, internal broker IDs, and known-unresolved forms never reach
# the network (they previously surfaced as misleading "delisted" errors).
_UNRESOLVED_YAHOO = frozenset({
    "VERTIV", "QUALCOMM", "APPL", "NVIDIA", "INTE", "BROADCOM",
    "BTCUSD", "ETHUSD",
})
_INTERNAL_ID_RE = None  # compiled lazily to keep import light


def _assert_resolved_yahoo(symbol: str) -> str:
    """Validate a Yahoo ticker before any yfinance call.

    Raises ValueError for company names, internal broker IDs
    (``*_US_EQ``/``*_EQ`` …), heuristic truncations, and overlong inputs.
    Returns the stripped symbol otherwise.
    """
    global _INTERNAL_ID_RE
    import re as _re

    s = (symbol or "").strip()
    if not s or " " in s or len(s) > 16:
        raise ValueError(f"unresolved Yahoo symbol: {symbol!r}")
    if _INTERNAL_ID_RE is None:
        _INTERNAL_ID_RE = _re.compile(r"_(US|DE|IM|L|D)_EQ$|_EQ$", _re.IGNORECASE)
    if _INTERNAL_ID_RE.search(s) or s.upper() in _UNRESOLVED_YAHOO:
        raise ValueError(f"unresolved Yahoo symbol: {symbol!r}")
    return s


# Per-run market-data registry: SUPPORTED symbols whose Yahoo request failed
# are marked FETCH_FAILED and skipped for the rest of the run.
_FETCH_FAILED: set[str] = set()


def reset_market_data_registry() -> None:
    """Clear per-run FETCH_FAILED state (called at engine start)."""
    _FETCH_FAILED.clear()


def _check_supported(symbol: str, purpose: str) -> str | None:
    """Return the symbol when a yfinance request is allowed, else None."""
    try:
        from investment_engine.portfolio.symbols import support_state, SUPPORTED
    except ImportError:
        return symbol
    s = _assert_resolved_yahoo(symbol)
    if s in _FETCH_FAILED:
        logger.debug("Skipping %s (FETCH_FAILED this run)", s)
        return None
    if support_state(s, purpose) != SUPPORTED:
        logger.debug("Skipping %s (%s %s)", s, support_state(s, purpose), purpose)
        return None
    return s


def _mark_fetch_failed(symbol: str) -> None:
    _FETCH_FAILED.add(symbol)
    logger.debug("Market-data FETCH_FAILED for %s; skipped for rest of run", symbol)


def fetch_recent_headlines(query: str, *, limit: int = 2, max_age_days: int = 2) -> list[dict[str, str]]:
    """Fetch only recent headline metadata from Google News RSS.

    The report receives title, source, date and URL only. This avoids passing
    full article text to the model while keeping every statement traceable.
    """
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "PortfolioAI/1.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", query, exc)
        return []

    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        if published < cutoff:
            continue
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "unknown").strip()
        if title and link:
            items.append({"title": title, "source": source, "published": published.date().isoformat(), "url": link})
        if len(items) >= limit:
            break
    return items


def recent_earnings_date(symbol: str) -> str | None:
    """Return the closest reported/upcoming earnings date when Yahoo exposes it.

    Company names and empty inputs are rejected without a network call
    (they are not Yahoo tickers — previously misreported as delisted).
    """
    s = (symbol or "").strip()
    if not s or " " in s or len(s) > 16:
        logger.debug("Earnings lookup skipped for non-ticker input: %r", symbol)
        return None
    try:
        s = _assert_resolved_yahoo(s)
    except ValueError:
        logger.debug("Earnings lookup skipped for unresolved symbol: %r", symbol)
        return None
    if _check_supported(s, "earnings") is None:
        return None
    try:
        import yfinance as yf

        dates = yf.Ticker(s).get_earnings_dates(limit=4)
        if dates is None or dates.empty:
            return None
        index = dates.index[0]
        return index.date().isoformat() if hasattr(index, "date") else str(index)
    except Exception as exc:
        _mark_fetch_failed(s)
        logger.debug("Earnings lookup unavailable for %s: %s", s, exc)
        return None


# Source types that alone justify a (confirmed) earnings tag.
CONFIRMED_EARNINGS_SOURCES: frozenset = frozenset({"official_ir", "exchange_notice"})


def classify_earnings_status(
    display_symbol: str,
    date_iso: str,
    source_type: str | None = None,
) -> str:
    """Classify an earnings date as 'confirmed' or 'estimated'.

    'confirmed' requires source_type 'official_ir' or 'exchange_notice'.
    yfinance/aggregator/calendar-derived dates (including past ones) are
    always 'estimated'.
    """
    _ = (display_symbol, date_iso)
    if (source_type or "").strip().lower() in CONFIRMED_EARNINGS_SOURCES:
        return "confirmed"
    return "estimated"


def analyst_consensus(symbol: str) -> str | None:
    """Return a compact broker-consensus snapshot when Yahoo exposes one."""
    try:
        symbol = _assert_resolved_yahoo(symbol)
    except ValueError:
        return None
    if _check_supported(symbol, "market_data") is None:
        return None
    try:
        import yfinance as yf

        summary = yf.Ticker(symbol).recommendations_summary
        if summary is None or summary.empty:
            return None
        row = summary.iloc[0].to_dict()
        period = row.get("period", "recent")
        buys = int(row.get("strongBuy", 0) or 0) + int(row.get("buy", 0) or 0)
        holds = int(row.get("hold", 0) or 0)
        sells = int(row.get("sell", 0) or 0) + int(row.get("strongSell", 0) or 0)
        return f"{period}: buy={buys}, hold={holds}, sell={sells}"
    except Exception as exc:
        _mark_fetch_failed(symbol)
        logger.debug("Analyst consensus unavailable for %s: %s", symbol, exc)
        return None


def fetch_technical_indicators(symbol: str, period: str = "3mo", interval: str = "1d") -> dict | None:
    """Fetch key technical indicators for a resolved Yahoo symbol using yfinance.
    
    Falls back to Playwright scraper (TradingView) when yfinance fails and 
    playwright fallback is enabled in settings.
    """
    try:
        symbol = _assert_resolved_yahoo(symbol)
    except ValueError:
        return None
    if _check_supported(symbol, "market_data") is None:
        return None
    
    # Try yfinance first
    try:
        return _fetch_technical_yfinance(symbol, period, interval)
    except Exception as exc:
        _mark_fetch_failed(symbol)
        logger.debug("yfinance technical indicators unavailable for %s: %s", symbol, exc)
    
    # Fallback to Playwright scraper (TradingView)
    try:
        from investment_engine.config.settings import EngineSettings
        settings = EngineSettings.from_defaults()
        if getattr(settings, "use_playwright_fallback", True):
            logger.info("Falling back to Playwright scraper for %s", symbol)
            return _fetch_technical_playwright(symbol)
    except Exception as exc:
        logger.debug("Playwright fallback failed for %s: %s", symbol, exc)
    
    _mark_fetch_failed(symbol)
    logger.debug("All technical indicator sources failed for %s", symbol)
    return None


def _fetch_technical_yfinance(symbol: str, period: str = "3mo", interval: str = "1d") -> dict | None:
    """Fetch technical indicators using yfinance (original implementation)."""
    import yfinance as yf
    import pandas as pd
    
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist is None or hist.empty or len(hist) < 50:
        return None
    
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    volume = hist['Volume']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # SMAs
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    
    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    macd_hist = macd - signal
    
    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Support/Resistance (simple: recent lows/highs)
    recent_high = high.rolling(20).max()
    recent_low = low.rolling(20).min()
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    current_price = float(close.iloc[-1])
    current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    current_sma20 = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None
    current_sma50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None
    current_sma200 = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None
    current_macd = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None
    current_signal = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None
    current_bb_upper = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None
    current_bb_lower = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None
    current_support = float(recent_low.iloc[-1]) if not pd.isna(recent_low.iloc[-1]) else None
    current_resistance = float(recent_high.iloc[-1]) if not pd.isna(recent_high.iloc[-1]) else None
    current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    
    return {
        "price": current_price,
        "RSI_14": round(current_rsi, 1) if current_rsi else None,
        "SMA_20": round(current_sma20, 2) if current_sma20 else None,
        "SMA_50": round(current_sma50, 2) if current_sma50 else None,
        "SMA_200": round(current_sma200, 2) if current_sma200 else None,
        "MACD": round(current_macd, 4) if current_macd else None,
        "MACD_Signal": round(current_signal, 4) if current_signal else None,
        "BB_Upper": round(current_bb_upper, 2) if current_bb_upper else None,
        "BB_Lower": round(current_bb_lower, 2) if current_bb_lower else None,
        "Support": round(current_support, 2) if current_support else None,
        "Resistance": round(current_resistance, 2) if current_resistance else None,
        "ATR": round(current_atr, 2) if current_atr else None,
    }


def _fetch_technical_playwright(symbol: str) -> dict | None:
    """Fetch technical indicators from TradingView using Playwright.
    
    Scrapes the TradingView symbol page for key technical indicators.
    """
    try:
        from playwright.sync_api import sync_playwright
        from investment_engine.config.settings import EngineSettings
        
        settings = EngineSettings.from_defaults()
        headless = getattr(settings, "playwright_headless", True)
        timeout = getattr(settings, "playwright_timeout_seconds", 30) * 1000
        
        # TradingView URL for the symbol
        # Map Yahoo symbol to TradingView format if needed
        tv_symbol = _yahoo_to_tradingview(symbol)
        url = f"https://www.tradingview.com/symbols/{tv_symbol}/technicals/"
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)  # Wait for JS to load
            
            # Extract technical indicators from the page
            # TradingView has a technicals summary widget
            indicators = {}
            
            # Try to extract from the technical summary
            try:
                # RSI
                rsi_el = page.query_selector('[data-name="RSI"] .value, .technicals-RSI .value')
                if rsi_el:
                    indicators["RSI_14"] = float(rsi_el.inner_text().strip())
            except:
                pass
            
            try:
                # MACD
                macd_el = page.query_selector('[data-name="MACD"] .value, .technicals-MACD .value')
                if macd_el:
                    indicators["MACD"] = float(macd_el.inner_text().strip())
            except:
                pass
            
            # Extract moving averages
            for ma_period in [20, 50, 200]:
                try:
                    ma_el = page.query_selector(f'[data-name="SMA{ma_period}"] .value, .technicals-SMA{ma_period} .value')
                    if ma_el:
                        indicators[f"SMA_{ma_period}"] = float(ma_el.inner_text().strip().replace(",", ""))
                except:
                    pass
            
            # Support/Resistance
            try:
                support_el = page.query_selector('[data-name="Support"] .value, .technicals-Support .value')
                if support_el:
                    indicators["Support"] = float(support_el.inner_text().strip().replace(",", ""))
            except:
                pass
            
            try:
                resistance_el = page.query_selector('[data-name="Resistance"] .value, .technicals-Resistance .value')
                if resistance_el:
                    indicators["Resistance"] = float(resistance_el.inner_text().strip().replace(",", ""))
            except:
                pass
            
            browser.close()
            
            if indicators:
                return indicators
            
    except Exception as exc:
        logger.debug("Playwright technical fetch failed for %s: %s", symbol, exc)
    
    return None


def _yahoo_to_tradingview(symbol: str) -> str:
    """Convert Yahoo Finance symbol to TradingView format."""
    # Handle common suffixes
    if symbol.endswith(".DE"):
        return symbol.replace(".DE", ":XETR")
    elif symbol.endswith(".L"):
        return symbol.replace(".L", ":LSE")
    elif symbol.endswith(".CO"):
        return symbol.replace(".CO", ":COPENHAGEN")
    elif symbol.endswith(".AS"):
        return symbol.replace(".AS", ":AMSTERDAM")
    elif symbol.endswith(".MI"):
        return symbol.replace(".MI", ":MILAN")
    elif symbol.endswith(".PA"):
        return symbol.replace(".PA", ":PARIS")
    elif symbol.endswith(".SW"):
        return symbol.replace(".SW", ":SWISS")
    elif symbol.endswith(".VI"):
        return symbol.replace(".VI", ":VIENNA")
    elif symbol.endswith(".ST"):
        return symbol.replace(".ST", ":STOCKHOLM")
    elif symbol.endswith(".HE"):
        return symbol.replace(".HE", ":HELSINKI")
    elif symbol.endswith(".OL"):
        return symbol.replace(".OL", ":OSLO")
    elif symbol.endswith(".IC"):
        return symbol.replace(".IC", ":REYKJAVIK")
    elif symbol.endswith(".KS"):
        return symbol.replace(".KS", ":KOREA")
    elif symbol.endswith(".TW"):
        return symbol.replace(".TW", ":TAIPEI")
    elif symbol.endswith(".HK"):
        return symbol.replace(".HK", ":HONGKONG")
    elif symbol.endswith(".TO"):
        return symbol.replace(".TO", ":TSX")
    elif symbol.endswith(".V"):
        return symbol.replace(".V", ":TSXV")
    elif symbol.endswith(".AX"):
        return symbol.replace(".AX", ":ASX")
    # US symbols - TradingView uses them directly
    return symbol
