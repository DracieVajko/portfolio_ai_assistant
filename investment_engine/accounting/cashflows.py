"""Account cash-flow ingestion, multi-currency normalization, and performance.

Single home for everything that turns broker cash movements into a verified
net-deposits baseline and broker-equity-based account performance:

- one centralized EUR conversion (historical rates, GBX handling),
- read-only T212 transaction-history ingestion (paginated, cached, sanitized),
- manual TOML fallback baseline (``data/portfolio_performance.toml``),
- source-priority resolution (verified API > manual > unavailable),
- ``build_account_performance``: broker equity minus net deposits.

Financial semantics (non-negotiable):
- Broker total equity already reflects all costs, P&L, dividends and FX
  effects. Fees are display-only analytics and are NEVER subtracted twice.
- BUY/SELL orders are not deposits/withdrawals. Unknown types never
  silently become capital flows.
- A manual baseline is never combined with API deposits/withdrawals from
  the same period (no double-counting).
"""

from __future__ import annotations

import logging
import re
from datetime import date as _date
from datetime import datetime as _dt
from datetime import timezone as _tz
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flow-type classification
# ---------------------------------------------------------------------------

DEPOSIT = "DEPOSIT"
WITHDRAW = "WITHDRAW"
FEE = "FEE"
TRANSFER = "TRANSFER"
INTEREST_ON_FREE_CASH = "INTEREST_ON_FREE_CASH"
LENDING_INTEREST = "LENDING_INTEREST"
UNKNOWN = "unknown"

_KNOWN_TYPES = frozenset({
    DEPOSIT, WITHDRAW, FEE, TRANSFER, INTEREST_ON_FREE_CASH, LENDING_INTEREST,
})

# Raw broker strings (any case, spaces/dashes tolerated) -> canonical type.
_TYPE_ALIASES: tuple[tuple[str, str], ...] = (
    ("deposit", DEPOSIT),
    ("withdraw", WITHDRAW),
    ("fee", FEE),
    ("commission", FEE),
    ("charge", FEE),
    ("transfer", TRANSFER),
    ("interest_on_free_cash", INTEREST_ON_FREE_CASH),
    ("free_cash_interest", INTEREST_ON_FREE_CASH),
    ("cash_interest", INTEREST_ON_FREE_CASH),
    ("lending_interest", LENDING_INTEREST),
    ("stock_lending", LENDING_INTEREST),
)


def classify_flow_type(raw_type: Any) -> str:
    """Map a raw broker transaction type to a canonical flow type (or unknown)."""
    s = re.sub(r"[\s\-]+", "_", str(raw_type or "").strip().lower())
    if not s:
        return UNKNOWN
    if s.upper() in _KNOWN_TYPES:
        return s.upper()
    for alias, canonical in _TYPE_ALIASES:
        if alias in s:
            return canonical
    return UNKNOWN


# ---------------------------------------------------------------------------
# Centralized EUR conversion (the only FX function for ledger items)
# ---------------------------------------------------------------------------

STATIC_FX_TO_EUR = {"EUR": 1.0, "USD": 0.922, "GBP": 1.183, "DKK": 0.134,
                    "CHF": 1.065, "SEK": 0.087, "NOK": 0.085}

_fx_cache: dict[tuple[str, str], tuple[float, str]] = {}


def _static_rate(currency: str) -> tuple[float | None, str]:
    rate = STATIC_FX_TO_EUR.get(currency.upper())
    if rate is None:
        return None, "unresolved"
    return rate, "static_fallback (estimate)"


def _yahoo_historical_rate(base_currency: str, on_date: _date) -> tuple[float | None, str]:
    """Historical <BASE>EUR rate via yfinance daily history (cached per run)."""
    key = (base_currency.upper(), on_date.isoformat())
    if key in _fx_cache:
        return _fx_cache[key]
    try:
        import yfinance as yf
        pair = "EURUSD=X" if base_currency.upper() == "USD" else f"{base_currency.upper()}EUR=X"
        hist = yf.Ticker(pair).history(
            start=on_date.isoformat(),
            end=(on_date.fromordinal(on_date.toordinal() + 5)).isoformat(),
            interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            result: tuple[float | None, str] = (None, "unresolved")
        else:
            close = hist["Close"]
            try:
                px = float(close.iloc[0])
            except (TypeError, ValueError, IndexError):
                px = None
            if px is None or px <= 0:
                result = (None, "unresolved")
            elif base_currency.upper() == "USD":
                result = (1.0 / px, f"yfinance EURUSD=X {on_date.isoformat()}")
            else:
                result = (px, f"yfinance {pair} {on_date.isoformat()}")
    except Exception as exc:
        logger.debug("Historical FX unavailable for %s @ %s: %s", base_currency, on_date, exc)
        result = (None, "unresolved")
    _fx_cache[key] = result
    return result


def convert_to_eur(
    amount: float,
    currency: str,
    on_date: _date | str | None = None,
    *,
    allow_estimate: bool = False,
    rate_fetcher: Callable[[str, _date], tuple[float | None, str]] | None = None,
) -> tuple[float | None, float | None, str, str]:
    """Convert one monetary amount to EUR (the only ledger FX function).

    GBX/pence amounts are divided by 100 to GBP first. Returns
    (amount_eur, fx_to_eur, fx_rate_source, conversion_status) where status is
    "ok", "estimate", or "unresolved". Without a historical rate the item is
    unresolved unless allow_estimate permits the static fallback.
    """
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return None, None, "unresolved", "unresolved"
    ccy = str(currency or "EUR").strip().upper()
    if ccy in ("GBX", "GBXP", "PENCE", "P"):
        amt = amt / 100.0
        ccy = "GBP"
    if ccy == "EUR":
        return round(amt, 2), 1.0, "EUR", "ok"
    day: _date | None = None
    if isinstance(on_date, _date):
        day = on_date
    elif on_date:
        try:
            day = _date.fromisoformat(str(on_date)[:10])
        except (ValueError, TypeError):
            day = None
    fetcher = rate_fetcher or _yahoo_historical_rate
    rate, source = (fetcher(ccy, day) if day is not None else (None, "unresolved"))
    if rate is None or rate <= 0:
        if allow_estimate:
            rate, source = _static_rate(ccy)
            if rate is not None:
                return round(amt * rate, 2), rate, source, "estimate"
        return None, None, "unresolved", "unresolved"
    return round(amt * rate, 2), rate, source, "ok"


# ---------------------------------------------------------------------------
# Ledger normalization (safe fields only — never refs, IDs, or payloads)
# ---------------------------------------------------------------------------

SAFE_LEDGER_FIELDS = frozenset({
    "date", "type", "original_amount", "original_currency", "fx_to_eur",
    "fx_rate_source", "amount_eur", "source", "verified", "conversion_status",
})

_DATE_KEYS = ("date", "dateTime", "datetime", "timestamp", "time", "created", "settled")
_TYPE_KEYS = ("type", "kind", "category", "transactionType")
_AMOUNT_KEYS = ("amount", "value", "total", "netAmount", "grossAmount")
_CCY_KEYS = ("currency", "ccy", "currencyCode")


def _first(raw: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if isinstance(raw, dict) and raw.get(k) is not None:
            return raw[k]
    return None


def _norm_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, _dt):
            dt = value if value.tzinfo else value.replace(tzinfo=_tz.utc)
            return dt.date().isoformat()
        s = str(value).strip()
        if not s:
            return ""
        if "T" in s or "+" in s[10:] or s.endswith("Z"):
            return _dt.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
        return s[:10]
    except (ValueError, TypeError):
        return str(value)[:10]


def normalize_cashflow_item(
    raw: dict,
    *,
    source: str = "t212_api",
    rate_fetcher: Callable | None = None,
) -> dict:
    """Normalize one raw broker transaction to safe report fields only."""
    flow_type = classify_flow_type(_first(raw, _TYPE_KEYS))
    day = _norm_date(_first(raw, _DATE_KEYS))
    try:
        original_amount = float(_first(raw, _AMOUNT_KEYS) or 0.0)
    except (TypeError, ValueError):
        original_amount = 0.0
    original_currency = str(_first(raw, _CCY_KEYS) or "EUR").strip().upper() or "EUR"
    amount_eur, fx_to_eur, fx_source, conv_status = convert_to_eur(
        original_amount, original_currency, day or None, rate_fetcher=rate_fetcher)
    item = {
        "date": day,
        "type": flow_type,
        "original_amount": round(original_amount, 2),
        "original_currency": original_currency,
        "fx_to_eur": fx_to_eur,
        "fx_rate_source": fx_source,
        "amount_eur": amount_eur,
        "source": source,
        "verified": conv_status == "ok",
        "conversion_status": conv_status,
    }
    return {k: item[k] for k in SAFE_LEDGER_FIELDS}


def summarize_cashflows(items: list[dict]) -> dict:
    """Capital-flow totals from normalized items (EUR only, post-conversion).

    BUY/SELL-like, FEE, interest, dividend, tax, transfer and unknown types
    never enter net deposits; fees/interest/tax stay display-only analytics.
    """
    totals = {"deposits": 0.0, "withdrawals": 0.0, "fees": 0.0,
              "interest": 0.0, "tax": 0.0, "transfers": 0.0, "unknown": 0.0}
    unresolved = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        t = it.get("type", UNKNOWN)
        amt = it.get("amount_eur")
        if amt is None:
            unresolved += 1
            continue
        try:
            amt = float(amt)
        except (TypeError, ValueError):
            unresolved += 1
            continue
        if t == DEPOSIT:
            totals["deposits"] += amt
        elif t == WITHDRAW:
            totals["withdrawals"] += amt
        elif t == FEE:
            totals["fees"] += amt
        elif t in (INTEREST_ON_FREE_CASH, LENDING_INTEREST):
            totals["interest"] += amt
        elif t == TRANSFER:
            totals["transfers"] += amt
        elif t == UNKNOWN:
            totals["unknown"] += 1
        # Anything else (orders, dividends, tax rows without mapping) is ignored
        # for capital flows by design.
    net = totals["deposits"] - totals["withdrawals"]
    return {
        "total_deposits_eur": round(totals["deposits"], 2),
        "total_withdrawals_eur": round(totals["withdrawals"], 2),
        "net_deposits_eur": round(net, 2),
        "fees_eur": round(totals["fees"], 2),
        "interest_eur": round(totals["interest"], 2),
        "transfers_eur": round(totals["transfers"], 2),
        "unknown_count": int(totals["unknown"]),
        "unresolved_count": unresolved,
        "item_count": len(items or []),
    }


# ---------------------------------------------------------------------------
# T212 transaction-history ingestion (read-only, paginated, cached, sanitized)
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_CACHE = Path("data/cache/t212_cashflow_history.json")


def _extract_next_cursor(payload: Any) -> str | None:
    """Next-page cursor from broker pagination (cursor / nextPagePath / links)."""
    if not isinstance(payload, dict):
        return None
    for key in ("cursor", "nextCursor", "next_cursor", "pageCursor"):
        val = payload.get(key)
        if val:
            return str(val)
    for key in ("nextPagePath", "next_page", "next"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            try:
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(val).query)
                for ck in ("cursor", "page", "offset"):
                    if qs.get(ck):
                        return qs[ck][0]
            except Exception:
                continue
    return None


def _extract_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for key in ("items", "transactions", "data", "results"):
            val = payload.get(key)
            if isinstance(val, list):
                return [p for p in val if isinstance(p, dict)]
    return []


def _history_status(items: list[dict], summary: dict, exhausted: bool, pages: int, max_pages: int) -> str:
    fully_paginated = exhausted and pages < max_pages
    if not items:
        return "UNAVAILABLE"
    if fully_paginated and summary["unresolved_count"] == 0:
        return "VERIFIED"
    return "PARTIAL"


def ingest_history_pages(
    pages: list[Any],
    exhausted: bool,
    *,
    rate_fetcher: Callable | None = None,
    max_pages: int = 40,
) -> dict:
    """Normalize raw history pages into items + completeness metadata."""
    raw_items: list[dict] = []
    for payload in pages or []:
        raw_items.extend(_extract_items(payload))
    items = [normalize_cashflow_item(raw, rate_fetcher=rate_fetcher) for raw in raw_items]
    summary = summarize_cashflows(items)
    return {
        "items": items,
        "status": _history_status(items, summary, exhausted, len(pages or []), max_pages),
        "pages": len(pages or []),
        "exhausted": exhausted,
        "item_count": len(items),
        "summary": summary,
    }


def fetch_cashflow_history(
    request_fn: Callable[..., Any],
    *,
    limit: int = 50,
    max_pages: int = 40,
    rate_fetcher: Callable | None = None,
) -> dict:
    """Pull the full transaction history via a read-only request callable.

    request_fn(endpoint, params) -> payload. Never logs references, account
    IDs, or raw payloads. Returns normalized items + completeness metadata.
    """
    import time as _time
    items: list[dict] = []
    cursor: str | None = None
    pages = 0
    exhausted = False
    seen_cursors: set[str] = set()
    while pages < max_pages:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        try:
            payload = request_fn("/api/v0/equity/history/transactions", params)
        except Exception as exc:
            logger.warning("Cash-flow history request failed: %s", type(exc).__name__)
            break
        if not isinstance(payload, dict) or payload.get("error") or payload.get("status") in (
                "auth_failed", "forbidden", "not_found", "error"):
            logger.warning("Cash-flow history unavailable: %s",
                           payload.get("status", "error") if isinstance(payload, dict) else "error")
            break
        batch = _extract_items(payload)
        if not batch:
            exhausted = True
            break
        items.extend(normalize_cashflow_item(raw, rate_fetcher=rate_fetcher) for raw in batch)
        pages += 1
        nxt = _extract_next_cursor(payload)
        if not nxt or nxt in seen_cursors:
            exhausted = True
            break
        seen_cursors.add(nxt)
        cursor = nxt
        _time.sleep(0.2)
    summary = summarize_cashflows(items)
    return {
        "items": items,
        "status": _history_status(items, summary, exhausted, pages, max_pages),
        "pages": pages,
        "exhausted": exhausted,
        "item_count": len(items),
        "summary": summary,
    }


def save_history_cache(result: dict, path: Path | str = DEFAULT_HISTORY_CACHE) -> Path:
    """Cache normalized history + cursor metadata (safe fields only)."""
    from datetime import datetime as _dt2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": _dt2.now(_tz.utc).isoformat(),
        "cursor_metadata": {"pages": result.get("pages", 0),
                            "exhausted": result.get("exhausted", False),
                            "item_count": result.get("item_count", 0),
                            "status": result.get("status", "UNAVAILABLE")},
        "items": [{k: it.get(k) for k in SAFE_LEDGER_FIELDS} for it in result.get("items", [])],
    }
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return path


def load_history_cache(path: Path | str = DEFAULT_HISTORY_CACHE) -> dict | None:
    """Load a cached history snapshot (normalized items only)."""
    try:
        import json as _json
        payload = _json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            return None
        return payload
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Manual fallback ledger (data/portfolio_performance.toml)
# ---------------------------------------------------------------------------

MANUAL_SOURCE = "Manual verified baseline"
API_SOURCE = "Trading212 transaction history (verified)"
UNAVAILABLE_SOURCE = "Unavailable — no verified cash-flow source"


def default_baseline_path(root: Path | str | None = None) -> Path:
    base = Path(root) if root else Path.cwd()
    return base / "data" / "portfolio_performance.toml"


def load_manual_baseline(path: Path | str | None = None) -> dict | None:
    """Read and validate the manual net-deposits baseline (None if absent/invalid)."""
    import tomllib
    try:
        with open(Path(path), "rb") as fh:
            cfg = tomllib.load(fh)
    except (OSError, ValueError):
        return None
    section = (cfg or {}).get("performance_baseline", {}) if isinstance(cfg, dict) else {}
    try:
        net = float(section.get("net_deposits_eur"))
    except (TypeError, ValueError):
        return None
    return {
        "net_deposits_eur": round(net, 2),
        "verified_at": str(section.get("verified_at", "") or ""),
        "source": str(section.get("source", "") or "manual_verified"),
        "note": str(section.get("note", "") or ""),
    }


def enforce_history_coverage(api_result: dict | None, account_start: str | None = None) -> dict | None:
    """Downgrade a VERIFIED-candidate history to PARTIAL when its window does
    not cover the account history (earliest item newer than account_start)."""
    if not (isinstance(api_result, dict) and api_result.get("status") == "VERIFIED" and account_start):
        return api_result
    try:
        earliest = min((str(it.get("date", ""))[:10] for it in api_result.get("items", [])
                        if isinstance(it, dict) and str(it.get("date", ""))[:10]),
                       default="")
        if earliest and earliest > str(account_start)[:10]:
            api = dict(api_result)
            api["status"] = "PARTIAL"
            api["coverage_note"] = (
                f"history starts {earliest}, account activity starts "
                f"{str(account_start)[:10]} — window does not cover full history")
            logger.info("Cash-flow history downgraded to PARTIAL: %s", api["coverage_note"])
            return api
    except (TypeError, ValueError):
        pass
    return api_result


def resolve_net_deposits(
    api_result: dict | None,
    manual: dict | None,
    account_start: str | None = None,
) -> dict:
    """Select the net-deposits source: verified API > manual > unavailable.

    A VERIFIED-candidate API history additionally must cover the account
    history (see enforce_history_coverage). Never combines a manual baseline
    with API deposits/withdrawals from the same period (no double-counting):
    exactly one source wins.
    """
    api = enforce_history_coverage(api_result, account_start)
    api_ok = (isinstance(api, dict) and api.get("status") == "VERIFIED"
              and isinstance(api.get("summary"), dict))
    if api_ok:
        summary = api["summary"]
        return {
            "net_deposits_eur": summary.get("net_deposits_eur"),
            "total_deposits_eur": summary.get("total_deposits_eur"),
            "total_withdrawals_eur": summary.get("total_withdrawals_eur"),
            "fees_eur": summary.get("fees_eur", 0.0),
            "interest_eur": summary.get("interest_eur", 0.0),
            "net_deposits_source": API_SOURCE,
            "net_deposits_status": "VERIFIED",
        }
    if isinstance(manual, dict) and isinstance(manual.get("net_deposits_eur"), (int, float)):
        return {
            "net_deposits_eur": manual["net_deposits_eur"],
            "total_deposits_eur": None,
            "total_withdrawals_eur": None,
            "fees_eur": None,
            "interest_eur": None,
            "net_deposits_source": MANUAL_SOURCE,
            "net_deposits_status": "MANUAL",
        }
    return {
        "net_deposits_eur": None,
        "total_deposits_eur": None,
        "total_withdrawals_eur": None,
        "fees_eur": None,
        "interest_eur": None,
        "net_deposits_source": UNAVAILABLE_SOURCE,
        "net_deposits_status": "UNAVAILABLE",
    }


# ---------------------------------------------------------------------------
# Centralized account performance
# ---------------------------------------------------------------------------

FAIL_PERFORMANCE_NOTE = (
    "Position reconciliation is FAIL; account performance uses broker total equity "
    "and verified net cash flows, not the summed position rows. The reconciliation "
    "delta is not a reported portfolio loss."
)


def build_account_performance(
    broker_total_equity_eur,
    reported_free_cash_eur,
    pie_cash_eur,
    blocked_cash_eur,
    net_deposits_eur,
    cash_flow_source,
    cash_flow_status,
    fees_eur=None,
    interest_eur=None,
    tax_eur=None,
    total_deposits_eur=None,
    total_withdrawals_eur=None,
) -> dict:
    """Broker-equity-based account performance (single calculation function).

    net_pnl_after_costs = equity − net deposits (broker equity already embeds
    all trading costs, FX effects, dividends and fees — fees_eur is
    informational and is NEVER subtracted again).
    """
    def _num(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    equity = _num(broker_total_equity_eur)
    free_cash = _num(reported_free_cash_eur)
    pie_cash = _num(pie_cash_eur)
    blocked = _num(blocked_cash_eur)
    net_dep = _num(net_deposits_eur)
    reported_cash = ((free_cash or 0.0) + (pie_cash or 0.0) + (blocked or 0.0)
                     if free_cash is not None or pie_cash is not None or blocked is not None else None)

    out: dict[str, Any] = {
        "broker_total_equity_eur": equity,
        "reported_free_cash_eur": free_cash,
        "pie_cash_eur": pie_cash,
        "reported_cash_eur": round(reported_cash, 2) if reported_cash is not None else None,
        "blocked_cash_eur": blocked,
        "net_deposits_eur": net_dep,
        "net_deposits_source": cash_flow_source,
        "net_deposits_status": cash_flow_status,
        "total_deposits_eur": _num(total_deposits_eur),
        "total_withdrawals_eur": _num(total_withdrawals_eur),
        "fees_eur": _num(fees_eur),
        "interest_eur": _num(interest_eur),
        "tax_eur": _num(tax_eur),
        "net_pnl_after_costs_eur": None,
        "return_pct": None,
        "cash_allocation_pct": None,
        "performance_status": "Unavailable",
        "note": "",
    }
    if equity is None or net_dep is None or net_dep <= 0:
        out["note"] = ("Account performance unavailable — requires broker total equity "
                       "and a positive verified net-deposits baseline.")
        return out
    pnl = equity - net_dep
    out["net_pnl_after_costs_eur"] = round(pnl, 2)
    out["return_pct"] = round(pnl / net_dep * 100, 4)
    if equity > 0 and reported_cash is not None:
        out["cash_allocation_pct"] = round(reported_cash / equity * 100, 2)
    out["performance_status"] = "Available"
    if str(cash_flow_status or "").upper() in ("PARTIAL", "UNAVAILABLE"):
        out["note"] = "Account performance unavailable — no verified cash-flow source."
        out["performance_status"] = "Unavailable"
        out["net_pnl_after_costs_eur"] = None
        out["return_pct"] = None
        out["cash_allocation_pct"] = None
    elif str(cash_flow_status or "").upper() == "MANUAL":
        out["note"] = ("Net deposits from manual verified baseline; broker-equity P&L "
                       "embeds all trading costs and FX effects (fees shown separately, not re-subtracted).")
    else:
        out["note"] = ("Net deposits from verified Trading212 transaction history; broker-equity P&L "
                       "embeds all trading costs and FX effects (fees shown separately, not re-subtracted).")
    return out
