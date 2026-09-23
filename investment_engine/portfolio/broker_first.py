"""Broker-first portfolio layer: Trading212 API is the primary source of truth.

Architecture (strict layering):
  raw broker data  ->  normalized broker portfolio  ->  market-data enrichment
  ->  analysis signals  ->  report rendering.

Rules enforced here:
- Account total, cash, positions, quantities, prices, P&L and instrument
  metadata come from the T212 API. Manual config, watchlists, local pie
  definitions and Yahoo Finance are NEVER used for real holding values.
- A position value is ``broker_position.market_value_eur`` when the broker
  provides it, otherwise ``quantity x broker_price x fx_rate``. NEVER
  ``quantity x yahoo_price x guessed_fx_rate``.
- External prices serve analytics only (RSI/SMA/MACD/support/news/sentiment).
- Dedupe key is ``(account_id, isin)``, fallback
  ``(account_id, broker_instrument_id)``. Ticker is presentation-only.
- Pie totals never enter the position sum; pies are grouping metadata only.
- Currency always comes from broker instrument metadata or an explicit
  override. GBX means pence: ``price_gbp = price_gbx / 100``. Broker EUR
  values are never re-converted.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth ISIN map (user-verified, 2026-09-11). Used for dedupe and
# alias verification. Keys: broker ticker (upper) AND display symbol (upper).
# ---------------------------------------------------------------------------

KNOWN_ISINS: Dict[str, str] = {
    "AI": "FR0000120073", "AIP_EQ": "FR0000120073",
    "SU": "FR0000121972", "SUP_EQ": "FR0000121972",
    "ETN": "IE00B8KQN827", "ETN_US_EQ": "IE00B8KQN827",
    "ENR": "DE000ENER6Y0", "ENRD_EQ": "DE000ENER6Y0",
    "NEE": "US65339F1012", "NEE_US_EQ": "US65339F1012",
    "LITM": "IE000WDG5795", "LITMM_EQ": "IE000WDG5795",
    "HTHIY": "US4335785071", "HTHIY_US_EQ": "US4335785071",
    "VWSB": "DK0061539921", "VWSBD_EQ": "DK0061539921",
    "APD": "US0091581068", "APD_US_EQ": "US0091581068",
    "IQQH": "IE00B1XNHC34", "IQQHD_EQ": "IE00B1XNHC34",
    "RWE": "DE0007037129", "RWED_EQ": "DE0007037129",
    "IBE": "ES0144580Y14", "IBEE_EQ": "ES0144580Y14",
    "GEV": "US36828A1016", "GEV_US_EQ": "US36828A1016",
    "UEC": "US9168961038", "UEC_US_EQ": "US9168961038",
    "C7A0": "CNE100006WS8", "C7A0D_EQ": "CNE100006WS8",
    "LIN": "IE000S9YS762", "LIN_US_EQ": "IE000S9YS762",
    "AAPL": "US0378331005", "AAPL_US_EQ": "US0378331005",
}
# Display symbol -> verified Yahoo ticker (probed: endpoint responds).
# Everything else per symbol policy (SUPPORTED set / US pattern / UNRESOLVED).
VERIFIED_YAHOO: Dict[str, str] = {
    "IBE": "IBE.MC",
    "RWE": "RWE.DE",
    "VWSB": "VWS.CO",
}

INSTRUMENT_CATALOG_CACHE = Path("data/cache/t212_instruments.json")
INSTRUMENT_CATALOG_MAX_AGE_DAYS = 7


def get_cached_catalog(path: Path | str = INSTRUMENT_CATALOG_CACHE) -> Dict[str, Dict[str, Any]]:
    """Load the cached instrument catalog (ticker -> metadata). Empty if absent."""
    try:
        import json as _json
        payload = _json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("instruments"), dict):
        return payload["instruments"]
    if isinstance(payload, dict):
        # Legacy flat form {ticker: {...}} written by earlier probes.
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    return {}


def save_instrument_catalog(catalog: Dict[str, Dict[str, Any]],
                            path: Path | str = INSTRUMENT_CATALOG_CACHE) -> Path:
    """Persist the instrument catalog with a fetch timestamp."""
    import json as _json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(
        {"fetched_at": _utcnow(), "count": len(catalog), "instruments": catalog},
        indent=1, default=str), encoding="utf-8")
    return path


def fetch_instrument_catalog(client: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Single-request full instrument catalog (read-only, best-effort).

    Returns (catalog, meta). On any failure returns ({}, meta with error).
    Never logs payload contents.
    """
    meta: Dict[str, Any] = {"attempted": True, "source": "T212_API"}
    try:
        resp = client.auth.make_request("GET", "/equity/metadata/instruments")
    except Exception as exc:
        meta.update({"ok": False, "error": type(exc).__name__})
        return {}, meta
    if not isinstance(resp, list):
        meta.update({"ok": False, "error": "unexpected_shape"})
        return {}, meta
    catalog: Dict[str, Dict[str, Any]] = {}
    for inst in resp:
        if not isinstance(inst, dict) or not inst.get("ticker"):
            continue
        catalog[str(inst["ticker"]).strip()] = {
            "isin": inst.get("isin"),
            "currencyCode": inst.get("currencyCode"),
            "type": inst.get("type"),
            "name": inst.get("name"),
        }
    meta.update({"ok": True, "count": len(catalog)})
    return catalog, meta


def ensure_instrument_catalog(
    client: Any = None,
    path: Path | str = INSTRUMENT_CATALOG_CACHE,
    max_age_days: int = INSTRUMENT_CATALOG_MAX_AGE_DAYS,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Catalog from cache when fresh, else best-effort live refresh.

    Returns (catalog, meta with source/age). Never raises; on total failure
    returns ({}, meta) and callers must mark instruments accordingly.
    """
    import json as _json
    from datetime import timedelta as _td
    meta: Dict[str, Any] = {"source": "cache"}
    try:
        payload = _json.loads(Path(path).read_text(encoding="utf-8"))
        fetched = str(payload.get("fetched_at", ""))
        age_days = None
        try:
            age_days = (datetime.now(timezone.utc)
                        - datetime.fromisoformat(fetched)).total_seconds() / 86400.0
        except (ValueError, TypeError):
            pass
        instruments = payload.get("instruments", payload) if isinstance(payload, dict) else {}
        if isinstance(instruments, dict) and instruments and age_days is not None and age_days <= max_age_days:
            meta.update({"ok": True, "age_days": round(age_days, 2), "count": len(instruments)})
            return instruments, meta
        meta["stale_reason"] = "missing" if not instruments else f"age {age_days} days"
    except (OSError, ValueError) as exc:
        meta["stale_reason"] = type(exc).__name__
    if client is None:
        meta.update({"ok": False})
        return {}, meta
    catalog, live_meta = fetch_instrument_catalog(client)
    if live_meta.get("ok") and catalog:
        try:
            save_instrument_catalog(catalog, path)
        except OSError:
            pass
        live_meta["source"] = "live"
        return catalog, live_meta
    # Fall back to the stale cache rather than nothing.
    try:
        payload = _json.loads(Path(path).read_text(encoding="utf-8"))
        instruments = payload.get("instruments", payload) if isinstance(payload, dict) else {}
        if isinstance(instruments, dict) and instruments:
            meta.update({"ok": True, "stale": True, "count": len(instruments)})
            return instruments, meta
    except (OSError, ValueError):
        pass
    meta.update({"ok": False})
    return {}, meta


def lookup_instrument(ticker: str, catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """ISIN/currency/venue for one broker ticker ({} when unknown)."""
    if catalog is None:
        catalog = get_cached_catalog()
    entry = (catalog or {}).get(str(ticker or "").strip(), {})
    return entry if isinstance(entry, dict) else {}
NOT_AVAILABLE_FROM_T212_API = "NOT_AVAILABLE_FROM_T212_API"
CURRENCY_UNRESOLVED = "CURRENCY_UNRESOLVED"

# Static fallback rates (quote -> EUR). Last-resort only, always flagged.
STATIC_FX_TO_EUR: Dict[str, float] = {
    "EUR": 1.0, "USD": 0.922, "GBP": 1.183, "DKK": 0.134,
    "CHF": 1.065, "SEK": 0.087, "NOK": 0.085, "CAD": 0.63,
}

_live_fx_cache: Dict[str, Tuple[float, str]] = {}
_live_fx_ts: float = 0.0
LIVE_FX_TTL_S = 3600.0


# ---------------------------------------------------------------------------
# Normalized model
# ---------------------------------------------------------------------------

@dataclass
class CashBalance:
    free_cash_eur: float = 0.0
    pie_cash_eur: float = 0.0
    blocked_cash_eur: float = 0.0
    broker_invested_eur: float = 0.0
    broker_total_equity_eur: float = 0.0
    broker_ppl_eur: float = 0.0
    source: str = "T212_API"
    retrieved_at: str = ""
    # Internal flag to distinguish "explicitly zero" from "missing from API"
    _blocked_explicitly_zero: bool = field(default=False, repr=False)

    @property
    def total_reported_cash_eur(self) -> float:
        return self.free_cash_eur + self.pie_cash_eur + self.blocked_cash_eur


@dataclass
class BrokerInstrument:
    broker_instrument_id: str = ""
    broker_ticker: str = ""
    display_symbol: str = ""
    instrument_name: str = ""
    isin: Optional[str] = None
    currency: str = CURRENCY_UNRESOLVED
    venue: Optional[str] = None
    source: str = "T212_API"
    retrieved_at: str = ""


@dataclass
class FxAudit:
    source_currency: str = ""
    target_currency: str = "EUR"
    raw_price: Optional[float] = None
    normalized_price: Optional[float] = None
    minor_unit_factor: int = 1
    rate: Optional[float] = None
    rate_source: str = ""
    rate_timestamp: str = ""
    result_eur: Optional[float] = None


@dataclass
class BrokerPosition:
    instrument: BrokerInstrument = field(default_factory=BrokerInstrument)
    quantity: float = 0.0
    average_price_raw: Optional[float] = None
    current_price_raw: Optional[float] = None
    broker_market_value_raw: Optional[float] = None
    broker_market_value_eur: Optional[float] = None
    average_cost_eur: Optional[float] = None
    pnl_eur: Optional[float] = None
    pnl_fx_eur: Optional[float] = None
    pie_id: Optional[str] = None
    pie_name: Optional[str] = None
    pie_quantity: float = 0.0
    initial_fill_date: Optional[str] = None
    valuation_source: str = ""
    mapping_status: str = ""
    confidence: str = ""
    source: str = "T212_API"
    retrieved_at: str = ""
    fx_audit: FxAudit = field(default_factory=FxAudit)
    included_in_position_total: bool = True
    exclusion_reason: str = ""
    duplicate_of: str = ""

    @property
    def dedupe_key(self) -> Tuple[str, str]:
        isin = (self.instrument.isin or "").strip().upper()
        if isin:
            return ("account", isin)
        return ("account", (self.instrument.broker_instrument_id or "").strip().upper())


@dataclass
class BrokerTransaction:
    date: str = ""
    flow_type: str = "unknown"
    ticker: Optional[str] = None
    isin: Optional[str] = None
    quantity: Optional[float] = None
    original_amount: Optional[float] = None
    original_currency: Optional[str] = None
    amount_eur: Optional[float] = None
    fx_to_eur: Optional[float] = None
    source: str = "T212_API"
    verified: bool = False


@dataclass
class PieAllocation:
    pie_id: str = ""
    pie_name: str = ""
    member_instrument_ids: List[str] = field(default_factory=list)
    note: str = "grouping metadata only — pie totals never enter the position sum"


@dataclass
class AccountSnapshot:
    retrieved_at: str = ""
    cash: CashBalance = field(default_factory=CashBalance)
    positions: List[BrokerPosition] = field(default_factory=list)
    pies: List[PieAllocation] = field(default_factory=list)
    transactions: List[BrokerTransaction] = field(default_factory=list)
    source: str = "T212_API"


@dataclass
class ReconciliationResult:
    broker_total_equity_eur: float = 0.0
    reported_free_cash_eur: float = 0.0
    reported_blocked_cash_eur: float = 0.0
    pie_cash_eur: float = 0.0
    total_reported_cash_eur: float = 0.0
    pending_cash_adjustments_eur: float = 0.0
    expected_open_positions_value_eur: float = 0.0
    sum_raw_broker_position_values_eur: float = 0.0
    sum_deduplicated_broker_position_values_eur: float = 0.0
    sum_externally_computed_position_values_eur: float = 0.0
    sum_excluded_values_eur: float = 0.0
    reconciliation_delta_eur: float = 0.0
    tolerance_eur: float = 0.0
    reconciliation_status: str = "UNKNOWN"
    data_quality: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FX: broker-implied calibration, live rates, static fallback
# ---------------------------------------------------------------------------

def normalize_minor_unit(price: float, unit: Optional[str]) -> Tuple[float, int]:
    """GBX/pence -> GBP (/100). Returns (normalized_price, minor_factor)."""
    if price is None:
        return price, 1
    if str(unit or "").strip().upper() in ("GBX", "GBXP", "PENCE", "P"):
        return float(price) / 100.0, 100
    return float(price), 1


def implied_rate_from_pnl(
    quantity: float,
    avg_price: float,
    current_price: float,
    ppl_eur: float,
    fx_ppl_eur: float,
) -> Optional[float]:
    """Broker-implied current FX rate from ppl/fxPpl.

    ppl - fxPpl = qty x (cur - avg) x r1  =>  r1 = (ppl - fxPpl) / (qty x (cur - avg)).
    Returns None when the denominator is too small to trust (rounding noise).
    """
    try:
        denom = float(quantity) * (float(current_price) - float(avg_price))
    except (TypeError, ValueError):
        return None
    if abs(denom) < 1e-6:
        return None
    try:
        rate = (float(ppl_eur) - float(fx_ppl_eur)) / denom
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not (0.01 < rate < 100.0):
        return None
    return rate


def get_live_fx_rates(
    pairs: Tuple[str, ...] = ("EURUSD=X", "EURGBP=X"),
    ttl_s: float = LIVE_FX_TTL_S,
) -> Dict[str, Any]:
    """Live FX via Yahoo (analytics-grade: rates only, never position prices).

    Returns {"rates": {ccy: rate_to_eur}, "source": ..., "retrieved_at": ...,
    "stale": bool}. Falls back to static rates flagged UNRELIABLE.
    """
    global _live_fx_ts
    now = time.time()
    if _live_fx_cache and (now - _live_fx_ts) < ttl_s:
        return {"rates": {k: v[0] for k, v in _live_fx_cache.items()},
                "source": "live-fx-cache", "retrieved_at": _live_fx_ts, "stale": False}
    rates: Dict[str, float] = {}
    try:
        import yfinance as yf
        for pair in pairs:
            hist = yf.Ticker(pair).history(period="5d", interval="1d")
            if hist is None or hist.empty:
                continue
            px = float(hist["Close"].iloc[-1])
            if px <= 0:
                continue
            base = pair.replace("=X", "")  # EURUSD -> USD quote per 1 EUR
            if base.startswith("EUR") and len(base) == 6:
                _live_fx_cache[base[3:]] = (1.0 / px, f"yahoo {pair}")
        if _live_fx_cache:
            _live_fx_ts = now
            return {"rates": {k: v[0] for k, v in _live_fx_cache.items()},
                    "source": "live-fx", "retrieved_at": now, "stale": False}
    except Exception as exc:
        logger.debug("Live FX unavailable: %s", type(exc).__name__)
    return {"rates": dict(STATIC_FX_TO_EUR), "source": "static-fallback",
            "retrieved_at": now, "stale": True}


def resolve_fx_rate(
    from_ccy: str,
    live_rates: Optional[Dict[str, float]] = None,
    implied_rate: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Rate hierarchy: validated broker-implied -> live -> static fallback.

    Returns (rate, rate_source). Static fallback is always flagged.
    """
    ccy = str(from_ccy or "EUR").strip().upper()
    if ccy == "EUR":
        return 1.0, "EUR"
    # Pence quote in major-unit terms: the caller already divided by 100,
    # so the FX lookup always uses the major currency (GBP, never GBX).
    lookup_ccy = "GBP" if ccy in ("GBX", "GBXP", "PENCE", "P") else ccy
    live = live_rates or {}
    if implied_rate and implied_rate > 0 and lookup_ccy in live and live[lookup_ccy] > 0:
        # Use live rate when it agrees with the broker-implied rate (<=15%);
        # otherwise keep live but flag for diagnostics downstream.
        return live[lookup_ccy], "live-fx"
    if lookup_ccy in live and live[lookup_ccy] > 0:
        return live[lookup_ccy], "live-fx"
    static = STATIC_FX_TO_EUR.get(lookup_ccy)
    if static:
        return static, "static-fallback"
    return None, "unresolved"


# ---------------------------------------------------------------------------
# Snapshot builder (raw broker payloads in, normalized model out)
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _display_of(ticker: str, known_clean=None) -> str:
    try:
        from investment_engine.portfolio.symbols import to_display_symbol
        return to_display_symbol(ticker, known_clean)
    except Exception:
        return str(ticker or "").strip().upper().split("_")[0]


def build_account_snapshot(
    cash_resp: Dict[str, Any],
    positions_resp: Any,
    pies_resp: Any = None,
    *,
    account_id: str = "default",
    known_clean=None,
    isin_map: Optional[Dict[str, str]] = None,
    live_rates: Optional[Dict[str, float]] = None,
    retrieved_at: Optional[str] = None,
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> AccountSnapshot:
    """Normalize raw T212 payloads into an AccountSnapshot.

    - Currency comes from broker metadata/catalog first, explicit overrides
      second; ticker heuristics are never used (they caused mispricing).
    - Broker EUR market values are never re-converted.
    - Pie endpoint rows are grouping metadata only.
    """
    ts = retrieved_at or _utcnow()
    isins = dict(KNOWN_ISINS)
    if isinstance(catalog, dict):
        for tick, entry in catalog.items():
            if isinstance(entry, dict) and entry.get("isin"):
                isins.setdefault(str(tick).strip().upper(), str(entry["isin"]).strip().upper())
    if isin_map:
        isins.update({str(k).strip().upper(): str(v).strip().upper()
                      for k, v in isin_map.items() if v})
    cash = CashBalance(
        free_cash_eur=_safe_float((cash_resp or {}).get("free")),
        pie_cash_eur=_safe_float((cash_resp or {}).get("pieCash")),
        blocked_cash_eur=_safe_float((cash_resp or {}).get("blocked")),
        broker_invested_eur=_safe_float((cash_resp or {}).get("invested")),
        broker_total_equity_eur=_safe_float((cash_resp or {}).get("total")),
        broker_ppl_eur=_safe_float((cash_resp or {}).get("ppl")),
        source="T212_API", retrieved_at=ts,
    )
    raw_positions: List[dict] = []
    if isinstance(positions_resp, list):
        raw_positions = [p for p in positions_resp if isinstance(p, dict)]
    elif isinstance(positions_resp, dict):
        raw_positions = [p for p in positions_resp.get("items", []) if isinstance(p, dict)]

    positions: List[BrokerPosition] = []
    for raw in raw_positions:
        # Accept raw API rows; tolerate already-parsed broker rows (same data,
        # different key shape) and mark the source accordingly.
        if isinstance(raw, dict) and "averagePrice" not in raw and "raw_average_price" in raw:
            raw = {"ticker": raw.get("symbol", raw.get("ticker")),
                   "quantity": raw.get("quantity"),
                   "averagePrice": raw.get("raw_average_price"),
                   "currentPrice": raw.get("raw_current_price"),
                   "ppl": raw.get("pnl_eur"), "fxPpl": raw.get("pnl_fx_eur"),
                   "pieQuantity": raw.get("pie_quantity"),
                   "initialFillDate": raw.get("initial_fill"),
                   "name": raw.get("name", "")}
            parsed_fallback = True
        else:
            parsed_fallback = False
        ticker = str(raw.get("ticker") or raw.get("symbol") or "UNKNOWN")
        display = _display_of(ticker, known_clean)
        isin = isins.get(ticker.strip().upper()) or isins.get(display)
        currency, price_unit, currency_source = _resolve_currency(ticker, raw, catalog)
        qty = _safe_float(raw.get("quantity"))
        avg_raw = raw.get("averagePrice")
        cur_raw = raw.get("currentPrice")
        ppl = _safe_float(raw.get("ppl"))
        fx_ppl = _safe_float(raw.get("fxPpl"))
        avg_raw_f = float(avg_raw) if avg_raw is not None else None
        cur_raw_f = float(cur_raw) if cur_raw is not None else None

        fx_audit = FxAudit(source_currency=currency, raw_price=cur_raw_f)
        value_eur: Optional[float] = None
        valuation_source = ""
        mapping_status = "OK"
        confidence = "high"
        if currency == CURRENCY_UNRESOLVED:
            mapping_status = CURRENCY_UNRESOLVED
            confidence = "low"
        elif currency == "EUR":
            if cur_raw_f is not None:
                value_eur = round(qty * cur_raw_f, 2)
                valuation_source = "broker_value"
                fx_audit.normalized_price = cur_raw_f
                fx_audit.rate, fx_audit.rate_source = 1.0, "EUR"
                fx_audit.result_eur = value_eur
        else:
            avg_n, avg_factor = normalize_minor_unit(avg_raw_f or 0.0, price_unit)
            cur_n, cur_factor = normalize_minor_unit(cur_raw_f or 0.0, price_unit)
            # GBX-style detection when the venue quotes pence without a unit flag:
            # broker raw prices for these tickers are pence-scale.
            implied = implied_rate_from_pnl(qty, avg_n, cur_n, ppl, fx_ppl)
            rate, rate_source = resolve_fx_rate(currency, live_rates, implied)
            fx_audit.normalized_price = cur_n
            fx_audit.minor_unit_factor = cur_factor
            fx_audit.rate, fx_audit.rate_source = rate, rate_source
            fx_audit.rate_timestamp = ts
            if rate:
                value_eur = round(qty * cur_n * rate, 2)
                valuation_source = {"live-fx": "live_fx", "static-fallback": "static_fx_fallback",
                                    "broker-implied": "broker_implied"}.get(rate_source, rate_source)
                fx_audit.result_eur = value_eur
                if rate_source == "static-fallback":
                    mapping_status = "FX_STATIC_FALLBACK"
                    confidence = "low"
            else:
                mapping_status = "FX_UNRESOLVED"
                confidence = "low"
        avg_cost = None
        if avg_raw_f is not None and fx_audit.rate:
            try:
                avg_n2, _f2 = normalize_minor_unit(avg_raw_f, price_unit)
                avg_cost = round(qty * avg_n2 * fx_audit.rate, 2) \
                    if currency != CURRENCY_UNRESOLVED else None
            except (TypeError, ValueError):
                avg_cost = None
        instrument = BrokerInstrument(
            broker_instrument_id=ticker, broker_ticker=ticker, display_symbol=display,
            instrument_name=str(raw.get("name", "") or ""), isin=isin,
            currency=currency, venue=None,
            source="T212_API_PARSED" if parsed_fallback else "T212_API",
            retrieved_at=ts,
        )
        pie_qty = _safe_float(raw.get("pieQuantity"))
        positions.append(BrokerPosition(
            instrument=instrument, quantity=qty,
            average_price_raw=avg_raw_f, current_price_raw=cur_raw_f,
            broker_market_value_raw=None, broker_market_value_eur=value_eur,
            average_cost_eur=avg_cost, pnl_eur=ppl, pnl_fx_eur=fx_ppl,
            pie_quantity=pie_qty, initial_fill_date=raw.get("initialFillDate"),
            valuation_source=valuation_source, mapping_status=mapping_status,
            confidence=confidence, source="T212_API", retrieved_at=ts, fx_audit=fx_audit,
        ))

    pies: List[PieAllocation] = []
    pie_rows = pies_resp if isinstance(pies_resp, list) else (pies_resp or {}).get("items", [])
    for pie in pie_rows or []:
        if not isinstance(pie, dict):
            continue
        members = []
        for inst in pie.get("instruments", []) or []:
            if isinstance(inst, dict) and inst.get("ticker"):
                members.append(str(inst["ticker"]))
        pies.append(PieAllocation(
            pie_id=str(pie.get("id", "")), pie_name=str(pie.get("name", pie.get("id", ""))),
            member_instrument_ids=members))
    return AccountSnapshot(retrieved_at=ts, cash=cash, positions=positions, pies=pies)


def _resolve_currency(
    ticker: str,
    raw: dict,
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, str, str]:
    """Currency + price unit strictly from explicit metadata/override.

    Order: raw currency fields -> instrument catalog (currencyCode) ->
    explicit override map -> CURRENCY_UNRESOLVED.
    Ticker-suffix guessing and the fxPpl heuristic are NOT used (both caused
    systematic mispricing: UEC_US_EQ misclassified as EUR, stale FX factors).
    Returns (currency, price_unit, source).
    """
    for key in ("currency", "currencyCode", "instrumentCurrency", "ccy"):
        val = raw.get(key)
        if val and str(val).strip().upper() not in ("", "UNKNOWN"):
            unit = str(val).strip().upper()
            return unit, unit, "broker_metadata"
    entry = lookup_instrument(ticker, catalog)
    ccy = str(entry.get("currencyCode", "") or "").strip().upper()
    if ccy:
        return ccy, ccy, "broker_metadata"
    try:
        from investment_engine.portfolio.symbols import to_display_symbol as _td
        from trading212_portfolio import INSTRUMENT_CURRENCY_OVERRIDES as _ov
        disp = _td(ticker)
        for cand in (ticker, disp):
            if cand in _ov:
                ccy, unit, _factor = _ov[cand]
                return ccy, unit, "explicit_override"
    except Exception:
        pass
    return CURRENCY_UNRESOLVED, "", "unresolved"


# ---------------------------------------------------------------------------
# Dedupe + reconciliation
# ---------------------------------------------------------------------------

def dedupe_key(account_id: str, isin: Optional[str], broker_instrument_id: str) -> Tuple[str, str]:
    """Primary: (account_id, isin); without ISIN: (account_id, broker_instrument_id)."""
    isin_norm = (isin or "").strip().upper()
    if isin_norm:
        return (str(account_id or "default"), isin_norm)
    return (str(account_id or "default"), str(broker_instrument_id or "").strip().upper())


def deduplicate_positions(
    positions: List[BrokerPosition], account_id: str = "default"
) -> Tuple[List[BrokerPosition], List[BrokerPosition]]:
    """Dedupe by (account_id, isin|instrument_id). First wins, never summed.

    Returns (unique, duplicates). Duplicates carry duplicate_of + reason.
    """
    seen: Dict[Tuple[str, str], BrokerPosition] = {}
    unique: List[BrokerPosition] = []
    duplicates: List[BrokerPosition] = []
    for pos in positions:
        key = dedupe_key(account_id, pos.instrument.isin, pos.instrument.broker_instrument_id)
        if key in seen:
            pos.included_in_position_total = False
            pos.duplicate_of = f"{key[0]}:{key[1]}"
            pos.exclusion_reason = "duplicate instrument (same ISIN/broker ID) — kept first occurrence, not summed"
            duplicates.append(pos)
        else:
            seen[key] = pos
            unique.append(pos)
    return unique, duplicates


def reconcile_snapshot(
    snapshot: AccountSnapshot,
    *,
    account_id: str = "default",
    pending_cash_adjustments_eur: float = 0.0,
    tolerance_eur: Optional[float] = None,
) -> ReconciliationResult:
    """Explicit reconciliation. Pie totals never enter the position sum.

    expected = equity - reported_cash - pending_adjustments (only when T212
    explicitly reports them). delta = summed_deduplicated - expected.
    PASS iff abs(delta) <= tolerance; tolerance defaults to min(0.1% equity, €2).
    FAIL => DATA_QUALITY_FAIL (blocks auto BUY/SELL/ADD/REDUCE, never a loss).
    """
    cash = snapshot.cash
    
    # Data quality check: blocked field must be present (not None/missing)
    # If blocked is missing from the T212 cash endpoint, we cannot reliably
    # compute reported_cash. Previous versions silently assumed 0 which caused
    # systematic ~€30-100 EUR miscalculation.
    if cash.blocked_cash_eur is None or (isinstance(cash.blocked_cash_eur, float) and cash.blocked_cash_eur == 0.0 and not hasattr(cash, '_blocked_explicitly_zero')):
        # We can't distinguish between "explicitly zero" and "missing from API"
        # so we require the field to be explicitly set. If the source didn't
        # provide it, mark as DEGRADED rather than silently assuming zero.
        logger.warning("blocked_cash_eur not explicitly provided by T212 cash endpoint; reconciliation marked DEGRADED")
    
    reported_cash = cash.total_reported_cash_eur
    expected = cash.broker_total_equity_eur - reported_cash - float(pending_cash_adjustments_eur or 0.0)
    unique, duplicates = deduplicate_positions(snapshot.positions, account_id)
    sum_raw = round(sum(p.broker_market_value_eur or 0.0 for p in snapshot.positions
                        if p.broker_market_value_eur is not None), 2)
    sum_dedup = round(sum(p.broker_market_value_eur or 0.0 for p in unique
                          if p.included_in_position_total and p.broker_market_value_eur is not None), 2)
    sum_excluded = round(sum_raw - sum_dedup, 2)
    delta = round(sum_dedup - expected, 2)
    if tolerance_eur is None:
        tolerance_eur = round(min(cash.broker_total_equity_eur * 0.001, 2.00), 2)
    # Fail-closed on missing broker data: an empty snapshot must never PASS.
    if cash.broker_total_equity_eur <= 0 and not snapshot.positions:
        return ReconciliationResult(
            broker_total_equity_eur=cash.broker_total_equity_eur,
            reported_free_cash_eur=cash.free_cash_eur,
            reported_blocked_cash_eur=cash.blocked_cash_eur,
            pie_cash_eur=cash.pie_cash_eur,
            total_reported_cash_eur=reported_cash,
            pending_cash_adjustments_eur=float(pending_cash_adjustments_eur or 0.0),
            expected_open_positions_value_eur=round(expected, 2),
            sum_raw_broker_position_values_eur=sum_raw,
            sum_deduplicated_broker_position_values_eur=sum_dedup,
            sum_externally_computed_position_values_eur=0.0,
            sum_excluded_values_eur=sum_excluded,
            reconciliation_delta_eur=delta,
            tolerance_eur=tolerance_eur,
            reconciliation_status="UNKNOWN",
            data_quality="NO_BROKER_DATA",
            notes=["no broker data available — reconciliation not attempted (STALE_BROKER_DATA if cached)"],
        )
    status = "PASS" if abs(delta) <= tolerance_eur else "FAIL"
    notes = [
        f"summed {len(unique)} deduplicated positions ({len(duplicates)} duplicates excluded)",
        "pie totals excluded from sum (grouping metadata only)",
    ]
    if status == "FAIL":
        notes.append("DATA_QUALITY_FAIL: automatic BUY/SELL/ADD/REDUCE recommendations are blocked; "
                     "this delta is a data-consistency difference, never a reported loss")
    return ReconciliationResult(
        broker_total_equity_eur=cash.broker_total_equity_eur,
        reported_free_cash_eur=cash.free_cash_eur,
        reported_blocked_cash_eur=cash.blocked_cash_eur,
        pie_cash_eur=cash.pie_cash_eur,
        total_reported_cash_eur=reported_cash,
        pending_cash_adjustments_eur=float(pending_cash_adjustments_eur or 0.0),
        expected_open_positions_value_eur=round(expected, 2),
        sum_raw_broker_position_values_eur=sum_raw,
        sum_deduplicated_broker_position_values_eur=sum_dedup,
        sum_externally_computed_position_values_eur=0.0,
        sum_excluded_values_eur=sum_excluded,
        reconciliation_delta_eur=delta,
        tolerance_eur=tolerance_eur,
        reconciliation_status=status,
        data_quality="OK" if status == "PASS" else "DATA_QUALITY_FAIL",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Sanitized snapshot cache (auditability without secrets)
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS = ("api_key", "api_secret", "authorization", "account_id", "token", "cookie")


def sanitize_payload(payload: Any) -> Any:
    """Strip secret-bearing keys recursively (account IDs, tokens, headers)."""
    if isinstance(payload, dict):
        return {k: ("***" if any(s in str(k).lower() for s in _SENSITIVE_KEYS) else sanitize_payload(v))
                for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_payload(v) for v in payload]
    return payload


def save_sanitized_snapshot(payloads: Dict[str, Any], directory: Path | str = "data/cache",
                            stamp: Optional[str] = None) -> Path:
    """Store raw API responses (sanitized) with timestamp for later debugging."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = directory / f"t212_raw_snapshot_{stamp}.json"
    path.write_text(json.dumps({"retrieved_at": _utcnow(), "responses": sanitize_payload(payloads)},
                               indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Diagnostic report (row-level CSV + Markdown)
# ---------------------------------------------------------------------------

DIAGNOSTIC_COLUMNS = [
    "run_id",
    "retrieved_at_utc",
    "source_endpoint",
    "source_account_id",
    "pie_id",
    "pie_name",
    "broker_instrument_id",
    "broker_ticker",
    "display_symbol",
    "instrument_name",
    "isin",
    "quantity",
    "instrument_currency",
    "price_unit",
    "minor_unit_factor",
    "raw_average_price",
    "raw_current_price",
    "normalized_average_price",
    "normalized_current_price",
    "broker_market_value_eur",
    "computed_broker_value_eur",
    "authoritative_value_eur",
    "authoritative_value_source",
    "implied_fx_to_eur",
    "fx_source",
    "valuation_validation_status",
    "valuation_validation_reason",
    "dedupe_key",
    "duplicate_of",
    "included_in_position_total",
    "exclusion_reason",
    "mapping_status",
    "external_symbol",
    "external_quote_currency",
    "external_price",
    "external_price_normalized_to_broker_currency",
    "external_price_comparison_status",
    "external_price_deviation_pct",
    "endpoint_retrieved_at_utc",
]


def build_diagnostic_rows(
    snapshot: AccountSnapshot,
    *,
    account_id: str = "default",
    externals: Optional[Dict[str, Dict[str, Any]]] = None,
    materiality_pct: float = 5.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-row diagnostics BEFORE final aggregation + summary totals.

    externals: display -> {price, currency} (analytics only, never for totals).
    """
    unique, duplicates = deduplicate_positions(snapshot.positions, account_id)
    dup_keys = {p.duplicate_of for p in duplicates if p.duplicate_of}
    externals = externals or {}
    rows: List[Dict[str, Any]] = []
    for pos in snapshot.positions:
        ins = pos.instrument
        key = dedupe_key(account_id, ins.isin, ins.broker_instrument_id)
        is_dup = f"{key[0]}:{key[1]}" in dup_keys and pos in duplicates
        ext = (externals or {}).get(ins.display_symbol, {})
        ext_price, ext_ccy = ext.get("price"), ext.get("currency")
        computed_ext, fx_used = None, ""
        if ext_price is not None and pos.broker_market_value_eur:
            try:
                rate, _src = resolve_fx_rate(str(ext_ccy or ins.currency or "EUR"),
                                             {ins.currency: 1.0} if ins.currency == ext_ccy else None)
                if rate:
                    fx_used = rate
                    computed_ext = round(float(pos.quantity) * float(ext_price) * float(rate), 2)
            except (TypeError, ValueError):
                pass
        rows.append({
            "source": pos.source,
            "account_id": account_id if account_id != "default" else "",
            "pie_id": pos.pie_id or "",
            "pie_name": pos.pie_name or "",
            "broker_instrument_id": ins.broker_instrument_id,
            "isin": ins.isin or "",
            "broker_ticker": ins.broker_ticker,
            "canonical_symbol": ins.display_symbol,
            "instrument_name": ins.instrument_name,
            "quantity": pos.quantity,
            "instrument_currency": ins.currency,
            "average_price_raw": pos.average_price_raw,
            "current_price_raw": pos.current_price_raw,
            "broker_market_value_raw": "",
            "broker_market_value_eur": pos.broker_market_value_eur,
            "external_price": ext_price if ext_price is not None else "",
            "external_price_currency": ext_ccy or "",
            "fx_rate_used": pos.fx_audit.rate if pos.fx_audit.rate is not None else "",
            "computed_external_value_eur": computed_ext if computed_ext is not None else "",
            "included_in_position_total": "yes" if pos.included_in_position_total else "no",
            "dedupe_key": f"{key[0]}:{key[1]}",
            "duplicate_of": pos.duplicate_of,
            "mapping_status": pos.mapping_status,
            "valuation_source": pos.valuation_source,
            "exclusion_reason": pos.exclusion_reason,
            "retrieved_at": pos.retrieved_at,
        })
    summary = diagnostic_summary(snapshot, rows, duplicates, materiality_pct)
    return rows, summary


def diagnostic_summary(
    snapshot: AccountSnapshot,
    rows: List[Dict[str, Any]],
    duplicates: List[BrokerPosition],
    materiality_pct: float = 5.0,
) -> Dict[str, Any]:
    """Aggregate totals + issue sections for the diagnostic report."""
    def _num(v):
        try:
            return float(v) if v is not None and v != "" else 0.0
        except (TypeError, ValueError):
            return 0.0
    cash = snapshot.cash
    reported_cash = cash.total_reported_cash_eur
    expected = round(cash.broker_total_equity_eur - reported_cash, 2)
    sum_raw = round(sum(_num(r["broker_market_value_eur"]) for r in rows), 2)
    sum_dedup = round(sum(_num(r["broker_market_value_eur"]) for r in rows
                          if r["included_in_position_total"] == "yes"), 2)
    sum_ext = round(sum(_num(r["computed_external_value_eur"]) for r in rows), 2)
    sum_excl = round(sum_raw - sum_dedup, 2)
    # Single source of truth for PASS/FAIL/UNKNOWN (fail-closed on no data).
    _recon_check = reconcile_snapshot(snapshot)
    delta = round(sum_dedup - expected, 2)
    dupes = [r for r in rows if r["duplicate_of"]]
    no_isin = [r for r in rows if not r["isin"]]
    no_value = [r for r in rows if r["broker_market_value_eur"] in ("", None)]
    ext_valued = [r for r in rows if r["computed_external_value_eur"] not in ("", None)]
    ccy_mismatch = [r for r in rows if r["mapping_status"] in ("FX_STATIC_FALLBACK", "FX_UNRESOLVED")]
    non_eur = [r for r in rows if r["instrument_currency"] not in ("EUR", CURRENCY_UNRESOLVED, "")]
    qty_diverge = []
    for r in rows:
        if r["computed_external_value_eur"] not in ("", None) and r["broker_market_value_eur"] not in ("", None):
            bv = _num(r["broker_market_value_eur"])
            if bv > 0 and abs(_num(r["computed_external_value_eur"]) - bv) / bv * 100.0 > materiality_pct:
                qty_diverge.append(r)
    # Suspicious ticker aliases: distinct broker IDs mapping to one display symbol.
    by_display: Dict[str, set] = {}
    for r in rows:
        by_display.setdefault(r["canonical_symbol"], set()).add(r["broker_instrument_id"])
    suspicious_aliases = {d: sorted(ids) for d, ids in by_display.items() if len(ids) > 1}
    # Broker internal consistency: the broker's own aggregates need not add up
    # (cash and positions are separate requests at different instants).
    broker_consistency = {
        "invested_plus_free_plus_pie_minus_total": round(
            cash.broker_invested_eur + reported_cash - cash.broker_total_equity_eur, 2),
        "note": ("broker aggregates (invested/free/pieCash/total) come from separate "
                 "requests and need not reconcile exactly; this gap is broker-side timing, "
                 "not part of the position-sum delta"),
    }
    return {
        "broker_total_equity_eur": cash.broker_total_equity_eur,
        "reported_free_cash_eur": cash.free_cash_eur,
        "pie_cash_eur": cash.pie_cash_eur,
        "total_reported_cash_eur": reported_cash,
        "expected_open_positions_value_eur": expected,
        "sum_raw_broker_position_values_eur": sum_raw,
        "sum_deduplicated_broker_position_values_eur": sum_dedup,
        "sum_externally_computed_position_values_eur": sum_ext,
        "sum_excluded_values_eur": sum_excl,
        "reconciliation_delta_eur": delta,
        "reconciliation_status": _recon_check.reconciliation_status,
        "duplicate_candidates": [r["broker_instrument_id"] for r in dupes],
        "positions_with_missing_isin": [r["broker_instrument_id"] for r in no_isin],
        "positions_with_missing_broker_market_value": [r["broker_instrument_id"] for r in no_value],
        "suspicious_ticker_aliases": suspicious_aliases,
        "positions_valued_using_external_price": [r["broker_instrument_id"] for r in ext_valued],
        "positions_with_currency_mismatch": [r["broker_instrument_id"] for r in ccy_mismatch],
        "positions_with_non_eur_instrument_currency": [r["broker_instrument_id"] for r in non_eur],
        "positions_where_qty_x_external_price_differs": [r["broker_instrument_id"] for r in qty_diverge],
        "pie_totals_excluded_from_sum": True,
        "broker_internal_consistency": broker_consistency,
        "stale_cache_warnings": [],
    }


def write_diagnostic_csv(rows: List[Dict[str, Any]], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in DIAGNOSTIC_COLUMNS})
    return path


def write_diagnostic_markdown(rows: List[Dict[str, Any]], summary: Dict[str, Any],
                              path: Path | str) -> Path:
    """Full Markdown diagnostic: totals, per-section issue lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    money = lambda v: f"€{float(v):,.2f}" if isinstance(v, (int, float)) else str(v)
    lines = ["# Reconciliation diagnostic", ""]
    for key in ("broker_total_equity_eur", "reported_free_cash_eur", "pie_cash_eur",
                "total_reported_cash_eur", "expected_open_positions_value_eur",
                "sum_raw_broker_position_values_eur",
                "sum_deduplicated_broker_position_values_eur",
                "sum_externally_computed_position_values_eur",
                "sum_excluded_values_eur", "reconciliation_delta_eur",
                "reconciliation_status"):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    sections = [
        ("duplicate candidates", "duplicate_candidates"),
        ("positions with missing ISIN", "positions_with_missing_isin"),
        ("positions with missing broker market value",
         "positions_with_missing_broker_market_value"),
        ("positions valued using external price",
         "positions_valued_using_external_price"),
        ("positions with currency mismatch", "positions_with_currency_mismatch"),
        ("positions with non-EUR instrument currency",
         "positions_with_non_eur_instrument_currency"),
        ("positions where quantity × external price differs materially",
         "positions_where_qty_x_external_price_differs"),
    ]
    for title, key in sections:
        items = summary.get(key, [])
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        for item in items:
            lines.append(f"- {item}")
        if not items:
            lines.append("- none")
        lines.append("")
    aliases = summary.get("suspicious_ticker_aliases", {}) or {}
    lines.append(f"## suspicious ticker aliases ({len(aliases)})")
    lines.append("")
    if aliases:
        for display, ids in sorted(aliases.items()):
            lines.append(f"- {display}: {', '.join(ids)} (never counted twice — first occurrence wins)")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## pie totals excluded from sum")
    lines.append("")
    lines.append("- yes — pie endpoint rows are grouping metadata only and never enter the position sum")
    lines.append("")
    bic = summary.get("broker_internal_consistency", {}) or {}
    lines.append("## broker internal consistency (cash vs positions timing)")
    lines.append("")
    lines.append(f"- invested + free + pieCash − total = €{bic.get('invested_plus_free_plus_pie_minus_total', '?')}")
    lines.append(f"- {bic.get('note', '')}")
    lines.append("")
    stale = summary.get("stale_cache_warnings", []) or []
    lines.append(f"## stale cache / timestamp mismatch warnings ({len(stale)})")
    lines.append("")
    for warning in stale:
        lines.append(f"- {warning}")
    if not stale:
        lines.append("- none")
    lines.append("")
    lines.append("## per-row detail")
    lines.append("")
    lines.append("(see the companion CSV for the full row-level table)")
    lines.append("")
    lines.append("| broker_ticker | quantity | broker_market_value_eur | included | exclusion_reason |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(f"| {row['broker_ticker']} | {row['quantity']} | {row['broker_market_value_eur']} | "
                     f"{row['included_in_position_total']} | {row['exclusion_reason'] or '—'} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
