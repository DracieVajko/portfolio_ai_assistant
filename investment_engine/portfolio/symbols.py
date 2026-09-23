"""Broker / display / Yahoo symbol normalization (single mapping layer).

Trading212 internal IDs look like ``AAPL_US_EQ`` or ``VWSBd_EQ`` (the trailing
lowercase letter before ``_EQ`` is the T212 exchange code). Reports must render
clean display tickers (``AAPL``, ``VWSB``) and real company names - never the
internal ID. Yahoo Finance must be queried with valid Yahoo tickers, never
with company names.

Single alias precedence (config-over-static):
  ``resolve_alias()`` is the ONLY place where config ``symbol_aliases`` and
  static ``YAHOO_OVERRIDES`` are merged. Config entries (upper-normalized)
  always win over static entries. ``to_yahoo_symbol()`` delegates to it.

Exchange-suffix proposals (validation-gated):
  ``propose_yahoo_candidates()`` derives ranked EU candidates from the T212
  trailing exchange letter (d/l/p/e -> .DE/.L/.PA/.MC) plus catalog
  ``currencyCode`` / ``venue`` / ``isin`` country. Candidates are PROPOSALS
  only - a candidate enters ``SUPPORTED_YAHOO`` (i.e. may be fetched) only
  after out-of-band validation (earnings/price probe). Call sites must log
  the returned source label via ``logger`` for provenance.

US-pattern restriction:
  A bare ``^[A-Z]{1,5}(-[A-Z])?$`` ticker is SUPPORTED only when explicitly
  verified (in ``SUPPORTED_YAHOO`` / ``verified_symbols``) or when catalog /
  ISIN confirms a US venue (ISIN ``US*``, currency ``USD``, venue NYSE /
  NASDAQ / OTC). Otherwise UNRESOLVED - this prevents wrong-venue fetches
  such as SU / AI / ENR (Paris / Xetra) being queried as US stocks.

All helpers are deterministic and dependency-free (no network, no yfinance)
so they can be unit-tested offline.
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
UNKNOWN_INSTRUMENT = "Unknown instrument"
COMPANY_OVERRIDES: Dict[str, str] = {
    "TTWO": "Take-Two Interactive",
    "TUYA": "Tuya Inc",
    "ASGN": "ASGN Incorporated",
    "SYN": "Synergia Energy",
    "SYNL": "Synergia Energy",
    "EGT": "European Green Transition",
    "DJGTEEX": "iShares Dow Jones Global Titans 50 UCITS ETF",
    "DJGTEEXD": "iShares Dow Jones Global Titans 50 UCITS ETF",
}
DISPLAY_OVERRIDES: Dict[str, str] = {
    "DJGTEEXD": "DJGTEEX",
}
YAHOO_OVERRIDES: Dict[str, str] = {
    "VRT": "VRT",
    "QCOM": "QCOM",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "INTC": "INTC",
    "AVGO": "AVGO",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
    "KAS-USD": "KAS-USD",
    "SUI-USD": "SUI-USD",
    "TAO-USD": "TAO-USD",
    "RENDER-USD": "RENDER-USD",
    "VERTIV": "VRT",
    "QUALCOMM": "QCOM",
    "NVIDIA": "NVDA",
    "INTE": "INTC",
    "BROADCOM": "AVGO",
    "APPL": "AAPL",
    "IBE": "IBE.MC",
    "RWE": "RWE.DE",
    "VWSB": "VWS.CO",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "KASUSDT": "KAS-USD",
    "SUIUSDT": "SUI-USD",
    "RNDRUSDT": "RENDER-USD",
    "TAOUSDT": "TAO-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "TSLA": "TSLA",
    "STLA": "STLA",
    "SLDP": "SLDP",
    "BYND": "BYND",
    "BYDDY": "BYDDY",
    "EGT": "EGT.L",
    "LITM": "LIT",
    "DDD": "DDD",
    "000660": "000660.KS",
}
NO_EARNINGS_YAHOO: frozenset = frozenset({
    "BTC-USD", "ETH-USD", "SOL-USD", "KAS-USD", "SUI-USD", "RENDER-USD", "TAO-USD",
    "XEON", "CSH2", "ERNX",
})
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
UNRESOLVED = "UNRESOLVED"
FETCH_FAILED = "FETCH_FAILED"
SUPPORTED_YAHOO: frozenset = frozenset({
    "VRT", "QCOM", "AAPL", "NVDA", "INTC", "AVGO",
    "BTC-USD", "ETH-USD", "SOL-USD", "KAS-USD", "RENDER-USD",
    "SUI-USD", "TAO-USD",
    "NEE", "MSFT", "AMD", "TTWO", "TUYA", "ASML", "IBM", "TSM",
    "ARM", "BMY", "CF", "MU", "CSCO", "WDC", "SNDK", "MRVL",
    "SYY", "LTC", "AFL", "RY", "TD", "BMO",
    "C7A0.F", "VWS.CO", "IBE.MC", "RWE.DE",
    "TSLA", "STLA", "SLDP", "BYND", "BYDDY", "EGT.L", "LIT", "DDD",
    "000660.KS",
    "C", "CVX", "JNJ", "JPM", "WMT",
})
MAPPING_UNAVAILABLE_NOTE = "Market-data mapping unavailable; broker valuation retained."
def _normalize_alias_key(key: Any) -> str:
    return str(key or "").strip().upper()
def resolve_alias(key: Any, aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve one key to a Yahoo ticker with config-over-static precedence.
    Config ``symbol_aliases`` (keys upper-normalized) always win over static
    ``YAHOO_OVERRIDES``. Returns the mapped value (stripped) or None.
    Deterministic, network-free. Use ``resolve_alias_with_source()`` when
    provenance logging is needed.
    """
    k = _normalize_alias_key(key)
    if not k:
        return None
    if aliases:
        for ak, av in aliases.items():
            if _normalize_alias_key(ak) == k:
                v = str(av or "").strip()
                if v:
                    logger.debug("resolve_alias %r -> %r via config", k, v)
                    return v
                break
    v = YAHOO_OVERRIDES.get(k)
    if v is not None:
        logger.debug("resolve_alias %r -> %r via static", k, v)
    return v
def resolve_alias_with_source(key: Any, aliases: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
    """Same as ``resolve_alias`` but also returns the provenance label.
    Returns ``(value_or_None, source)`` where source is config/static/none.
    """
    k = _normalize_alias_key(key)
    if not k:
        return None, "none"
    if aliases:
        for ak, av in aliases.items():
            if _normalize_alias_key(ak) == k:
                v = str(av or "").strip()
                if v:
                    return v, "config"
                break
    v = YAHOO_OVERRIDES.get(k)
    if v is not None:
        return v, "static"
    return None, "none"
EXCHANGE_SUFFIX_MAP: Dict[str, str] = {
    "d": ".DE",
    "l": ".L",
    "p": ".PA",
    "e": ".MC",
}
CURRENCY_SUFFIX_MAP: Dict[str, Optional[str]] = {
    "GBX": ".L",
    "GBP": ".L",
    "GBp": ".L",
    "P": ".L",
    "PENCE": ".L",
    "EUR": None,
    "USD": "",
    "DKK": ".CO",
    "SEK": ".ST",
    "CHF": ".SW",
}
ISIN_COUNTRY_SUFFIX_MAP: Dict[str, str] = {
    "FR": ".PA",
    "DE": ".DE",
    "ES": ".MC",
    "IT": ".MI",
    "NL": ".AS",
    "DK": ".CO",
    "GB": ".L",
    "IE": ".L",
    "US": "",
    "CA": ".TO",
    "CH": ".SW",
    "SE": ".ST",
}
def _extract_exchange_letter(internal_id: str) -> Optional[str]:
    s = (internal_id or "").strip()
    if not s:
        return None
    up = s.upper()
    stem = None
    for suffix in _INTERNAL_SUFFIXES:
        if up.endswith(suffix):
            stem = s[: len(s) - len(suffix)]
            break
    if stem is None:
        return None
    if len(stem) >= 1 and stem[-1].isalpha() and stem[-1].islower():
        return stem[-1].lower()
    return None
def _is_us_venue(isin: Optional[str] = None, currency: Optional[str] = None, venue: Optional[str] = None) -> bool:
    if isin and str(isin).strip().upper().startswith("US"):
        return True
    if currency and str(currency).strip().upper() in ("USD", "US", "US$"):
        return True
    if venue:
        v = str(venue).strip().upper()
        if v in ("US", "USA", "NYSE", "NASDAQ", "OTC", "XNAS", "XNYS", "ARCX", "BATS", "NYSEARCA", "NASDAQGS", "NASDAQGM", "Nyse", "Nasdaq"):
            return True
        for token in ("NYSE", "NASDAQ", "OTC", "XNAS", "XNYS", "ARCX"):
            if token in v:
                return True
    return False
def propose_yahoo_candidates(internal_id: str, display_symbol: str = "", catalog: Optional[Dict[str, Any]] = None, aliases: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """Propose ranked Yahoo candidates (deterministic, network-free).
    Sources (ranked): config > static > exchange_suffix > currency > isin_country.
    Returns a deduplicated list of ``(candidate, source_label)`` in rank order.
    A candidate must pass out-of-band validation before entering SUPPORTED_YAHOO;
    this function never mutates the verified sets and never performs network I/O.
    Call sites must log the source label via ``logger`` for provenance.
    """
    disp = (display_symbol or "").strip().upper() or to_display_symbol(internal_id)
    disp = disp.split("_")[0] if disp else ""
    out: List[Tuple[str, str]] = []
    seen: Dict[str, int] = {}
    def _add(cand: str, source: str) -> None:
        c = (cand or "").strip()
        if not c or c.upper() == "UNKNOWN":
            return
        if c in seen:
            idx = seen[c]
            prev_c, prev_s = out[idx]
            if source not in prev_s:
                out[idx] = (prev_c, prev_s + " + " + source)
                logger.debug("propose_yahoo %r -> %r additional via %s", internal_id, c, source)
            return
        seen[c] = len(out)
        out.append((c, source))
        logger.debug("propose_yahoo %r -> %r via %s", internal_id, c, source)
    for key in ((internal_id or "").strip().upper(), _internal_stem((internal_id or "").strip().upper()), disp):
        if key:
            val, src = resolve_alias_with_source(key, aliases)
            if val:
                _add(val, src)
    letter = _extract_exchange_letter(internal_id or "")
    if letter and letter in EXCHANGE_SUFFIX_MAP and disp:
        _add(f"{disp}{EXCHANGE_SUFFIX_MAP[letter]}", f"exchange_suffix:{letter}->{EXCHANGE_SUFFIX_MAP[letter]}")
    cat = catalog or {}
    ccy = str(cat.get("currencyCode", "") or cat.get("currency", "") or "").strip().upper()
    if ccy and disp:
        suffix = CURRENCY_SUFFIX_MAP.get(ccy)
        if suffix:
            _add(f"{disp}{suffix}", f"currency:{ccy}->{suffix}")
        elif suffix == "":
            _add(disp, f"currency:{ccy}->US")
    isin = str(cat.get("isin", "") or "").strip().upper()
    venue = str(cat.get("venue", "") or cat.get("exchange", "") or cat.get("market", "") or "").strip()
    if isin and len(isin) >= 2 and disp:
        cc = isin[:2]
        suffix = ISIN_COUNTRY_SUFFIX_MAP.get(cc)
        if suffix:
            _add(f"{disp}{suffix}", f"isin_country:{cc}->{suffix}")
        elif suffix == "":
            _add(disp, f"isin_country:{cc}->US")
    if venue and disp:
        vup = venue.strip().upper()
        mapped = None
        if "PARIS" in vup or vup in ("PA", "XPAR"):
            mapped = ".PA"
        elif "XETRA" in vup or "FRANKFURT" in vup or vup in ("DE", "XFRA"):
            mapped = ".DE"
        elif "LONDON" in vup or vup in ("L", "XLON"):
            mapped = ".L"
        elif "MADRID" in vup or vup in ("MC", "XMAD"):
            mapped = ".MC"
        elif "COPENHAGEN" in vup or vup in ("CO", "XCSE"):
            mapped = ".CO"
        if mapped:
            _add(f"{disp}{mapped}", f"venue:{venue}->{mapped}")
    return out
def support_state(yahoo_symbol: Optional[str], purpose: str = "market_data", venue: Optional[str] = None, isin: Optional[str] = None, currency: Optional[str] = None, verified_symbols: Optional[Any] = None) -> str:
    """Classify a Yahoo ticker: SUPPORTED / UNSUPPORTED / UNRESOLVED.
    purpose=earnings additionally excludes instruments without an earnings
    calendar (crypto, cash-like ETFs).
    US-pattern tickers (``^[A-Z]{1,5}(-[A-Z])?$``) are SUPPORTED only when
    explicitly verified (in ``SUPPORTED_YAHOO`` or ``verified_symbols``) or
    when ``isin`` / ``currency`` / ``venue`` confirms a US venue (ISIN ``US*``,
    currency ``USD``, venue NYSE/NASDAQ/OTC). Otherwise UNRESOLVED. Explicit
    ``UNRESOLVED_YAHOO`` entries (SU/AI/ENR, ...) always stay UNRESOLVED.
    Dotted EU forms (``*.DE``/``*.L``/...) require explicit SUPPORTED entry;
    venue hints alone never auto-verify them (validation-gated).
    """
    global _US_STOCK_RE
    if not yahoo_symbol:
        return UNSUPPORTED
    y = str(yahoo_symbol).strip()
    if not y or y.upper() == "UNKNOWN":
        return UNSUPPORTED
    if y.upper() in ("XEON", "CSH2", "ERNX"):
        return UNSUPPORTED
    if purpose == "earnings" and y in NO_EARNINGS_YAHOO:
        return UNSUPPORTED
    if y in SUPPORTED_YAHOO:
        return SUPPORTED
    if y.upper() in UNRESOLVED_YAHOO:
        return UNRESOLVED
    if verified_symbols:
        try:
            vset = {str(v).strip() for v in verified_symbols}
            vup = {str(v).strip().upper() for v in verified_symbols}
            if y in vset or y.upper() in vup:
                return SUPPORTED
        except TypeError:
            pass
    if _US_STOCK_RE is None:
        import re as _re_us
        _US_STOCK_RE = _re_us.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
    if _US_STOCK_RE.match(y):
        if _is_us_venue(isin=isin, currency=currency, venue=venue):
            logger.debug("support_state %r SUPPORTED via US venue (isin=%r currency=%r venue=%r)", y, isin, currency, venue)
            return SUPPORTED
        return UNRESOLVED
    return UNRESOLVED
UNRESOLVED_YAHOO: frozenset = frozenset({
    "EGTL", "SYNL", "SYN", "SYN.L", "IQQH", "HY9H",
    "C7A0", "VWSB", "SU", "AI", "ENR",
})
_US_STOCK_RE = None
_INTERNAL_SUFFIXES = ("_US_EQ", "_DE_EQ", "_IM_EQ", "_L_EQ", "_D_EQ", "_EQ")
def _internal_stem(internal_key: str) -> str:
    stem = internal_key or ""
    for suffix in _INTERNAL_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem
def to_display_symbol(internal_id: str, known_clean: Optional[set] = None) -> str:
    """Map a T212 internal ID to a clean display ticker.
    ``AAPL_US_EQ`` -> ``AAPL``; ``VWSBd_EQ`` -> ``VWSB``; ``SUp_EQ`` -> ``SU``.
    A trailing lowercase letter is an exchange code when the input carried an
    ``_EQ``-family suffix, so it is stripped unconditionally in that case
    (fixes VWSBD vs VWSB divergence without needing ``known_clean``).
    Without an ``_EQ`` suffix the strip applies only when the remainder is a
    known clean ticker (protects real trailing letters, e.g. SYNL).
    Already-clean symbols pass through.
    """
    s = (internal_id or "").strip()
    if not s or s.upper() == "UNKNOWN":
        return "UNKNOWN"
    up = s.upper()
    stem = s
    had_eq_suffix = False
    for suffix in _INTERNAL_SUFFIXES:
        if up.endswith(suffix):
            stem = s[: len(s) - len(suffix)]
            had_eq_suffix = True
            break
    else:
        if "_" in stem:
            stem = stem.split("_")[0]
    if re.match(r"^[A-Za-z0-9.]+_[A-Za-z]{1,3}$", stem):
        stem = stem.rsplit("_", 1)[0]
    if len(stem) > 1 and stem[-1].islower():
        if known_clean is not None and stem[:-1].upper() in known_clean:
            stem = stem[:-1]
        elif known_clean is None and stem[-1] in ("d", "l", "p", "e") and len(stem) > 4:
            stem = stem[:-1]
    disp = stem.upper()
    return DISPLAY_OVERRIDES.get(disp, disp)
def resolve_company_name(internal_id: str, display_symbol: str, instrument_name: str = "", known_names: Optional[Dict[str, str]] = None) -> str:
    """Resolve a company name; never falls back to an internal broker ID.
    Order: broker symbol -> display symbol -> instrument name ->
    ``Unknown instrument``.
    """
    known = known_names or {}
    internal_key = (internal_id or "").strip().upper()
    display_key = (display_symbol or "").strip().upper()
    base_key = display_key.split("_")[0] if display_key else ""
    for key in (internal_key, display_key, base_key):
        if key and known.get(key):
            return str(known[key])
    if internal_key and COMPANY_OVERRIDES.get(internal_key):
        return COMPANY_OVERRIDES[internal_key]
    if display_key and COMPANY_OVERRIDES.get(display_key):
        return COMPANY_OVERRIDES[display_key]
    if base_key and COMPANY_OVERRIDES.get(base_key):
        return COMPANY_OVERRIDES[base_key]
    name = (instrument_name or "").strip()
    if name and not re.match(r"^[A-Z0-9]+(_[A-Z]+)+$", name.upper()):
        return name
    if name and name.upper() not in (internal_key, display_key):
        return name
    return UNKNOWN_INSTRUMENT
def to_yahoo_symbol(internal_id: str, display_symbol: str = "", aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Map an internal broker ID to a Yahoo Finance ticker.
    Single merging point: ``resolve_alias()`` (config-over-static).
    Returns ``None`` for instruments without a Yahoo representation instead
    of guessing. Never returns company names. Crypto pairs are converted
    (``BTCUSD`` -> ``BTC-USD``).
    EU suffix derivation lives in ``propose_yahoo_candidates()`` and is NOT
    auto-applied here (validation-gated).
    """
    internal_key = (internal_id or "").strip().upper()
    display_key = (display_symbol or "").strip().upper() or to_display_symbol(internal_id)
    internal_stem = _internal_stem(internal_key)
    for key in (internal_key, internal_stem, display_key, display_key.split("_")[0]):
        if key:
            hit = resolve_alias(key, aliases)
            if hit:
                return hit
    m = re.match(r"^([A-Z0-9]{2,12})(USD|USDT)$", display_key)
    if m:
        return f"{m.group(1)}-USD"
    if not display_key or display_key == "UNKNOWN":
        return None
    if display_key in NO_EARNINGS_YAHOO:
        return None
    if " " in display_key or len(display_key) > 12:
        return None
    return display_key
def is_earnings_supported(yahoo_symbol: Optional[str]) -> bool:
    """Whether an earnings lookup is meaningful for a Yahoo ticker."""
    if not yahoo_symbol:
        return False
    return yahoo_symbol not in NO_EARNINGS_YAHOO
def build_known_maps(pie_slices: Optional[List[tuple]] = None, config_assets: Optional[List[dict]] = None, extra_names: Optional[Dict[str, str]] = None) -> tuple:
    """Build (known_clean_set, known_names) from pie slices + config assets.
    pie_slices: iterable of (slice, name). config_assets: portfolio_config
    asset dicts with broker_symbol / yahoo_symbol / name.
    """
    known_clean: set = set()
    known_names: Dict[str, str] = {}
    for item in pie_slices or []:
        try:
            sl, nm = item
        except (TypeError, ValueError):
            continue
        sl_u = str(sl or "").strip().upper()
        if sl_u:
            known_clean.add(sl_u)
            if nm:
                known_names.setdefault(sl_u, str(nm))
    for asset in config_assets or []:
        if not isinstance(asset, dict):
            continue
        for key in ("broker_symbol", "yahoo_symbol"):
            val = str(asset.get(key) or "").strip().upper()
            if val and val != "UNKNOWN":
                known_clean.add(val)
                base = val.split("_")[0].split(".")[0]
                if base:
                    known_clean.add(base)
        nm = asset.get("name")
        for key in (asset.get("broker_symbol"), asset.get("yahoo_symbol")):
            if key and nm:
                known_names.setdefault(str(key).strip().upper(), str(nm))
    for k, v in (extra_names or {}).items():
        ku = str(k).strip().upper()
        if ku:
            known_clean.add(ku)
            known_clean.add(ku.split("_")[0])
            if v:
                known_names.setdefault(ku, str(v))
    return known_clean, known_names
