from __future__ import annotations

import re
from dataclasses import asdict
from datetime import date as _date
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from investment_engine.research.market_regime import RegimeResult
from investment_engine.research.peak_valley import PriceStructure, KeyLevel
from investment_engine.portfolio.symbols import (
    COMPANY_OVERRIDES,
    UNKNOWN_INSTRUMENT,
    resolve_company_name,
    to_display_symbol,
)


# ---------------------------------------------------------------------------
# Signal normalization (single source of truth for portfolio vs actions)
# ---------------------------------------------------------------------------

_BUY_TOKENS = {"BUY", "ACCUMULATE", "ACCUMLATE", "OVERWEIGHT", "INCREASE", "ADD", "ACCUM", "LONG"}
_SELL_TOKENS = {"SELL", "TRIM", "REDUCE", "UNDERWEIGHT", "EXIT", "SHORT"}
_HOLD_TOKENS = {"HOLD", "WAIT", "WATCH", "NEUTRAL", "NONE", "ROTATE"}


def normalize_signal(raw: Any) -> str:
    """Normalize any raw AI/LLM action token to BUY / SELL / HOLD.

    Deterministic. Handles the legacy 'ACCUMLATE' typo and pie-level
    TRIM/REDUCE (both are sales of exposure).
    """
    s = str(raw or "").strip().upper()
    # Strip emoji / punctuation, keep first token
    s = re.sub(r"[^A-Z]", "", s)
    if s in _BUY_TOKENS:
        return "BUY"
    if s in _SELL_TOKENS:
        return "SELL"
    return "HOLD"


def signal_emoji(sig: str) -> str:
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL"}.get(str(sig).upper(), "🟡 HOLD")


def build_canonical_signals(
    ai_recs: dict | None,
    known_clean: set | None = None,
) -> dict[str, dict]:
    """Build display-keyed ticker -> {signal, raw, conflict, confidence}.

    One canonical mapping, built once and shared by every rendered section
    (T212 Portfolio, Priority Actions, AI Recommendations, PIE evaluation,
    KPI text). Keys are upper-case display symbols; the raw internal broker
    ID is indexed as an alias to the same entry so both lookup forms agree.

    Priority: trades (specific SELL/BUY with qty) win over portfolio_actions.
    """
    canon: dict[str, dict] = {}
    if not isinstance(ai_recs, dict):
        return canon

    def _store(sym: str, raw_action: str, source: str) -> None:
        internal = str(sym or "").strip().upper()
        if not internal:
            return
        display = to_display_symbol(sym, known_clean)
        norm = normalize_signal(raw_action)
        # All keys share one entry object so updates stay consistent.
        entry = canon.get(display) or canon.get(internal)
        if entry is None:
            entry = {"signal": norm, "raw": [], "sources": [], "conflict": False,
                     "display": display, "internal": internal}
            canon[display] = entry
            canon[internal] = entry
        if norm != entry["signal"]:
            if source == "trade":
                entry["signal"] = norm
            entry["conflict"] = True
        entry["raw"].append(str(raw_action))
        entry["sources"].append(source)
        # Keep alias keys pointing at the entry.
        canon[display] = entry
        canon[internal] = entry

    for t in ai_recs.get("trades", []) or []:
        if isinstance(t, dict) and t.get("asset"):
            _store(t["asset"], t.get("side", "HOLD"), "trade")
    for a in ai_recs.get("portfolio_actions", []) or []:
        if isinstance(a, dict) and a.get("asset"):
            _store(a["asset"], a.get("action", "HOLD"), "portfolio_action")
    return canon


def ensure_canonical_defaults(canonical: dict | None, display_tickers) -> dict:
    """Fill HOLD defaults so EVERY portfolio ticker has a canonical signal.

    Returns the same mapping (mutated). Tickers absent from structured AI
    recommendations default to HOLD with provenance 'default'; the technical
    observation is preserved in row Notes, never as a divergent signal.
    """
    canon = canonical if isinstance(canonical, dict) else {}
    for disp in display_tickers or []:
        key = str(disp or "").strip().upper()
        if not key or key == "UNKNOWN":
            continue
        if key not in canon:
            canon[key] = {"signal": "HOLD", "raw": ["default"], "sources": ["default"],
                          "conflict": False, "display": key, "internal": key}
    return canon


def lookup_canonical(
    canon: dict[str, dict],
    symbol: str,
    known_clean: set | None = None,
) -> dict | None:
    """Lookup by internal ID, display symbol, then base symbol."""
    if not symbol:
        return None
    key = str(symbol).strip().upper()
    if key in canon:
        return canon[key]
    disp = to_display_symbol(symbol, known_clean)
    if disp in canon:
        return canon[disp]
    base = key.split("_")[0]
    if base in canon:
        return canon[base]
    if disp.split("_")[0] in canon:
        return canon[disp.split("_")[0]]
    return None


def _base_symbol(sym: str) -> str:
    return str(sym or "").strip().upper().split("_")[0]


# ---------------------------------------------------------------------------
# Unified portfolio rows (exactly one holdings table)
# ---------------------------------------------------------------------------

def compute_reconciliation(
    t212_data: dict | None,
    rows: list[dict] | None,
) -> dict:
    """Authoritative reconciliation from unified rows (no speculation).

    positions_value = sum(unified position value_eur)
    reported_cash  = cash_free + cash_pie + cash_blocked (all cash belonging to account)
    derived_total   = positions_value + reported_cash
    diff            = derived_total - total_equity (signed)
    """
    summary = (t212_data or {}).get("account_summary", {}) or {}
    cash = (t212_data or {}).get("cash", {}) or {}
    total_equity = float(summary.get("total_equity", 0) or 0)
    free_cash = cash.get("free", summary.get("cash_free", 0))
    pie_cash = cash.get("pie_cash", summary.get("cash_pie", 0))
    blocked_cash = cash.get("blocked", summary.get("cash_blocked", 0))
    try:
        reported_cash = float(free_cash or 0) + float(pie_cash or 0) + float(blocked_cash or 0)
    except (TypeError, ValueError):
        reported_cash = 0.0
    positions_value = sum(
        float(r.get("market_value", 0) or 0) for r in (rows or []) if isinstance(r, dict)
    )
    derived_total = positions_value + reported_cash
    diff = derived_total - total_equity
    # Cash diagnosis (no cause claimed): the cash balance implied by the
    # broker total vs the cash actually reported.
    implied_cash = total_equity - positions_value
    cash_delta = reported_cash - implied_cash
    threshold = float(summary.get("reconciliation_threshold", 0) or 0)
    if threshold <= 0:
        # Strict tolerance: min(0.1% of broker equity, €2.00).
        threshold = round(min(total_equity * 0.001, 2.00), 2)
    # Fail-closed on missing broker data: an empty snapshot must never PASS.
    if total_equity <= 0 and not rows:
        status = "UNKNOWN"
    else:
        status = str(summary.get("reconciliation_status", "") or "").upper()
        if status not in ("PASS", "FAIL"):
            status = "PASS" if abs(diff) <= threshold else "FAIL"
    return {
        "total_equity": total_equity,
        "positions_value": positions_value,
        "reported_cash": reported_cash,
        "free_cash": float(free_cash or 0),
        "pie_cash": float(pie_cash or 0),
        "blocked_cash": float(blocked_cash or 0),
        "implied_cash": implied_cash,
        "cash_delta": cash_delta,
        "derived_total": derived_total,
        "diff": diff,
        "threshold": threshold,
        "status": status,
    }


def apply_fail_guards(ai_recs: dict | None, recon_status: str) -> dict:
    """Deterministic FAIL-state guards for structured recommendations.

    When reconciliation FAILs, concrete BUY trades and cash deployment
    instructions are removed (SELLs kept). Idempotent. Returns the
    (possibly new) dict.
    """
    if not isinstance(ai_recs, dict) or str(recon_status or "").upper() != "FAIL":
        return ai_recs or {}
    import copy
    guarded = copy.deepcopy(ai_recs)
    guarded["trades"] = [t for t in guarded.get("trades", []) or []
                         if not (isinstance(t, dict) and str(t.get("side", "")).strip().upper() == "BUY")]
    cd = guarded.get("cash_deployment")
    if isinstance(cd, dict):
        try:
            free = float(cd.get("free_cash", 0) or 0)
        except (TypeError, ValueError):
            free = 0.0
        guarded["cash_deployment"] = {"free_cash": free, "deploy_amount": 0.0,
                                      "deploy_pct": 0.0, "reserve": free, "targets": []}
    return guarded


def sell_risk_rule(row: dict | None) -> bool:
    """Risk-reduction SELL rule: weight>=5% or P&L<=-8% or price below support."""
    if not isinstance(row, dict):
        return False
    try:
        if float(row.get("weight", 0) or 0) >= 5.0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if float(row.get("pnl_pct", 0) or 0) <= -8.0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        px = float(row.get("current_price", 0) or 0)
        sup = row.get("support")
        if sup is not None and px > 0 and px < float(sup):
            return True
    except (TypeError, ValueError):
        pass
    return False


def apply_trade_safety(
    canonical: dict | None,
    rows: list[dict] | None,
    technicals: dict | None = None,
    recon_status: str = "UNKNOWN",
    earnings_status: dict | None = None,
    concentration_cap_pct: float = 7.0,
) -> dict:
    """Enforce trade safety rules on the shared canonical mapping (in place).

    A BUY requires: reconciliation PASS, valid current price, valid technical
    data, weight below the concentration cap, and no averaging-down solely on
    negative P&L (oversold RSI<=30, earnings<=14d, or price near support may
    still justify it). In FAIL state a SELL survives only via the
    risk-reduction rule (weight>=5%, P&L<=-8%, or price below support);
    otherwise owned positions demote to canonical HOLD. Demotions apply to the
    matching row and its Notes. Returns the mapping.
    """
    canon = canonical if isinstance(canonical, dict) else {}
    recon_ok = str(recon_status or "").upper() == "PASS"
    by_display: dict[str, dict] = {}
    for r in rows or []:
        if isinstance(r, dict) and (r.get("display_symbol") or r.get("ticker")):
            by_display[str(r.get("display_symbol") or r.get("ticker")).strip().upper()] = r
    seen: set[str] = set()

    def _demote(entry: dict, row: dict | None, signal: str, reason: str, verb: str) -> None:
        entry["signal"] = signal
        entry["gated"] = True
        entry["gate_reason"] = reason
        if row is not None:
            row["signal"] = signal
            note = f"{verb} withheld: {reason}."
            row["notes"] = f"{row['notes']}; {note}" if row.get("notes") else note

    for key, entry in list(canon.items()):
        if not isinstance(entry, dict) or entry.get("signal") not in ("BUY", "SELL"):
            continue
        disp = str(entry.get("display") or key).strip().upper()
        if disp in seen:
            continue
        seen.add(disp)
        row = by_display.get(disp)
        if entry["signal"] == "SELL":
            if recon_ok or sell_risk_rule(row):
                continue
            _demote(entry, row, "HOLD", "no risk-reduction rule under FAIL", "SELL")
            continue
        reason = ""
        if not recon_ok:
            reason = "reconciliation FAIL"
        else:
            try:
                px = float((row or {}).get("current_price", 0) or 0)
                mv = float((row or {}).get("market_value", 0) or 0)
            except (TypeError, ValueError):
                px, mv = 0.0, 0.0
            if px <= 0 or mv <= 0:
                reason = "missing valid price"
        tech = (technicals or {}).get(disp) if isinstance(technicals, dict) else None
        tech_valid = isinstance(tech, dict) and any(
            tech.get(k) is not None for k in ("RSI_14", "Support", "Resistance", "price")
        )
        if not reason and not tech_valid:
            reason = "missing technical data"
        if not reason and row is not None:
            try:
                if float(row.get("weight", 0) or 0) >= concentration_cap_pct:
                    reason = (
                        f"weight {float(row.get('weight', 0)):.1f}% at/above "
                        f"concentration cap {concentration_cap_pct:.1f}%"
                    )
            except (TypeError, ValueError):
                pass
        if not reason and row is not None:
            try:
                pnl = float(row.get("pnl_pct", 0) or 0)
            except (TypeError, ValueError):
                pnl = 0.0
            if pnl < 0:
                catalyst = False
                try:
                    rsi = float((tech or {}).get("RSI_14")) if tech and (tech or {}).get("RSI_14") is not None else None
                    if rsi is not None and rsi <= 30:
                        catalyst = True
                except (TypeError, ValueError):
                    pass
                if not catalyst:
                    try:
                        from datetime import date as _d
                        earn = None
                        if isinstance(earnings_status, dict):
                            val = earnings_status.get(disp)
                            earn = val[0][:10] if isinstance(val, (tuple, list)) and val and isinstance(val[0], str) else None
                        if earn and 0 <= (_d.fromisoformat(earn) - _d.today()).days <= 14:
                            catalyst = True
                    except (ValueError, TypeError):
                        pass
                if not catalyst and row is not None:
                    try:
                        px = float(row.get("current_price", 0) or 0)
                        sup = row.get("support")
                        if sup is not None and px > 0 and abs(px - float(sup)) / px <= 0.02:
                            catalyst = True
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                if not catalyst:
                    reason = "averaging down on negative P&L without oversold/trigger"
        if reason:
            _demote(entry, row, "HOLD", reason, "BUY")
    return canon


def _hold_observation(signal: str, tech: dict | None, support: Any, resistance: Any) -> str:
    """Deterministic technical observation preserved alongside a HOLD signal."""
    if signal != "HOLD":
        return ""
    rsi = (tech or {}).get("RSI_14") if isinstance(tech, dict) else None
    try:
        rsi_f = float(rsi) if rsi is not None else None
    except (TypeError, ValueError):
        rsi_f = None
    watch = ""
    try:
        if resistance is not None and float(resistance) > 0:
            watch = f"watch €{float(resistance):.2f} resistance before upgrade"
    except (TypeError, ValueError):
        watch = ""
    if rsi_f is not None and rsi_f <= 45:
        base = f"improving setup off RSI {rsi_f:.1f}"
    elif rsi_f is not None and rsi_f >= 70:
        base = f"overbought RSI {rsi_f:.1f}; wait for cooldown"
    else:
        base = "steady setup"
    if watch:
        return f"HOLD — {base}; {watch}."
    return f"HOLD — {base}."


def build_unified_portfolio_rows(
    t212_data: dict | None,
    ai_recs: dict | None = None,
    technicals: dict | None = None,
    names: dict | None = None,
    known_clean: set | None = None,
    canonical: dict | None = None,
    yahoo_map: dict | None = None,
) -> list[dict]:
    """Build the single authoritative holdings list.

    ``all_positions`` is the complete broker snapshot when non-empty;
    ``positions`` is a fallback only. Entries are never merged or summed:
    duplicate broker IDs keep the first occurrence (counted, not summed).
    Each row carries ``internal_id`` (broker), ``display_symbol`` (rendered)
    and ``company_name`` (never an internal ID).
    Sorted by market value desc. Deterministic.
    """
    if not isinstance(t212_data, dict):
        return []
    summary = t212_data.get("account_summary", {}) or {}
    # Authoritative source first; fallback only when unavailable.
    raw_positions: list[dict] = []
    source = "positions"
    if isinstance(summary.get("all_positions"), list) and summary["all_positions"]:
        raw_positions = summary["all_positions"]
        source = "all_positions"
    elif isinstance(t212_data.get("all_positions"), list) and t212_data["all_positions"]:
        raw_positions = t212_data["all_positions"]
        source = "all_positions"
    else:
        raw_positions = list(t212_data.get("positions", []) or [])
        if not raw_positions and isinstance(summary.get("positions"), list):
            raw_positions = summary["positions"]

    # Deduplicate by broker ID: first occurrence wins, never summed.
    by_sym: dict[str, dict] = {}
    order: list[str] = []
    duplicate_ids: set[str] = set()
    for p in raw_positions:
        if not isinstance(p, dict) or not p.get("symbol"):
            continue
        key = str(p["symbol"]).strip().upper()
        if key in by_sym:
            duplicate_ids.add(key)
            continue
        by_sym[key] = p
        order.append(key)

    total_equity = float(summary.get("total_equity", 0) or 0)
    if total_equity <= 0:
        total_equity = sum(float(p.get("value_eur", p.get("value", 0)) or 0) for p in by_sym.values())

    canon = canonical if isinstance(canonical, dict) else build_canonical_signals(ai_recs, known_clean)
    # Every portfolio ticker gets a canonical entry (HOLD default), so no
    # section can invent a divergent signal for an uncovered ticker.
    try:
        ensure_canonical_defaults(
            canon,
            [to_display_symbol((by_sym[k].get("display_symbol") or by_sym[k].get("symbol", k)), known_clean) for k in order],
        )
    except Exception:
        pass
    tech_lookup: dict[str, dict] = {}
    for k, v in (technicals or {}).items():
        if isinstance(v, dict):
            tech_lookup[str(k).strip().upper()] = v
    name_lookup: dict[str, str] = {}
    for k, v in (names or {}).items():
        name_lookup[str(k).strip().upper()] = str(v)

    rows: list[dict] = []
    for sym_key in sorted(order, key=lambda s: float(by_sym[s].get("value_eur", by_sym[s].get("value", 0)) or 0), reverse=True):
        p = by_sym[sym_key]
        internal_id = str(p.get("symbol", sym_key))
        # Display always derived from the internal broker ID with the known
        # universe (a precomputed display_symbol without that context would
        # leak exchange codes such as EGTL instead of EGT).
        display = to_display_symbol(internal_id, known_clean)
        if not display or display == "UNKNOWN":
            display = to_display_symbol(internal_id, None)
        qty = float(p.get("quantity", 0) or 0)
        avg_cost = float(p.get("average_price_eur", p.get("avg_price", 0)) or 0)
        cur_price_eur = float(p.get("current_price_eur", p.get("price", 0)) or 0)
        cur_price_native = float(p.get("current_price", p.get("current_price_native", 0)) or 0)
        # Broker native quote currency (e.g., USD, EUR, GBX, DKK)
        quote_ccy = str(p.get("quote_currency", p.get("broker_quote_currency", ""))).upper()
        price_unit = str(p.get("price_unit", "")).upper()
        mval = float(p.get("value_eur", p.get("value", 0)) or 0)
        unreal = float(p.get("pnl_eur", p.get("pnl", 0)) or 0)
        realized_raw = p.get("realized_pnl_eur", p.get("realized_pnl"))
        realized = float(realized_raw) if realized_raw is not None else None
        total_pnl = unreal + realized if realized is not None else unreal
        pnl_pct = float(p.get("pnl_pct", 0) or 0)
        weight = (mval / total_equity * 100.0) if total_equity > 0 else 0.0

        entry = lookup_canonical(canon, internal_id, known_clean) or lookup_canonical(canon, display, known_clean)
        signal = entry["signal"] if entry else "HOLD"
        conflict = bool(entry and entry.get("conflict"))
        raw_list = (entry or {}).get("raw", [])
        confidence = "medium" if entry else "n/a"

        tech = tech_lookup.get(display) or tech_lookup.get(internal_id.strip().upper())
        support = (tech or {}).get("Support")
        resistance = (tech or {}).get("Resistance")

        # Market-data support state for this row (no yfinance calls here).
        yahoo_symbol = (yahoo_map or {}).get(display) if isinstance(yahoo_map, dict) else None
        try:
            from investment_engine.portfolio.symbols import (
                MAPPING_UNAVAILABLE_NOTE,
                support_state as _support_state,
            )
            md_state = _support_state(yahoo_symbol, "market_data") if yahoo_symbol else "UNSUPPORTED"
        except ImportError:
            MAPPING_UNAVAILABLE_NOTE = "Market-data mapping unavailable; broker valuation retained."
            md_state = "SUPPORTED" if tech else "UNRESOLVED"

        # External mapping status: separate from market_data_state
        # VERIFIED = mapping exists and currencies match for technical comparison
        # MAPPING_SUSPECT = known problematic mappings (VWSB/EUR vs VWS.CO/DKK, etc.)
        # UNRESOLVED = no mapping found
        # BROKER_ONLY = instrument only exists in broker, no external mapping
        ext_mapping_status = md_state
        if md_state == "SUPPORTED":
            # Check for known suspect mappings
            ext_sym = yahoo_symbol or ""
            if (display == "VWSB" and ext_sym == "VWS.CO") or \
               (display == "LITMM" and ext_sym == "LIT") or \
               (display == "SYNL" and "SYN" in ext_sym):
                ext_mapping_status = "MAPPING_SUSPECT"
            else:
                ext_mapping_status = "VERIFIED"
        elif md_state == "UNSUPPORTED":
            ext_mapping_status = "BROKER_ONLY"
        elif md_state == "UNRESOLVED":
            ext_mapping_status = "UNRESOLVED"

        company = resolve_company_name(
            internal_id, display, str(p.get("name", "") or ""), name_lookup
        )

        validated = str(p.get("validation_status", "PASS") or "PASS").upper() == "PASS"
        notes: list[str] = []
        if conflict:
            notes.append(f"signal conflict normalized to {signal} (raw: {','.join(sorted(set(raw_list)))})")
        if isinstance(entry, dict) and entry.get("gated") and entry.get("gate_reason"):
            notes.append(f"BUY withheld: {entry['gate_reason']}.")
        if not validated:
            notes.append(str(p.get("validation_reason", "validation FAIL")))
        if sym_key in duplicate_ids:
            notes.append("duplicate broker entry ignored (not summed)")
        if p.get("is_pie_constituent"):
            notes.append("pie-held")
        if p.get("fx_is_fallback"):
            notes.append("FX fallback")
        if md_state == "UNRESOLVED" and not tech:
            notes.append(MAPPING_UNAVAILABLE_NOTE)
        obs = _hold_observation(signal, tech if isinstance(tech, dict) else None, support, resistance)
        if obs:
            notes.append(obs)

        rows.append({
            "ticker": display,
            "display_symbol": display,
            "internal_id": internal_id,
            "company": company,
            "company_name": company,
            "quantity": qty,
            "avg_cost": avg_cost,
            "current_price": cur_price_eur,  # EUR-converted for portfolio valuation
            "current_price_native": cur_price_native,  # Native quote currency for technical comparison
            "quote_currency": quote_ccy,  # Native quote currency (USD, EUR, GBX, DKK, etc.)
            "price_unit": price_unit,  # Price unit (GBX for pence)
            "market_value": mval,
            "unrealized_pnl": unreal,
            "realized_pnl": realized,
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "weight": weight,
            "signal": signal,
            "confidence": confidence,
            "support": support,
            "resistance": resistance,
            "notes": "; ".join(notes),
            "is_pie_constituent": bool(p.get("is_pie_constituent", False)),
            "pnl_validated": validated,
            "source": source,
            "yahoo_symbol": yahoo_symbol,
            "market_data_state": md_state,
            "external_mapping_status": ext_mapping_status,
            "external_quote_currency": (tech or {}).get("quote_currency", "") if isinstance(tech, dict) else "",
        })
    return rows


def unresolved_market_data(rows: list[dict] | None) -> list[str]:
    """Display symbols whose market-data mapping is unavailable (UNRESOLVED)."""
    return sorted({str(r.get("display_symbol") or r.get("ticker"))
                   for r in (rows or []) if isinstance(r, dict)
                   and r.get("market_data_state") == "UNRESOLVED"})


def _fmt_eur(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"€{float(v):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


class RegimeReportGenerator:
    """Generates clean, actionable Markdown reports from regime analysis."""

    def __init__(self, include_charts: bool = False):
        self.include_charts = include_charts

    def generate_parts(
        self,
        result: RegimeResult,
        t212_data: dict | None = None,
        ai_recs: dict | None = None,
        earnings: dict | None = None,
        technicals: dict | None = None,
        names: dict | None = None,
        known_clean: set | None = None,
        canonical: dict | None = None,
        yahoo_map: dict | None = None,
    ) -> dict[str, str]:
        """Generate report as named parts for custom assembly order.

        Parts: exi2 (regime panel), portfolio (unified T212 holdings),
        ai (recommendations), warnings. Earnings radar is rendered
        separately by the caller (see main._format_earnings_radar) or via
        _earnings_calendar when earnings are supplied here.
        """
        exi2 = "\n\n".join([
            self._header(result),
            self._regime_summary(result),
            self._key_levels(result),
            self._news_section(result),
        ])
        parts: dict[str, str] = {
            "exi2": exi2,
            "portfolio": "",
            "ai": "",
            "warnings": self._warnings(result),
        }
        if ai_recs:
            try:
                _rows_for_ai = build_unified_portfolio_rows(
                    t212_data, ai_recs, technicals, names, known_clean, canonical, yahoo_map)
                _recon_for_ai = compute_reconciliation(t212_data, _rows_for_ai)["status"]
            except Exception:
                _rows_for_ai, _recon_for_ai = None, "UNKNOWN"
            # FAIL-state guards at the choke point: no owned-position BUY,
            # DCA deployment, or purchase instruction may render.
            _safe_recs = apply_fail_guards(ai_recs, _recon_for_ai)
            if t212_data:
                parts["portfolio"] = self._t212_portfolio(
                    t212_data, _safe_recs, technicals, names, known_clean, canonical, _rows_for_ai, yahoo_map)
            parts["ai"] = self._ai_recommendations(_safe_recs, known_clean, canonical, names, _recon_for_ai)
        if earnings:
            parts["earnings"] = self._earnings_calendar(result, earnings, names)
        return {k: v for k, v in parts.items() if v}

    def generate(
        self,
        result: RegimeResult,
        t212_data: dict | None = None,
        ai_recs: dict | None = None,
        earnings: dict | None = None,
        technicals: dict | None = None,
        names: dict | None = None,
        known_clean: set | None = None,
        canonical: dict | None = None,
        yahoo_map: dict | None = None,
    ) -> str:
        """Generate clean, actionable Markdown report (dashboard order)."""
        parts = self.generate_parts(result, t212_data, ai_recs, earnings, technicals, names, known_clean, canonical, yahoo_map)
        return "\n\n".join(
            parts[k] for k in ("exi2", "portfolio", "ai", "earnings", "warnings") if parts.get(k)
        )

    def _header(self, result: RegimeResult) -> str:
        regime_emoji = {
            "PEAK_HOLD": "🔴",
            "DECLINING": "🟠", 
            "MUST_BUY": "🟢",
            "NEUTRAL": "⚪",
            "ERROR": "❌",
        }.get(result.regime, "❓")

        dt = datetime.fromisoformat(result.generated_at.replace('Z', '+00:00'))
        return f"""# {regime_emoji} EXI2 Market Regime — {dt.strftime('%Y-%m-%d %H:%M UTC')}

**Regime:** **{result.regime}** ({result.confidence:.0%}) | **EXI2:** €{result.timeframes.get('daily', {}).get('indicators', {}).get('CLOSE', 0):.2f} | **RSI:** {result.timeframes.get('daily', {}).get('indicators', {}).get('RSI_14', 0):.0f} | **Trend:** {result.timeframes.get('daily', {}).get('trend', 'N/A')}"""

    def _regime_summary(self, result: RegimeResult) -> str:
        impl = result.implications
        return f"""## 📊 Regime Summary

| Action | Detail |
|--------|--------|
| **Portfolio Action** | {impl.get('portfolio_action', 'N/A')} |
| **Tech Allocation** | {impl.get('tech_allocation', 'N/A')} |
| **Crypto Allocation** | {impl.get('crypto_allocation', 'N/A')} |
| **DCA Multiplier** | {impl.get('dca_multiplier', 1.0)}x |
| **Cash Target** | {impl.get('cash_target_pct', 10)}% |
| **Guidance** | {impl.get('message', 'N/A')} |

**Primary Signal:** {result.primary_signal or 'None'}"""

    def _key_levels(self, result: RegimeResult) -> str:
        ps: PriceStructure = result.price_structure
        lines = ["## 🎯 Key Levels to Watch"]

        if ps.nearest_support:
            lines.append(f"- **Support:** €{ps.nearest_support:.2f} (break → €{ps.nearest_support * 0.97:.2f})")
        if ps.nearest_resistance:
            lines.append(f"- **Resistance:** €{ps.nearest_resistance:.2f} (clear → €{ps.nearest_resistance * 1.03:.2f})")

        daily_ind = result.timeframes.get("daily", {}).get("indicators", {})
        if daily_ind.get("SMA_200"):
            status = "Above" if daily_ind.get('CLOSE', 0) > daily_ind['SMA_200'] else "Below"
            lines.append(f"- **200-Day MA:** €{daily_ind['SMA_200']:.2f} ({status})")
        if daily_ind.get("SMA_50"):
            lines.append(f"- **50-Day MA:** €{daily_ind['SMA_50']:.2f}")

        # Fib levels
        if ps.peaks and ps.valleys:
            fib = self._compute_fib_levels(ps.peaks, ps.valleys)
            if fib:
                lines.append("- **Fib Levels:** " + " | ".join(f"{k}: €{v:.2f}" for k, v in fib.items() if k in ["0.382", "0.500", "0.618", "0.786"]))

        return "\n".join(lines)

    def _compute_fib_levels(self, peaks, valleys) -> dict[str, float] | None:
        if not peaks or not valleys:
            return None
        last_peak = max(peaks, key=lambda p: p.date)
        last_valley = max(valleys, key=lambda v: v.date)
        if last_peak.date > last_valley.date:
            high, low = last_peak.price, last_valley.price
        else:
            high, low = last_valley.price, last_peak.price
        diff = high - low
        return {
            "0.382": high - diff * 0.382,
            "0.500": high - diff * 0.500,
            "0.618": high - diff * 0.618,
            "0.786": high - diff * 0.786,
        }

    def _news_section(self, result: RegimeResult) -> str:
        ns = result.news_sentiment
        if not ns or ns.get("count", 0) == 0:
            return "## 📰 News (Last 48h)\n\nNo qualifying news found."

        lines = ["## 📰 News (Last 48h)", ""]
        lines.append(f"**Sentiment:** {ns.get('sentiment', 'N/A')} ({ns.get('score', 0)}/100) | **Articles:** {ns.get('count', 0)}")
        
        # Note: news items with URLs would be added here if passed in
        # For now, show sentiment summary
        if ns.get('key_topics'):
            lines.append(f"**Topics:** {', '.join(ns['key_topics'])}")
        
        return "\n".join(lines)

    def _t212_portfolio(
        self,
        data: dict | None,
        ai_recs: dict | None = None,
        technicals: dict | None = None,
        names: dict | None = None,
        known_clean: set | None = None,
        canonical: dict | None = None,
        rows: list[dict] | None = None,
        yahoo_map: dict | None = None,
        performance: dict | None = None,
    ) -> str:
        """Format the single unified T212 holdings table.

        Exactly one holdings section named 'T212 Portfolio'. The authoritative
        source is ``all_positions`` (complete broker snapshot); ``positions``
        is a fallback only and is never merged in. All rendered tickers are
        clean display symbols — internal broker IDs never appear in cells.
        """
        if not data or data.get("status") in ("failed", "disabled"):
            return "## 💼 T212 Portfolio\n\n❌ Unable to fetch portfolio data."

        summary = data.get("account_summary", {}) or {}
        cash = data.get("cash", {}) or {}

        lines = ["## 💼 T212 Portfolio", ""]
        # Backward-compat alias so legacy checks for "Trading212 Portfolio" keep passing.
        lines.append("*Trading212 Portfolio — unified holdings (authoritative broker snapshot, deduplicated).*")
        lines.append("")

        # Unified rows first: reconciliation is computed from them, not from
        # precomputed broker fields.
        if rows is None:
            rows = build_unified_portfolio_rows(data, ai_recs, technicals, names, known_clean, canonical, yahoo_map)
        recon = compute_reconciliation(data, rows)
        total_equity = recon["total_equity"]
        free_cash = recon["free_cash"]
        pie_cash = recon["pie_cash"]
        blocked_cash = float(summary.get("cash_blocked", cash.get("blocked", 0)) or 0)

        lines.append(f"**Total Equity (broker):** €{total_equity:,.2f}")
        lines.append(f"**Free Cash:** €{free_cash:,.2f} | **Pie Cash:** €{pie_cash:,.2f} | **Blocked:** €{blocked_cash:,.2f}")
        lines.append(f"**Positions value (unified):** €{recon['positions_value']:,.2f}")
        lines.append(
            f"**Reconciliation:** positions €{recon['positions_value']:,.2f} + cash €{recon['reported_cash']:,.2f} "
            f"= derived €{recon['derived_total']:,.2f} vs broker €{total_equity:,.2f} "
            f"(diff €{recon['diff']:+,.2f}, threshold €{recon['threshold']:,.2f}) → {recon['status']}"
        )
        if recon["status"] == "FAIL":
            lines.append("")
            lines.append("> ⚠️ **Reconciliation check failed — account-level totals withheld.**")
            lines.append(
                f"> Broker total: **€{total_equity:,.2f}** | Holdings + available cash: **€{recon['derived_total']:,.2f}** "
                f"| Difference: **€{recon['diff']:+,.2f}**. Per-position data below is shown only where validated."
            )
            lines.append(
                f"> Cash diagnosis (no cause claimed): implied cash €{recon['implied_cash']:,.2f} | "
                f"reported cash €{recon['reported_cash']:,.2f} | delta €{recon['cash_delta']:+,.2f}."
            )
        else:
            lines.append("*Position values use normalized quote units and are reconciled to broker equity.*")

        if recon["status"] == "PASS":
            lines.append(f"**Open Position Unrealized P&L (validated):** €{sum(r['unrealized_pnl'] for r in rows if r.get('pnl_validated')):,.2f}")
            lines.append("*Note: Realized P&L and account-level return require full transaction ledger (not available via API).*")
        elif isinstance(performance, dict) and performance.get("performance_status") == "Available":
            from investment_engine.accounting.cashflows import FAIL_PERFORMANCE_NOTE
            pnl = performance.get("net_pnl_after_costs_eur")
            ret = performance.get("return_pct")
            try:
                perf_line = f"**Account performance (broker equity):** Net P&L €{float(pnl):+,.2f} ({float(ret):+.4f}%)"
            except (TypeError, ValueError):
                perf_line = "**Account performance (broker equity):** available — see snapshot."
            lines.append(perf_line)
            lines.append(f"> {FAIL_PERFORMANCE_NOTE}")
        else:
            lines.append("**Account-level return and total P&L are withheld (reconciliation FAIL).**")
        lines.append("")

        # Unified holdings table — the ONLY per-position table.
        if not rows:
            lines.append("No positions to display.")
            return "\n".join(lines)

        lines.append("| Ticker | Company | Qty | Avg Cost | Current Price | Market Value | Unrealized P&L | Realized P&L | Total P&L | P&L % | Weight | Signal | Conf | Support | Resistance | Notes |")
        lines.append("|--------|---------|-----|----------|---------------|--------------|----------------|--------------|-----------|-------|--------|--------|------|---------|------------|-------|")
        for r in rows:
            if r.get("pnl_validated"):
                unreal_s = _fmt_eur(r["unrealized_pnl"])
                total_s = _fmt_eur(r["total_pnl"])
                pct_s = f"{r['pnl_pct']:+.1f}%"
            else:
                unreal_s = total_s = pct_s = "n/a"
            realized_s = _fmt_eur(r["realized_pnl"])
            lines.append(
                f"| {r['display_symbol']} | {r['company']} | {r['quantity']:.4f} | {_fmt_eur(r['avg_cost'])} | {_fmt_eur(r['current_price'])} | "
                f"{_fmt_eur(r['market_value'])} | {unreal_s} | {realized_s} | {total_s} | "
                f"{pct_s} | {r['weight']:.2f}% | {signal_emoji(r['signal'])} | {r['confidence']} | "
                f"{_fmt_num(r['support'])} | {_fmt_num(r['resistance'])} | {r['notes'] or '—'} |"
            )
        lines.append("")
        lines.append("*Realized P&L requires the full transaction ledger; shown as n/a when the ledger is not imported. Signal is the canonical signal shared by every section; observations are preserved in Notes.*")

        return "\n".join(lines)

    def _ai_recommendations(
        self,
        recs: dict,
        known_clean: set | None = None,
        canonical: dict | None = None,
        names: dict | None = None,
        recon_status: str = "UNKNOWN",
    ) -> str:
        """Format AI recommendations section (display symbols, canonical signals).

        When reconciliation is not PASS, raw BUY/DCA actions are rewritten to
        HOLD-withheld lines and DCA-driven position updates are suppressed.
        """
        if not recs:
            return ""
        if isinstance(recs, dict) and recs.get("error") and not recs.get("portfolio_actions") and not recs.get("trades"):
            return ""
        import re as _re_ai
        fail_state = str(recon_status or "").upper() != "PASS"

        def _disp(sym: str) -> str:
            return to_display_symbol(sym or "?", known_clean)

        def _fnum(v, fmt="{:.2f}", default="n/a"):
            try:
                return fmt.format(float(v)) if v is not None else default
            except (TypeError, ValueError):
                return default

        lines = ["## 🤖 AI Recommendations", ""]

        # Portfolio-level actions. In FAIL state, raw BUY/DCA actions are
        # rewritten (never displayed raw); otherwise the canonical signal is
        # annotated when the raw model output diverges.
        if recs.get("portfolio_actions"):
            lines.append("### Portfolio Actions")
            for action in recs["portfolio_actions"]:
                if not isinstance(action, dict):
                    continue
                act = str(action.get("action", "hold")).upper()
                disp = _disp(action.get("asset"))
                if fail_state and normalize_signal(act) == "BUY":
                    lines.append(
                        f"- **HOLD** — {disp}: original BUY/DCA proposal "
                        f"withheld because account reconciliation failed."
                    )
                    continue
                suffix = ""
                if isinstance(canonical, dict):
                    entry = canonical.get(disp.strip().upper())
                    if isinstance(entry, dict) and entry.get("signal") and entry["signal"] != normalize_signal(act):
                        suffix = f" (canonical: {entry['signal']} — see T212 Portfolio)"
                lines.append(f"- **{act}** {disp}: {action.get('reason', '')}{suffix}")
                if action.get("details"):
                    lines.append(f"  - {action['details']}")
            lines.append("")

        # Specific trade recommendations (BUY rows demoted by trade safety
        # are annotated instead of silently dropped).
        if recs.get("trades"):
            lines.append("### Trade Recommendations")
            lines.append("| Action | Asset | Qty | Price | Stop Loss | Take Profit | Risk |")
            lines.append("|--------|-------|-----|-------|-----------|-------------|------|")
            for trade in recs["trades"]:
                if not isinstance(trade, dict):
                    continue
                disp = _disp(trade.get("asset"))
                side = str(trade.get("side", "?")).strip().upper()
                note = ""
                if side == "BUY" and isinstance(canonical, dict):
                    entry = canonical.get(disp.strip().upper())
                    if isinstance(entry, dict) and entry.get("signal") != "BUY":
                        note = f" (withheld: {entry.get('gate_reason', 'safety gate')} — see T212 Portfolio)"
                lines.append(
                    f"| {trade.get('side', '?')}{note} | {disp} | {trade.get('qty', 'n/a')} | "
                    f"€{_fnum(trade.get('price'))} | €{_fnum(trade.get('stop_loss'))} | "
                    f"€{_fnum(trade.get('take_profit'))} | {_fnum(trade.get('risk_pct'), '{:.1f}%', 'n/a')} |"
                )
            lines.append("")

        # Stop loss / take profit updates. In FAIL state the whole block is
        # suppressed: no SL/TP change may render while reconciliation fails.
        updates = [u for u in (recs.get("position_updates") or []) if isinstance(u, dict)]
        if fail_state:
            updates = []
        if updates:
            lines.append("### Position Management")
            for upd in updates:
                lines.append(
                    f"- **{_disp(upd.get('asset'))}**: "
                    f"SL €{_fnum(upd.get('stop_loss'))} → €{_fnum(upd.get('new_stop'))} | "
                    f"TP €{_fnum(upd.get('take_profit'))} → €{_fnum(upd.get('new_tp'))} ({upd.get('reason', '')})"
                )
            lines.append("")

        # Cash deployment. In FAIL state the standalone block is suppressed
        # entirely: the withheld-deployment dashboard sentence is the only
        # place deployment figures may appear.
        if recs.get("cash_deployment") and not fail_state:
            cd = recs["cash_deployment"]
            if isinstance(cd, dict):
                lines.append("### Cash Deployment")
                lines.append(f"- **Free Cash:** €{_fnum(cd.get('free_cash'), '{:,.2f}', '0.00')}")
                lines.append(f"- **Deploy:** €{_fnum(cd.get('deploy_amount'), '{:,.2f}', '0.00')} ({_fnum(cd.get('deploy_pct'), '{:.0f}', '0')}%)")
                lines.append(f"- **Keep Reserve:** €{_fnum(cd.get('reserve'), '{:,.2f}', '0.00')}")
                for target in cd.get("targets", []) or []:
                    if not isinstance(target, dict):
                        continue
                    lines.append(f"  - {_disp(target.get('asset'))}: €{_fnum(target.get('amount'), '{:,.2f}')} @ €{_fnum(target.get('price'))} (max {_fnum(target.get('max_risk'), '{:.1f}')} risk)")
                lines.append("")

        return "\n".join(lines)

    def _earnings_calendar(
        self,
        result: RegimeResult | None = None,
        earnings: dict | None = None,
        names: dict | None = None,
    ) -> str:
        """Render Earnings Radar strictly as a month-grouped chronological timeline.

        Format:
            ## September 2026
            - Sep 12 — TICKER, Company — Q3 earnings (confirmed)

        Inline certainty tags only: (confirmed) / (estimated).
        No status column, no duplicate events, no prose summaries.
        """
        _ = result
        rows = _normalize_earnings_rows(earnings, names)
        if not rows:
            return "## Earnings Radar\n\n*No upcoming earnings found in Yahoo Finance.*"
        return _render_earnings_radar(rows)

    def _warnings(self, result: RegimeResult) -> str:
        warnings = result.warnings
        if not warnings:
            return ""
        
        lines = ["## ⚠️ Risk Warnings", ""]
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        return "\n".join(lines)


def _normalize_earnings_rows(earnings: dict | None, names: dict | None = None) -> list[tuple[str, str, str, str]]:
    """Normalize heterogeneous earnings inputs to (date_iso, ticker, company, status).

    Accepts: {ticker: 'YYYY-MM-DD'}, {ticker: (date, status)},
    {ticker: {'date':..,'status':..}}. Dedupes and filters past dates.
    """
    if not isinstance(earnings, dict) or not earnings:
        return []
    name_lookup: dict[str, str] = {}
    for k, v in (names or {}).items():
        name_lookup[str(k).strip().upper()] = str(v)
        name_lookup[_base_symbol(k)] = str(v)
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str, str]] = []
    today = _date.today().isoformat()
    for disp, val in earnings.items():
        ticker = str(disp or "").strip()
        if not ticker:
            continue
        d: str | None = None
        st: str | None = None
        if isinstance(val, (tuple, list)) and len(val) >= 1:
            d = str(val[0])[:10] if val[0] else None
            st = str(val[1]).lower().strip("() ") if len(val) > 1 and val[1] else None
        elif isinstance(val, dict):
            d = str(val.get("date", val.get("earnings_date", "")))[:10] or None
            st = str(val.get("status", "")).lower().strip("() ") if val.get("status") else None
        elif isinstance(val, str):
            if re.match(r"^\d{4}-\d{2}-\d{2}", val):
                d = val[:10]
            else:
                continue
        else:
            continue
        if not d or not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        if d < today:
            continue
        base = _base_symbol(ticker)
        if st not in ("confirmed", "estimated"):
            try:
                from investment_engine.research.market_data import classify_earnings_status as _cls
                st = _cls(ticker, d)
            except Exception:
                st = "estimated"
        company = name_lookup.get(ticker.strip().upper()) or name_lookup.get(base) or base
        key = (d, ticker.strip().upper())
        if key in seen:
            continue
        seen.add(key)
        rows.append((d, ticker, company, st))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def _earnings_quarter(date_iso: str) -> str:
    """Reporting quarter for an earnings date (reporting lag: previous calendar quarter)."""
    try:
        m = int(date_iso[5:7])
    except (ValueError, TypeError):
        return "earnings"
    mapping = {1: "Q4", 2: "Q4", 3: "Q4", 4: "Q1", 5: "Q1", 6: "Q1",
               7: "Q2", 8: "Q2", 9: "Q2", 10: "Q3", 11: "Q3", 12: "Q3"}
    return f"{mapping.get(m, '')} earnings".strip() or "earnings"


def _render_earnings_radar(rows: list[tuple[str, str, str, str]]) -> str:
    """Render normalized rows as month-grouped timeline with inline tags."""
    months: dict[str, list[tuple[str, str, str, str]]] = {}
    order: list[str] = []
    for d, ticker, company, st in rows:
        try:
            label = _date.fromisoformat(d).strftime("%B %Y")
        except ValueError:
            continue
        header = f"## {label}"
        if header not in months:
            months[header] = []
            order.append(header)
        months[header].append((d, ticker, company, st))
    lines = ["## Earnings Radar", ""]
    for header in order:
        items = sorted(months[header], key=lambda r: (r[0], r[1]))
        lines.append(header)
        lines.append("")
        for d, ticker, company, st in items:
            try:
                day = _date.fromisoformat(d).strftime("%b %d")
            except ValueError:
                day = d
            q = _earnings_quarter(d)
            lines.append(f"- {day} — {ticker}, {company} — {q} ({st})")
        lines.append("")
    return "\n".join(lines).rstrip()


class AIContextBuilder:
    """Builds a separate context file for AI with all detailed data."""
    
    def __init__(self):
        pass
    
    def build_context_file(self, result: RegimeResult, t212_data: dict = None, 
                           news_items: list = None, earnings_data: dict = None) -> str:
        """Build comprehensive context file for AI (not shown in main report)."""
        sections = [
            "# AI Context File - Detailed Market Data",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## EXI2 Regime Analysis",
            f"Regime: {result.regime} ({result.confidence:.0%})",
            f"Primary Signal: {result.primary_signal}",
            "",
            "## Multi-Timeframe Indicators",
        ]
        
        for tf_name, tf_data in result.timeframes.items():
            ind = tf_data.get("indicators", {})
            if ind:
                sections.append(f"\n### {tf_name.upper()}")
                for k, v in sorted(ind.items()):
                    sections.append(f"  {k}: {v}")
        
        sections.extend([
            "",
            "## Price Structure",
            f"Trend: {result.price_structure.current_trend}",
            f"Quality: {result.price_structure.trend_quality:.0%}",
            f"Support: €{result.price_structure.nearest_support:.2f}" if result.price_structure.nearest_support is not None else "Support: N/A",
            f"Resistance: €{result.price_structure.nearest_resistance:.2f}" if result.price_structure.nearest_resistance is not None else "Resistance: N/A",
            "",
            "## Key Levels",
        ])
        
        for level in result.price_structure.key_levels.get("support", [])[:5]:
            sections.append(f"  Support €{level.price:.2f} (strength: {level.strength})")
        for level in result.price_structure.key_levels.get("resistance", [])[:5]:
            sections.append(f"  Resistance €{level.price:.2f} (strength: {level.strength})")
        
        sections.extend([
            "",
            "## News Sentiment",
            f"Sentiment: {result.news_sentiment.get('sentiment')}",
            f"Score: {result.news_sentiment.get('score')}/100",
            f"Count: {result.news_sentiment.get('count')}",
            f"Topics: {', '.join(result.news_sentiment.get('key_topics', []))}",
        ])
        
        if t212_data:
            sections.extend(["", "## Trading212 Portfolio"])
            summary = t212_data.get("account_summary", {})
            sections.append(f"  Total Equity (broker): €{summary.get('total_equity', 0):,.2f}")
            sections.append(f"  Free Cash: €{t212_data.get('cash', {}).get('free', 0):,.2f}")
            sections.append(f"  Pie Cash: €{t212_data.get('cash', {}).get('pie_cash', 0):,.2f}")
            sections.append(f"  Validated positions market value: €{summary.get('invested', 0):,.2f}")
            recon_status = summary.get("reconciliation_status", "UNKNOWN")
            sections.append(f"  Reconciliation: {recon_status}")
            if recon_status == "PASS":
                sections.append(f"  Open Position Unrealized P&L (T212): €{summary.get('unrealized_pnl', 0):,.2f}")
                sections.append(f"  Open Position Unrealized P&L (recalc): €{summary.get('unrealized_pnl_calc', 0):,.2f}")
            else:
                sections.append("  ⚠️ Account performance unreconciled — derived return withheld.")
                sections.append("  (FAIL = broker total vs. súčet pozícií + cash sa líšia; pozície jednotlivo sú OK.)")
            sections.append("  Legenda: SYMBOL: qty @ avg_cena = hodnota (P&L %) [validation: kontrola qty vs. hodnota]")
            for pos in t212_data.get("positions", [])[:20]:
                val_eur = pos.get('value_eur', pos.get('value', 0))
                avg_price_eur = pos.get('average_price_eur', pos.get('avg_price', 0))
                pnl_pct = pos.get('pnl_pct', 0)
                validation_status = pos.get('validation_status', 'UNKNOWN')
                sections.append(f"  {pos.get('symbol')}: {pos.get('quantity')} @ €{avg_price_eur:.2f} = €{val_eur:,.2f} (P&L: {pnl_pct:+.1f}%) [validation: {validation_status}]")
        
        if news_items:
            sections.extend(["", "## Recent News (with URLs)"])
            for item in news_items[:20]:
                sections.append(f"  - [{item.get('title')}]({item.get('url')}) — {item.get('source')} — {item.get('published')}")
        
        if earnings_data:
            sections.extend(["", "## Earnings Calendar"])
            for sym, data in earnings_data.items():
                sections.append(f"  {sym}: {data}")
        
        return "\n".join(sections)


# =============================================================================
# NEW: AI Context Report Generator (machine-safe, broker-first)
# =============================================================================

class AIContextReportBuilder:
    """Generates a machine-safe AI context report using the broker-first model.

    This report NEVER mixes broker EUR values with external Yahoo prices in
    other currencies. It enforces strict namespace separation:
      A. BROKER_PORTFOLIO — broker-first EUR valuations only
      B. EXTERNAL_RESEARCH — external prices with explicit currency/source/mapping
      C. WATCHLIST_ONLY — research-only signals, never affect account values
    """

    VALUATION_ENGINE_VERSION = "broker-first-v1"

    MAPPING_SUSPECT_PAIRS = {
        ("VWSB", "VWS.CO"): "VWSB broker EUR (~28.42) vs VWS.CO DKK (~213) — MAPPING_SUSPECT",
        ("LITMM", "LIT"): "LITMM EUR vs LIT USD — MAPPING_SUSPECT",
        ("SYNL", "SYN"): "SYNL GBX vs unrelated EUR external — MAPPING_SUSPECT",
    }

    def __init__(self, max_timestamp_spread_seconds: int = 300):
        self.max_timestamp_spread_seconds = max_timestamp_spread_seconds

    def build_report(
        self,
        *,
        regime_result: RegimeResult,
        t212_data: dict | None,
        portfolio_rows: list[dict] | None,
        technicals: dict | None,
        yahoo_map: dict | None,
        external_research: dict | None = None,
        watchlist: list[dict] | None = None,
        reconciliation: ReconciliationResult | None = None,
        account_snapshot: AccountSnapshot | None = None,
        fx_info: dict | None = None,
        catalog_meta: dict | None = None,
        run_id: str = "",
        model_info: dict | None = None,  # New: model attribution info
        news_context_path: str | None = None,
    ) -> str:
        """Build the complete AI context report."""
        lines: list[str] = []

        # 1. Immutable run metadata
        lines.append(self._build_run_metadata(
            regime_result, t212_data, fx_info, catalog_meta, reconciliation, run_id))

        # 2. Account safety header
        lines.append(self._build_account_safety_header(reconciliation))

        # 3. Namespace A: BROKER_PORTFOLIO
        if portfolio_rows:
            lines.append(self._build_broker_portfolio_namespace(
                portfolio_rows, t212_data, reconciliation, account_snapshot))

        # 4. Namespace B: EXTERNAL_RESEARCH
        lines.append(self._build_external_research_namespace(
            technicals, yahoo_map, portfolio_rows, external_research))

        # 5. Namespace C: WATCHLIST_ONLY
        if watchlist:
            lines.append(self._build_watchlist_namespace(watchlist))

        # 6. Data limitations
        lines.append(self._build_data_limitations(
            portfolio_rows, technicals, yahoo_map))

        # 7. Snapshot consistency validation
        lines.append(self._build_snapshot_consistency(
            t212_data, fx_info, account_snapshot))

        # 8. Model Attribution & Sources (NEW)
        if model_info:
            lines.append(self._build_model_attribution(model_info))

        # 9. News context file path
        if news_context_path:
            lines.append("")
            lines.append("## News Context")
            lines.append("")
            lines.append(f"News context file: `{news_context_path}`")
            lines.append("Contains all news sources: portfolio holdings, Slovak news, Reddit, Trump policy watch, commodities/crypto, analyst recommendations.")

        return "\n".join(lines).rstrip() + "\n"

    # -------------------------------------------------------------------------
    # Run metadata
    # -------------------------------------------------------------------------
    def _build_run_metadata(
        self,
        regime_result: RegimeResult,
        t212_data: dict | None,
        fx_info: dict | None,
        catalog_meta: dict | None,
        reconciliation: ReconciliationResult | None,
        run_id: str,
    ) -> str:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        account_ts = ""
        positions_ts = ""
        if t212_data and isinstance(t212_data.get("account_summary"), dict):
            summary = t212_data["account_summary"]
            account_ts = summary.get("retrieved_at", "")
            positions_ts = summary.get("positions_retrieved_at", account_ts)
        fx_ts = fx_info.get("retrieved_at") if fx_info else ""
        fx_source = fx_info.get("source", "") if fx_info else ""
        catalog_ver = catalog_meta.get("fetched_at", "") if catalog_meta else ""
        catalog_age = catalog_meta.get("age_days", "") if catalog_meta else ""
        catalog_count = catalog_meta.get("count", "") if catalog_meta else ""
        recon_status = reconciliation.reconciliation_status if reconciliation else "UNKNOWN"
        recon_delta = reconciliation.reconciliation_delta_eur if reconciliation else 0.0

        lines = [
            "---",
            "run_metadata:",
            f"  run_id: \"{run_id or now}\"",
            f"  generated_at_utc: \"{now}\"",
            f"  broker_account_snapshot_timestamp: \"{account_ts or 'unavailable'}\"",
            f"  positions_snapshot_timestamp: \"{positions_ts or 'unavailable'}\"",
            f"  fx_timestamp: \"{fx_ts or 'unavailable'}\"",
            f"  fx_source: \"{fx_source or 'unavailable'}\"",
            f"  instrument_catalog_version: \"{catalog_ver or 'unavailable'}\"",
            f"  instrument_catalog_age_days: {catalog_age if catalog_age else 'unavailable'}",
            f"  instrument_catalog_count: {catalog_count if catalog_count else 'unavailable'}",
            f"  reconciliation_status: \"{recon_status}\"",
            f"  reconciliation_delta_eur: {recon_delta:.2f}",
            f"  valuation_engine_version: \"{self.VALUATION_ENGINE_VERSION}\"",
            "---",
            "",
        ]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Account safety header
    # -------------------------------------------------------------------------
    def _build_account_safety_header(
        self,
        reconciliation: ReconciliationResult | None,
    ) -> str:
        if reconciliation and reconciliation.reconciliation_status != "PASS":
            lines = [
                "ACCOUNT_SAFETY_STATE: DATA_QUALITY_FAIL",
                "TRADE_DEPLOYMENT: BLOCKED",
                "instruction: external signals are research-only; no BUY/SELL/ADD/REDUCE output for held positions.",
                "",
            ]
        else:
            lines = [
                "ACCOUNT_SAFETY_STATE: OK",
                "TRADE_DEPLOYMENT: ALLOWED",
                "instruction: normal operation — canonical signals apply.",
                "",
            ]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Namespace A: BROKER_PORTFOLIO
    # -------------------------------------------------------------------------
    def _build_broker_portfolio_namespace(
        self,
        portfolio_rows: list[dict],
        t212_data: dict | None,
        reconciliation: ReconciliationResult | None,
        account_snapshot: AccountSnapshot | None,
    ) -> str:
        lines = ["# BROKER_PORTFOLIO", ""]

        if reconciliation:
            lines.append("## Account Summary (broker-first EUR)")
            lines.append(f"- broker_total_equity_eur: €{reconciliation.broker_total_equity_eur:,.2f}")
            lines.append(f"- reported_free_cash_eur: €{reconciliation.reported_free_cash_eur:,.2f}")
            lines.append(f"- reported_blocked_cash_eur: €{reconciliation.reported_blocked_cash_eur:,.2f}")
            lines.append(f"- pie_cash_eur: €{reconciliation.pie_cash_eur:,.2f}")
            lines.append(f"- total_reported_cash_eur: €{reconciliation.total_reported_cash_eur:,.2f}")
            lines.append(f"- expected_open_positions_value_eur: €{reconciliation.expected_open_positions_value_eur:,.2f}")
            lines.append(f"- sum_deduplicated_broker_position_values_eur: €{reconciliation.sum_deduplicated_broker_position_values_eur:,.2f}")
            lines.append(f"- reconciliation_delta_eur: €{reconciliation.reconciliation_delta_eur:+,.2f}")
            lines.append(f"- tolerance_eur: €{reconciliation.tolerance_eur:,.2f}")
            lines.append(f"- reconciliation_status: {reconciliation.reconciliation_status}")
            lines.append(f"- data_quality: {reconciliation.data_quality}")
            lines.append("")

        if account_snapshot and account_snapshot.positions:
            lines.append("## Positions (broker-first, deduplicated by ISIN)")
            lines.append(
                "| broker_instrument_id | display_symbol | isin | instrument_currency | "
                "raw_quantity | raw_average_price | raw_current_price | price_unit | minor_factor | "
                "normalized_average_price | normalized_current_price | broker_market_value_eur | "
                "valuation_source | mapping_status | included_in_position_total | exclusion_reason |")
            lines.append(
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

            for pos in account_snapshot.positions:
                if not pos.included_in_position_total:
                    continue
                ins = pos.instrument
                fx = pos.fx_audit
                lines.append(
                    f"| {ins.broker_instrument_id} | {ins.display_symbol} | {ins.isin or ''} | "
                    f"{ins.currency} | {pos.quantity:.6f} | "
                    f"{pos.average_price_raw if pos.average_price_raw is not None else ''} | "
                    f"{pos.current_price_raw if pos.current_price_raw is not None else ''} | "
                    f"{fx.source_currency} | {fx.minor_unit_factor} | "
                    f"{fx.normalized_price if fx.normalized_price is not None else ''} | "
                    f"{fx.normalized_price if fx.normalized_price is not None else ''} | "
                    f"{pos.broker_market_value_eur if pos.broker_market_value_eur is not None else ''} | "
                    f"{pos.valuation_source} | {pos.mapping_status} | "
                    f"{'yes' if pos.included_in_position_total else 'no'} | {pos.exclusion_reason or '—'} |")

        lines.append("")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Namespace B: EXTERNAL_RESEARCH
    # -------------------------------------------------------------------------
    def _build_external_research_namespace(
        self,
        technicals: dict | None,
        yahoo_map: dict | None,
        portfolio_rows: list[dict] | None,
        external_research: dict | None,
    ) -> str:
        lines = ["# EXTERNAL_RESEARCH", ""]
        lines.append("All external prices, indicators, and levels below are from Yahoo Finance (or other "
                     "external sources). They are NEVER used for broker position valuation. "
                     "Every value displays its native quote currency, source, and mapping status.")
        lines.append("")

        if not technicals and not external_research:
            lines.append("(no external research data available)")
            lines.append("")
            return "\n".join(lines)

        # Build set of held display symbols
        held_displays = set()
        if portfolio_rows:
            for r in portfolio_rows:
                disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
                if disp:
                    held_displays.add(disp)

        # Technical indicators per symbol
        if technicals:
            lines.append("## Technical Indicators (Yahoo Finance, 3mo daily)")
            lines.append(
                "| external_symbol | source | quote_currency | quote_exchange | mapping_status | "
                "price | RSI_14 | SMA_20 | SMA_50 | MACD | Support | Resistance |")
            lines.append(
                "|---|---|---|---|---|---|---|---|---|---|---|---|")

            for disp, ind in sorted(technicals.items()):
                if not isinstance(ind, dict):
                    continue
                ext_sym = yahoo_map.get(disp, disp) if yahoo_map else disp
                mapping_status = self._get_mapping_status(disp, ext_sym, held_displays)
                quote_ccy, venue = self._infer_quote_currency_and_venue(ext_sym)
                price = ind.get("price")
                lines.append(
                    f"| {ext_sym} | Yahoo Finance | {quote_ccy} | {venue} | {mapping_status} | "
                    f"{self._fmt_val(price, quote_ccy)} | "
                    f"{self._fmt_val(ind.get('RSI_14'), quote_ccy)} | "
                    f"{self._fmt_val(ind.get('SMA_20'), quote_ccy)} | "
                    f"{self._fmt_val(ind.get('SMA_50'), quote_ccy)} | "
                    f"{self._fmt_val(ind.get('MACD'), quote_ccy)} | "
                    f"{self._fmt_val(ind.get('Support'), quote_ccy)} | "
                    f"{self._fmt_val(ind.get('Resistance'), quote_ccy)} |")

        # External research signals (if any)
        if external_research:
            lines.append("")
            lines.append("## External Research Signals")
            for key, val in sorted(external_research.items()):
                lines.append(f"- {key}: {val}")

        lines.append("")
        return "\n".join(lines)

    def _get_mapping_status(self, display: str, external_symbol: str, held_displays: set) -> str:
        if display in held_displays:
            # Check for explicit mapping suspect pairs
            for (broker, ext), msg in self.MAPPING_SUSPECT_PAIRS.items():
                if display == broker and external_symbol == ext:
                    return f"MAPPING_SUSPECT ({msg})"
            if external_symbol and external_symbol != display:
                return "VERIFIED"
            return "VERIFIED"
        return "WATCHLIST_ONLY"

    def _infer_quote_currency_and_venue(self, symbol: str) -> tuple[str, str]:
        """Infer quote currency and venue from Yahoo symbol."""
        sym = symbol.upper()
        if sym.endswith(".DE") or sym.endswith(".F") or sym.endswith(".MU") or sym.endswith(".HM"):
            return "EUR", "XETRA"
        if sym.endswith(".L"):
            return "GBX", "LSE"
        if sym.endswith(".CO"):
            return "DKK", "CPH"
        if sym.endswith(".AS"):
            return "EUR", "AMS"
        if sym.endswith(".MI"):
            return "EUR", "MIL"
        if sym.endswith(".PA"):
            return "EUR", "PAR"
        if sym.endswith(".BR"):
            return "EUR", "BRU"
        if sym.endswith(".OL"):
            return "NOK", "OSL"
        if sym.endswith(".HE"):
            return "EUR", "HEL"
        if sym.endswith(".ST"):
            return "SEK", "STO"
        if sym.endswith(".VI"):
            return "EUR", "VIE"
        if sym.endswith(".SW"):
            return "CHF", "SWX"
        if sym.endswith(".KS"):
            return "KRW", "KRX"
        if sym.endswith(".IC"):
            return "EUR", "ICE"
        # Default US
        return "USD", "NASDAQ/NYSE"

    def _fmt_val(self, val: Any, quote_ccy: str) -> str:
        if val is None:
            return "n/a"
        try:
            f = float(val)
            # Never prefix with € unless quote currency is EUR
            if quote_ccy == "EUR":
                return f"€{f:,.2f}"
            if quote_ccy == "GBX":
                return f"{f:,.2f} GBX"
            return f"{f:,.2f} {quote_ccy}"
        except (TypeError, ValueError):
            return "n/a"

    # -------------------------------------------------------------------------
    # Namespace C: WATCHLIST_ONLY
    # -------------------------------------------------------------------------
    def _build_watchlist_namespace(self, watchlist: list[dict]) -> str:
        lines = ["# WATCHLIST_ONLY", ""]
        lines.append("Research signals for instruments NOT held in the broker account. "
                     "These are RESEARCH_ONLY and must never affect account values or broker-position decisions.")
        lines.append("")
        lines.append(
            "| ticker | signal | horizon | catalyst | key_level | NOT_A_TRADE_INSTRUCTION |")
        lines.append(
            "|---|---|---|---|---|---|")

        for item in watchlist:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", ""))
            raw_signal = str(item.get("signal", "HOLD")).upper()
            signal = self._rename_signal(raw_signal)
            horizon = str(item.get("horizon", ""))
            catalyst = str(item.get("catalyst", item.get("reason", "")))
            key_level = str(item.get("key_level", item.get("support_resistance", "")))
            lines.append(f"| {ticker} | {signal} | {horizon} | {catalyst} | {key_level} | YES |")

        lines.append("")
        return "\n".join(lines)

    def _rename_signal(self, raw_signal: str) -> str:
        """Rename BUY->RESEARCH_BULLISH, SELL->RESEARCH_BEARISH, etc."""
        mapping = {
            "BUY": "RESEARCH_BULLISH",
            "ACCUMULATE": "RESEARCH_BULLISH",
            "ADD": "RESEARCH_BULLISH",
            "SELL": "RESEARCH_BEARISH",
            "TRIM": "RESEARCH_BEARISH",
            "REDUCE": "RESEARCH_BEARISH",
            "HOLD": "RESEARCH_WAIT",
            "WAIT": "RESEARCH_WAIT",
            "WATCH": "RESEARCH_WATCH",
            "NEUTRAL": "RESEARCH_WAIT",
        }
        return mapping.get(raw_signal, "RESEARCH_WAIT")

    # -------------------------------------------------------------------------
    # Data limitations
    # -------------------------------------------------------------------------
    def _build_data_limitations(
        self,
        portfolio_rows: list[dict] | None,
        technicals: dict | None,
        yahoo_map: dict | None,
    ) -> str:
        lines = ["# Data limitations", ""]

        unresolved_mappings = []
        broker_only = []
        excluded_from_technical = []
        gbx_instruments = []

        held = set()
        if portfolio_rows:
            for r in portfolio_rows:
                disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
                if disp:
                    held.add(disp)

            for disp in held:
                ext = yahoo_map.get(disp) if yahoo_map else None
                # Unresolved = no mapping at all (None or missing from map)
                if not ext:
                    if disp not in ["EUR", "CASH"]:
                        unresolved_mappings.append(disp)
                else:
                    # Has a mapping - check if technicals are available
                    if disp not in (technicals or {}):
                        if disp not in ["EUR", "CASH"]:
                            broker_only.append(disp)

        if technicals and yahoo_map:
            for disp, ext in yahoo_map.items():
                if disp in held:
                    mapping_status = self._get_mapping_status(disp, ext, set())
                    if mapping_status.startswith("MAPPING_SUSPECT"):
                        excluded_from_technical.append(f"{disp} ({ext}) — {mapping_status}")

        if portfolio_rows:
            for r in portfolio_rows:
                disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
                quote_ccy = str(r.get("quote_currency", "")).strip().upper()
                price_unit = str(r.get("price_unit", "")).strip().upper()
                if quote_ccy in ("GBX", "GBXP", "PENCE", "P") or price_unit in ("GBX", "GBXP", "PENCE", "P"):
                    factor = r.get("minor_factor", 100)
                    gbx_instruments.append(f"{disp} (price in pence, factor ÷{factor})")

        if unresolved_mappings:
            lines.append("## Unresolved mappings (broker instrument → no external symbol)")
            for m in sorted(unresolved_mappings):
                lines.append(f"- {m}")
            lines.append("")

        if broker_only:
            lines.append("## Broker-only symbols (mapped but no external market data)")
            for m in sorted(broker_only):
                lines.append(f"- {m}")
            lines.append("")

        if excluded_from_technical:
            lines.append("## Positions excluded from technical decisions (mapping suspect)")
            for m in sorted(excluded_from_technical):
                lines.append(f"- {m}")
            lines.append("")

        if gbx_instruments:
            lines.append("## GBX minor-unit instruments (raw price in pence, normalized ÷100)")
            for m in sorted(gbx_instruments):
                lines.append(f"- {m}")
            lines.append("")

        if not (unresolved_mappings or broker_only or excluded_from_technical or gbx_instruments):
            lines.append("(no data limitations identified)")
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Snapshot consistency validation
    # -------------------------------------------------------------------------
    def _build_snapshot_consistency(
        self,
        t212_data: dict | None,
        fx_info: dict | None,
        account_snapshot: AccountSnapshot | None,
    ) -> str:
        lines = ["# Snapshot consistency validation", ""]

        timestamps = []
        if t212_data and isinstance(t212_data.get("account_summary"), dict):
            ts = t212_data["account_summary"].get("retrieved_at")
            if ts:
                timestamps.append(("account_summary", ts))
            ts = t212_data["account_summary"].get("positions_retrieved_at")
            if ts:
                timestamps.append(("positions", ts))

        if fx_info:
            ts = fx_info.get("retrieved_at")
            if ts:
                timestamps.append(("fx", ts))

        if account_snapshot:
            ts = account_snapshot.retrieved_at
            if ts:
                timestamps.append(("account_snapshot", ts))

        unstable = False
        if len(timestamps) >= 2:
            from datetime import datetime
            parsed = []
            for label, ts in timestamps:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    parsed.append((label, dt))
                except (ValueError, AttributeError):
                    pass
            if len(parsed) >= 2:
                first = min(parsed, key=lambda x: x[1])
                last = max(parsed, key=lambda x: x[1])
                spread = (last[1] - first[1]).total_seconds()
                lines.append(f"- timestamp spread: {spread:.0f} seconds "
                             f"(max allowed: {self.max_timestamp_spread_seconds}s)")
                if spread > self.max_timestamp_spread_seconds:
                    unstable = True
                    lines.append(f"- STATUS: SNAPSHOT_TIMING_UNSTABLE (spread {spread:.0f}s > {self.max_timestamp_spread_seconds}s)")
                    lines.append("- WARNING: avoid producing portfolio-level conclusions from inconsistent snapshots")
                else:
                    lines.append("- STATUS: STABLE")
        else:
            lines.append("- insufficient timestamps for validation")

        lines.append("")
        return "\n".join(lines)


# -------------------------------------------------------------------------
    # Model Attribution Section (NEW)
    # -------------------------------------------------------------------------
    def _build_model_attribution(self, model_info: dict) -> str:
        """Build the model attribution section showing which model did what."""
        lines = [
            "# MODEL ATTRIBUTION & SOURCES",
            "",
            "This section documents which AI model was used for each stage,",
            "what was computed by Python (deterministic), and the fallback chain.",
            "",
        ]

        # Model info
        decision_model = model_info.get("decision_model", "deterministic_fallback")
        summary_model = model_info.get("summary_model", "deterministic_fallback")
        discovery_model = model_info.get("discovery_model", "deterministic_fallback")
        decision_detail_model = model_info.get("decision_detail_model", "deterministic_fallback")
        aggressiveness = model_info.get("ai_aggressiveness", "balanced")

        # Decision & Reasoning
        lines.append("## Decision & Reasoning")
        lines.append(f"- **Decision/Recommendation**: {self._format_model_name(decision_model)}")
        lines.append(f"- **Reasoning/Thinking**: {self._format_model_name(decision_detail_model)} (finr1 for math, gpt-oss/qwen for reasoning)")
        lines.append(f"- **Summary Generation**: {self._format_model_name(summary_model)} (gemma with fallback)")
        lines.append(f"- **Discovery/New Ideas**: {self._format_model_name(discovery_model)}")
        lines.append(f"- **Aggressiveness Profile**: {aggressiveness}")

        # Analyst Recommendations & External Sources
        lines.append("")
        lines.append("## Analyst Recommendations & External Sources")
        lines.append("- **Analyst Consensus**: Available via yfinance (when available)")
        lines.append("- **Finr1**: Math reasoning model for quantitative analysis")
        lines.append("- **Technical Indicators**: Python (yfinance + Playwright fallback)")

        # Python-sourced calculations
        lines.append("")
        lines.append("## Python-Sourced Calculations (Deterministic)")
        lines.append("- **Reconciliation & Valuation**: Python (broker-first, no LLM)")
        lines.append("- **Technical Indicators (RSI, MACD, SMA, ATR, Support/Resistance)**: Python (yfinance + Playwright fallback)")
        lines.append("- **Position Valuation & P&L**: Python (broker-first EUR valuation)")
        lines.append("- **Reconciliation Delta & Thresholds**: Python (strict tolerance)")
        lines.append("- **Position Weights & Concentration**: Python")
        lines.append("- **Earnings Dates**: Python (yfinance + calendar classification)")

        # Model Fallback Chain
        lines.append("")
        lines.append("## Model Fallback Chain")
        lines.append("- **Primary**: LM Studio (local, gpt-oss-20b / qwen3.8-9b / gemma4-12b)")
        lines.append("- **Fallback 1**: llama.cpp (Raspberry Pi, qwen3.8-9b)")
        lines.append("- **Fallback 2**: Ollama (Raspberry Pi, qwen3.8-9b-pi / qwen3-4b-pi / noema-2b)")
        lines.append("- **Fallback 3**: Gemini (free tier, gemini-2.5-flash)")
        lines.append("- **Fallback 4**: Mistral (free tier, open-mistral-nemo)")
        lines.append("- **Fallback 5**: Deterministic Python (always available)")

        return "\n".join(lines)

    def _format_model_name(self, model: str) -> str:
        """Format model name for display."""
        if not model or model == "deterministic_fallback":
            return "Python (deterministic)"
        model_lower = model.lower()
        if "gpt-oss" in model_lower:
            return "gpt-oss (OpenAI OSS)"
        elif "qwen" in model_lower:
            return "qwen (Alibaba)"
        elif "gemma" in model_lower:
            return "gemma (Google)"
        elif "mistral" in model_lower:
            return "mistral (Mistral AI)"
        elif "gemini" in model_lower:
            return "gemini (Google)"
        elif "finr1" in model_lower:
            return "finr1 (math reasoning)"
        elif "qwen" in model_lower and ("3.8" in model_lower or "9b" in model_lower):
            return "qwen3.8-9b (Alibaba)"
        else:
            return model


def generate_regime_markdown(
    result: RegimeResult,
    t212_data: dict = None,
    ai_recs: dict = None,
    earnings: dict | None = None,
    technicals: dict | None = None,
    names: dict | None = None,
    known_clean: set | None = None,
    canonical: dict | None = None,
    yahoo_map: dict | None = None,
) -> str:
    """Convenience function for quick report generation."""
    generator = RegimeReportGenerator()
    return generator.generate(result, t212_data, ai_recs, earnings, technicals, names, known_clean, canonical, yahoo_map)
