"""Two-document reporting: decision brief (concise, daily) + T212 snapshot (full).

The brief never renders a full holdings table and carries no sized orders;
the snapshot is the complete financial inventory. All sections are built
deterministically from canonical rows — no LLM prose enters either document.
"""

from __future__ import annotations

import re
from datetime import date as _date
from typing import Any, Dict, List, Optional

from investment_engine.portfolio.symbols import resolve_company_name
from investment_engine.reporting.regime_report import normalize_signal, signal_emoji


# Urgency weights (higher = more urgent). No hard item cap: every triggered
# position is listed, sorted by urgency descending.
_SCORE_SELL_SIGNAL = 100
_SCORE_BREACH = 80
_SCORE_EARNINGS_7D = 70
_SCORE_WEIGHT = 60
_SCORE_BREAKOUT = 55
_SCORE_PNL = 50
_SCORE_RSI = 40
_SCORE_NEAR_LEVEL = 30
_SCORE_PIE_WARNING = 25
_SCORE_UNAVAILABLE = 20


def _fnum(value: Any, fmt: str = "{:.2f}") -> str:
    try:
        return fmt.format(float(value)) if value is not None else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _feur(value: Any) -> str:
    try:
        return f"€{float(value):,.2f}" if value is not None else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _rsi_of(technicals: dict | None, display: str) -> float | None:
    try:
        ind = (technicals or {}).get(display)
        rsi = (ind or {}).get("RSI_14") if isinstance(ind, dict) else None
        return float(rsi) if rsi is not None else None
    except (TypeError, ValueError):
        return None


def _days_to_earnings(earnings_status: dict | None, display: str, report_date) -> tuple:
    """(days, date_iso) when an earnings date exists, else (None, '')."""
    try:
        val = (earnings_status or {}).get(display)
        d = val[0][:10] if isinstance(val, (tuple, list)) and val and isinstance(val[0], str) else None
        if not d:
            return None, ""
        delta = (_date.fromisoformat(d) - report_date).days
        return delta, d
    except (ValueError, TypeError):
        return None, ""


def build_monitoring_items(
    rows: list[dict] | None,
    *,
    earnings_status: dict | None = None,
    technicals: dict | None = None,
    pie_warnings: dict | None = None,
    recon_status: str = "UNKNOWN",
    report_date=None,
    canonical_map: dict | None = None,
) -> list[dict]:
    """Build urgency-sorted monitoring items for every triggered owned position.

    No item cap. In FAIL state owned positions stay canonical HOLD unless a
    risk-reduction SELL rule fires (weight>=5% or P&L<=-8% or price below
    support); urgent HOLDs present as REVIEW (presentation-only).
    
    Technical comparisons (support/resistance breach) are ONLY performed when:
    - external_mapping_status == "VERIFIED"
    - broker_quote_currency == external_quote_currency
    - broker_current_price_native is not None
    - external_support_native is not None
    """
    from datetime import date as _d
    if report_date is None:
        try:
            from investment_engine.research.market_data import report_now_bta
            report_date = report_now_bta().date()
        except Exception:
            report_date = _d.today()
    fail = str(recon_status or "").upper() != "PASS"
    items: list[dict] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
        if not disp or disp == "UNKNOWN":
            continue
        canon_sig = normalize_signal(r.get("signal", "HOLD"))
        entry = (canonical_map or {}).get(disp) if isinstance(canonical_map, dict) else None
        if isinstance(entry, dict) and entry.get("signal") in ("BUY", "SELL", "HOLD"):
            canon_sig = entry["signal"]
        triggers: list[tuple[int, str]] = []
        if canon_sig == "SELL":
            triggers.append((_SCORE_SELL_SIGNAL, "canonical SELL/REDUCE/TRIM signal"))
        try:
            weight = float(r.get("weight", 0) or 0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight >= 5.0:
            triggers.append((_SCORE_WEIGHT, f"weight {weight:.1f}% ≥ 5%"))
        try:
            pnl = float(r.get("pnl_pct", 0) or 0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl <= -8.0 or pnl >= 15.0:
            triggers.append((_SCORE_PNL, f"P&L {pnl:+.1f}% outside −8%/+15% band"))
        rsi = _rsi_of(technicals, disp)
        if rsi is not None and (rsi <= 25 or rsi >= 75):
            triggers.append((_SCORE_RSI, f"RSI {rsi:.1f} extreme"))
        
        # --- Technical comparison gate ---
        # Only compare if mapping is VERIFIED and currencies match
        ext_mapping_status = str(r.get("external_mapping_status", r.get("market_data_state", ""))).upper()
        broker_quote_ccy = str(r.get("quote_currency", r.get("broker_quote_currency", ""))).upper()
        ext_quote_ccy = str(r.get("external_quote_currency", "")).upper()
        # Get native prices (not EUR-converted)
        broker_price_native = r.get("current_price_native")
        if broker_price_native is None:
            # Fallback: if broker_price_eur and fx_rate available, reverse-calculate
            # But prefer native price field
            pass
        ext_support_native = r.get("support")
        ext_resistance_native = r.get("resistance")
        sup = None
        res = None
        
        technical_levels_allowed = (
            ext_mapping_status == "VERIFIED"
            and broker_quote_ccy
            and broker_quote_ccy == ext_quote_ccy
            and broker_price_native is not None
            and ext_support_native is not None
        )
        
        if technical_levels_allowed:
            try:
                px = float(broker_price_native)
                sup = float(ext_support_native) if ext_support_native is not None else None
                res = float(ext_resistance_native) if ext_resistance_native is not None else None
                
                breached = False
                if sup is not None and px > 0 and px < sup:
                    triggers.append((_SCORE_BREACH, f"stop-loss/invalidation breach: price {px:.2f} {broker_quote_ccy} below support {sup:.2f} {broker_quote_ccy}"))
                    breached = True
                elif res is not None and px > 0 and px > res:
                    triggers.append((_SCORE_BREAKOUT, f"breakout above resistance {res:.2f} {broker_quote_ccy}"))
                
                for lvl, lname in ((sup, "support"), (res, "resistance")):
                    try:
                        if lvl is not None and px > 0 and not breached and abs(px - lvl) / px <= 0.015:
                            triggers.append((_SCORE_NEAR_LEVEL, f"price within 1.5% of {lname} {lvl:.2f} {broker_quote_ccy}"))
                            break
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
            except (TypeError, ValueError):
                pass
        else:
            # Technical levels not comparable - add REVIEW_MAPPING flag
            if ext_mapping_status in ("MAPPING_SUSPECT", "UNRESOLVED", "BROKER_ONLY", "NOT_REQUIRED"):
                triggers.append((_SCORE_UNAVAILABLE, f"technical levels suppressed: mapping_status={ext_mapping_status}"))
        
        delta, earn_date = _days_to_earnings(earnings_status, disp, report_date)
        if delta is not None and 0 <= delta <= 7:
            triggers.append((_SCORE_EARNINGS_7D, f"earnings in {delta}d ({earn_date})"))
        if str(r.get("market_data_state", "") or "").upper() == "UNRESOLVED" and weight >= 2.0:
            triggers.append((_SCORE_UNAVAILABLE, f"market data unavailable (weight {weight:.1f}%)"))
        pie_note = (pie_warnings or {}).get(disp)
        if pie_note:
            triggers.append((_SCORE_PIE_WARNING, str(pie_note)))
        if not triggers:
            continue
        urgency = max(s for s, _ in triggers)
        why = "; ".join(t for _, t in sorted(triggers, key=lambda x: -x[0]))
        # Presentation status.
        # `breached` is only defined when technical_levels_allowed is True
        breached = False  # default
        risk_sell_rule = canon_sig == "SELL" and (weight >= 5.0 or pnl <= -8.0 or breached)
        if canon_sig == "SELL" and (not fail or risk_sell_rule):
            presentation = "SELL"
            action = (f"Risk-reduction review: consider trimming {disp} per stop discipline; "
                      f"no quantity in brief." if not fail else
                      f"Risk-reduction candidate under FAIL: manual review only, no automatic action.")
        elif fail:
            presentation = "REVIEW"
            action = "Review only; no transaction authorized."
        else:
            presentation = "HOLD"
            if canon_sig == "BUY":
                action = "Accumulation candidate — no sized order in this brief."
            else:
                action = "No action; hold position."
        try:
            mval = float(r.get("market_value", 0) or 0)
        except (TypeError, ValueError):
            mval = 0.0
        items.append({
            "display": disp,
            "company": str(r.get("company") or r.get("company_name") or disp),
            "presentation": presentation,
            "canonical": canon_sig,
            "triggers": why,
            "urgency": urgency,
            "market_value": mval,
            "weight": weight,
            "support": sup,
            "resistance": res,
            "earnings_date": earn_date if (delta is not None and 0 <= delta <= 7) else "",
            "action": action,
        })
    items.sort(key=lambda i: (-i["urgency"], -i["market_value"], i["display"]))
    return items


def extract_validated_ideas(
    candidate_section: str | None,
    news_by_symbol: dict | None,
    resolver=None,
) -> list[dict]:
    """Structured watchlist ideas: validated ticker + evidence + source URL.

    A bullet passes with a resolver-verified ticker (supported market data),
    at least one headline evidence with a valid URL, and a stated reason.
    Confidence high with ≥2 evidences, else medium.
    """
    headlines: list[tuple[str, str, str]] = []
    for _sym, entries in (news_by_symbol or {}).items():
        for e in entries or []:
            if isinstance(e, dict) and e.get("title") and e.get("url"):
                headlines.append((str(e["title"]), str(e["url"])))
    found: dict[str, dict] = {}
    for line in (candidate_section or "").splitlines():
        s = line.strip()
        if not s.startswith(("-", "*")):
            continue
        tickers = [t for t in re.findall(r"\b[A-Z]{2,6}\b", s)
                   if t not in ("BUY", "WATCH", "SELL", "HOLD", "THE", "AND", "FOR", "NOT")]
        reason = re.sub(r"^[-*]\s*", "", s)[:200]
        for t in tickers:
            if resolver is not None:
                try:
                    if not resolver(t):
                        continue
                except Exception:
                    continue
            ev = [(title, url) for title, url in headlines if t in title.upper()]
            if not ev:
                continue
            entry = found.setdefault(t, {"evidence": [], "urls": [], "reason": reason})
            for title, url in ev:
                if title not in entry["evidence"]:
                    entry["evidence"].append(title)
                if url not in entry["urls"]:
                    entry["urls"].append(url)
    ideas = []
    for t, info in sorted(found.items()):
        ideas.append({"ticker": t, "reason": info["reason"],
                      "evidence": info["evidence"][0][:140], "url": info["urls"][0],
                      "confidence": "high" if len(info["evidence"]) >= 2 else "medium"})
    return ideas[:5]


def _stance(rows, recon_status: str) -> str:
    if str(recon_status or "").upper() != "PASS":
        return "DEGRADED"
    buys = sum(1 for r in rows or [] if normalize_signal((r or {}).get("signal")) == "BUY")
    sells = sum(1 for r in rows or [] if normalize_signal((r or {}).get("signal")) == "SELL")
    if sells > 0 and sells >= buys:
        return "SELL"
    if buys > sells:
        return "BUY"
    return "HOLD"


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _mday(date_iso: str) -> str:
    """YYYY-MM-DD -> 'MMM DD' without locale dependence."""
    try:
        y, m, d = date_iso[:10].split("-")
        return f"{_MONTHS[int(m) - 1]} {int(d):02d}"
    except (ValueError, TypeError, IndexError):
        return date_iso
    if str(recon_status or "").upper() != "PASS":
        return "DEGRADED"
    buys = sum(1 for r in rows or [] if normalize_signal((r or {}).get("signal")) == "BUY")
    sells = sum(1 for r in rows or [] if normalize_signal((r or {}).get("signal")) == "SELL")
    if sells > 0 and sells >= buys:
        return "SELL"
    if buys > sells:
        return "BUY"
    return "HOLD"


def _perf_eur(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"€{float(value):+,.2f}"
    except (TypeError, ValueError):
        return "Unavailable"


def _perf_pct(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):+.4f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def _perf_alloc(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def render_brief(
    *,
    generated_at: str,
    regime_result,
    recon: dict,
    rows: list[dict],
    monitoring_items: list[dict],
    earnings_7d: dict,
    decision_news: list[dict],
    ideas: list[dict],
    cash_line: str,
    performance: dict | None = None,
) -> str:
    """Render the concise decision-first brief (exact section order)."""
    from investment_engine.accounting.cashflows import FAIL_PERFORMANCE_NOTE
    status = recon.get("status", "UNKNOWN")
    regime = getattr(regime_result, "regime", "UNKNOWN") if regime_result is not None else "UNKNOWN"
    try:
        rsi = regime_result.timeframes.get("daily", {}).get("indicators", {}).get("RSI_14", 0)
        regime_line = f"{regime} (RSI {float(rsi):.0f})"
    except Exception:
        regime_line = str(regime)
    if status == "FAIL":
        quality = (f"Data quality: **{status}** — account totals withheld; "
                   f"{len(rows or [])} validated positions shown.")
        constraint = "Reconciliation FAIL blocks all deployment and new buys."
    else:
        quality = (f"Data quality: **{status}** — reconciled; {len(rows or [])} positions.")
        top_w = max([float((r or {}).get("weight", 0) or 0) for r in (rows or [])] or [0.0])
        constraint = ("Single-name concentration caps new adds." if top_w >= 7.0
                      else "No binding constraint; normal sizing rules apply.")
    perf = performance or {}
    perf_ok = perf.get("performance_status") == "Available"
    lines = [
        "# Portfolio Decision Brief", "",
        f"Generated: {generated_at}",
        quality, "",
        "## Executive Decision", "",
        f"- Portfolio stance: **{_stance(rows, status)}**",
        f"- Market regime: **{regime_line}**",
        f"- Account reconciliation: **{status}**",
        f"- Broker total equity: {_feur(perf.get('broker_total_equity_eur', (recon or {}).get('total_equity')))}",
        f"- Net deposits: {_feur(perf.get('net_deposits_eur'))} "
        f"({perf.get('net_deposits_source', 'Unavailable — no verified cash-flow source')} / "
        f"{perf.get('net_deposits_status', 'UNAVAILABLE')})",
        f"- Net P&L after costs: {_perf_eur(perf.get('net_pnl_after_costs_eur')) if perf_ok else 'Unavailable'}",
        f"- Return on net deposits: {_perf_pct(perf.get('return_pct')) if perf_ok else 'Unavailable'}",
        f"- Reported free cash: {_feur(perf.get('reported_free_cash_eur'))}",
        f"- Total reported cash: {_feur(perf.get('reported_cash_eur'))}",
        f"- Cash allocation: {_perf_alloc(perf.get('cash_allocation_pct'))}",
        f"- Cash/deployment: {cash_line}",
        f"- Key safety constraint: {constraint}",
    ]
    if status == "FAIL":
        lines += ["", f"> {FAIL_PERFORMANCE_NOTE}"]
    lines += ["", "## Portfolio Monitoring", ""]
    if not monitoring_items:
        lines.append("No owned positions meet urgency criteria.")
    for it in monitoring_items:
        lines.append(f"- **{it['display']}**, {it['company']} — **{it['presentation']}** "
                     f"(canonical {it['canonical']}) — {it['triggers']}.")
        sr = []
        if it.get("support") is not None:
            sr.append(f"support {_fnum(it['support'])}")
        if it.get("resistance") is not None:
            sr.append(f"resistance {_fnum(it['resistance'])}")
        if sr:
            lines.append(f"  - Levels: {' / '.join(sr)}.")
        if it.get("earnings_date"):
            lines.append(f"  - Earnings: {it['earnings_date']} (within 7 days).")
        lines.append(f"  - Action: {it['action']}")
    lines += ["", "## Earnings — Next 7 Days", ""]
    window = (earnings_7d or {}).get("in_window", []) if isinstance(earnings_7d, dict) else []
    if not window:
        lines.append("No relevant earnings events in the next 7 days.")
    else:
        for ev in window:
            tag = "(confirmed)" if ev.get("status") == "confirmed" else "(estimated)"
            lines.append(f"- {_mday(ev['date'])} — {ev['display']}, {ev['company']} — earnings {tag}")
    nxt = (earnings_7d or {}).get("next_after") if isinstance(earnings_7d, dict) else None
    if isinstance(nxt, dict) and nxt.get("date"):
        tag = "(confirmed)" if nxt.get("status") == "confirmed" else "(estimated)"
        lines.append(f"Next relevant earnings after this window: {_mday(nxt['date'])} — "
                     f"{nxt['display']}, {nxt['company']} — earnings {tag}.")
    lines += ["", "## Decision-Relevant News — Last 48 Hours", ""]
    if not decision_news:
        lines.append("No decision-relevant verified news in the last 48 hours.")
    else:
        for n in decision_news:
            lines.append(f"- [{n['title']}]({n['url']}) — {n['source']} · {n.get('published', '')} ({n.get('age', '')})")
            lines.append(f"  - {n.get('why', '')}")
    lines += ["", "## Watchlist Candidates", ""]
    if status == "FAIL":
        lines.append("> Research-only watchlist; no deployment is authorized while account reconciliation is failing.")
        lines.append("")
    if not ideas:
        lines.append("No new ideas met the current evidence and validation threshold.")
    else:
        for idea in ideas:
            lines.append(f"- **{idea['ticker']}** — WATCH — {idea['reason']}")
            lines.append(f"  - Evidence: {idea['evidence']} ({idea['url']}) [confidence: {idea['confidence']}]")
    return "\n".join(lines).rstrip() + "\n"


def render_snapshot(
    *,
    generated_at: str,
    recon: dict,
    cash: dict,
    rows: list[dict],
    monitoring_by_display: dict,
    earnings_status: dict | None,
    t212_data: dict | None = None,
    performance: dict | None = None,
    ledger_metadata: dict | None = None,
) -> str:
    """Render the complete Trading212 financial snapshot."""
    from investment_engine.accounting.cashflows import FAIL_PERFORMANCE_NOTE
    status = recon.get("status", "UNKNOWN")
    perf = performance or {}
    perf_ok = perf.get("performance_status") == "Available"

    def _pval(key: str) -> str:
        return _feur(perf.get(key)) if perf_ok or perf.get(key) is not None else "Unavailable"

    lines = [
        "# T212 Portfolio Snapshot", "",
        f"Generated: {generated_at}",
        f"Data quality / reconciliation: **{status}**",
        "",
        "## Account Status", "",
        "### Broker Account Performance",
        "| Metric | Value |",
        "|---|---:|",
        f"| Broker total equity | {_feur(perf.get('broker_total_equity_eur', (recon or {}).get('total_equity')))} |",
        f"| Net deposits | {_pval('net_deposits_eur')} |",
        f"| Net deposits source | {perf.get('net_deposits_source', 'Unavailable — no verified cash-flow source')} |",
        f"| Net P&L after costs | {_perf_eur(perf.get('net_pnl_after_costs_eur')) if perf_ok else 'Unavailable'} |",
        f"| Return on net deposits | {_perf_pct(perf.get('return_pct')) if perf_ok else 'Unavailable'} |",
        f"| Reported free cash | {_feur(perf.get('reported_free_cash_eur'))} |",
        f"| Pie cash | {_feur(perf.get('pie_cash_eur'))} |",
        f"| Reported cash total | {_feur(perf.get('reported_cash_eur'))} |",
        f"| Cash allocation | {_perf_alloc(perf.get('cash_allocation_pct'))} |",
        f"| Fees recorded | {_feur(perf.get('fees_eur')) if perf.get('fees_eur') is not None else 'n/a (display only)'} |",
        f"| Interest recorded | {_feur(perf.get('interest_eur')) if perf.get('interest_eur') is not None else 'n/a (display only)'} |",
        f"| Tax recorded | {_feur(perf.get('tax_eur')) if perf.get('tax_eur') is not None else 'n/a (display only)'} |",
        "",
        "### Position Reconciliation",
        "| Metric | Value |",
        "|---|---:|",
        f"| Authoritative positions value | {_feur(recon.get('positions_value'))} |",
        f"| Implied cash from broker equity | {_feur(recon.get('implied_cash'))} |",
        f"| Reported cash | {_feur(recon.get('reported_cash'))} |",
        f"| Reconciliation delta | {_feur(recon.get('cash_delta'))} |",
        f"| Reconciliation threshold | {_feur(recon.get('threshold'))} |",
        f"| Reconciliation status | **{status}** |",
    ]
    if status == "FAIL":
        lines += ["",
                  "> Reconciliation check failed — derived account totals withheld. "
                  "No cause is asserted.",
                  "",
                  f"> {FAIL_PERFORMANCE_NOTE}"]
    lines += ["", "## Holdings", ""]
    if not rows:
        lines.append("No positions to display.")
    else:
        lines.append("| Ticker | Company | Qty | Avg Cost | Current Price | Market Value | "
                     "Unrealized P&L | Realized P&L | Total P&L | P&L % | Weight | "
                     "Canonical Signal | Monitoring Status | Support | Resistance | Market Data Status | Notes |")
        lines.append("|--------|---------|-----|----------|---------------|--------------|----------------|--------------|-----------|-------|--------|"
                     "----------------|-------------------|---------|------------|--------------------|-------|")
        for r in rows:
            if r.get("pnl_validated", True):
                unreal_s = _feur(r.get("unrealized_pnl"))
                total_s = _feur(r.get("total_pnl"))
                try:
                    pct_s = f"{float(r.get('pnl_pct', 0) or 0):+.1f}%"
                except (TypeError, ValueError):
                    pct_s = "n/a"
            else:
                unreal_s = total_s = pct_s = "n/a"
            mon = (monitoring_by_display or {}).get(
                str(r.get("display_symbol") or r.get("ticker") or "").strip().upper(), "HOLD")
            lines.append(
                f"| {r.get('display_symbol') or r.get('ticker')} | {r.get('company')} | "
                f"{float(r.get('quantity', 0) or 0):.4f} | {_feur(r.get('avg_cost'))} | {_feur(r.get('current_price'))} | "
                f"{_feur(r.get('market_value'))} | {unreal_s} | {_feur(r.get('realized_pnl'))} | {total_s} | "
                f"{pct_s} | {float(r.get('weight', 0) or 0):.2f}% | {signal_emoji(r.get('signal', 'HOLD'))} | {mon} | "
                f"{_fnum(r.get('support'))} | {_fnum(r.get('resistance'))} | "
                f"{r.get('market_data_state', 'n/a')} | {r.get('notes') or '—'} |"
            )
    lines += ["", "## Data Coverage", ""]
    unmapped = sorted({str(r.get("display_symbol") or r.get("ticker"))
                       for r in (rows or [])
                       if str(r.get("market_data_state", "") or "").upper() == "UNRESOLVED"})
    lines.append("- Instruments without market-data mapping: " + (", ".join(unmapped) if unmapped else "none."))
    no_earn: list[str] = []
    for r in rows or []:
        disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
        val = (earnings_status or {}).get(disp)
        dated = (isinstance(val, (tuple, list)) and len(val) > 0
                 and isinstance(val[0], str) and len(val[0]) >= 10)
        if not dated:
            no_earn.append(disp)
    lines.append("- Earnings unavailable (actual holdings): " + (", ".join(sorted(set(no_earn))) if no_earn else "none."))
    lines.append("- Realized P&L: unavailable — transaction ledger not imported (shows n/a).")
    led = ledger_metadata or {}
    lines.append("- Cash-flow source: " + str(led.get("selected_source", (perf or {}).get(
        "net_deposits_source", "Unavailable — no verified cash-flow source")))
        + f" ({led.get('selected_status', (perf or {}).get('net_deposits_status', 'UNAVAILABLE'))}).")
    if led.get("selected_status") == "VERIFIED":
        lines.append(f"- Historical FX conversion: complete ({led.get('api_items', 0)} API items, "
                     f"{led.get('api_unresolved_fx', 0)} unresolved).")
    else:
        lines.append("- Historical FX conversion: manual EUR baseline — no per-item conversion required.")
    lines.append(f"- Reconciliation: positions {_feur(recon.get('positions_value'))} + cash "
                 f"{_feur(recon.get('reported_cash'))} = derived {_feur(recon.get('derived_total'))} vs broker "
                 f"{_feur(recon.get('total_equity'))} (delta {_feur(recon.get('cash_delta'))}, "
                 f"threshold {_feur(recon.get('threshold'))}) → **{status}**.")
    return "\n".join(lines).rstrip() + "\n"
