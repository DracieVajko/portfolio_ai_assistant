"""Integrated portfolio report generator with pie exposure and historical ledger sections.

Extends the existing regime report with portfolio exposure, actionability,
and historical transaction ledger sections. Advisory only - no live trading.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from investment_engine.portfolio.exposure import (
    CanonicalExposure,
    generate_exposure_view,
    generate_actionability_view,
)
from investment_engine.accounting.reporting import generate_historical_ledger_report
from investment_engine.accounting import PerformanceResult
from investment_engine.accounting.reconciliation import ReconciliationResult


def generate_portfolio_exposure_section(
    canonical_exposures: List[CanonicalExposure],
    total_equity_eur: float,
) -> str:
    """Generate Portfolio Exposure section for the report."""
    exposure_view = generate_exposure_view(canonical_exposures, total_equity_eur)
    
    lines = ["## 📊 Portfolio Exposure", ""]
    lines.append("*Aggregates pie-held and standalone positions for risk analysis. CSV zero placeholders ignored.*")
    lines.append("")
    
    if not exposure_view["exposures"]:
        lines.append("No positions to display.")
        return "\n".join(lines)
    
    # Summary stats
    total_value = sum(e["total_value_eur"] for e in exposure_view["exposures"])
    pie_value = sum(e["pie_value_eur"] for e in exposure_view["exposures"])
    standalone_value = sum(e["standalone_value_eur"] for e in exposure_view["exposures"])
    
    lines.append(f"**Total Portfolio Value:** €{total_value:,.2f}")
    lines.append(f"**Pie-Held Value:** €{pie_value:,.2f} ({pie_value/total_value*100:.1f}%)" if total_value > 0 else "")
    lines.append(f"**Standalone Value:** €{standalone_value:,.2f} ({standalone_value/total_value*100:.1f}%)" if total_value > 0 else "")
    lines.append("")
    
    # Exposure table
    lines.append("| Symbol | Total Qty | Total Value | Weight % | Pie Value | Standalone | Pies | Concentration | Reliability |")
    lines.append("|--------|-----------|-------------|----------|-----------|------------|------|---------------|-------------|")
    
    for exp in exposure_view["exposures"]:
        symbol = exp["symbol"]
        total_qty = exp["total_quantity"]
        total_val = exp["total_value_eur"]
        weight = exp["total_weight_pct"]
        pie_val = exp["pie_value_eur"]
        standalone_val = exp["standalone_value_eur"]
        pies = ", ".join(exp["pies"]) if exp["pies"] else "—"
        conc = "⚠️ WARNING" if exp["concentration_warning"] else "OK"
        reliability = exp.get("allocation_note", "")
        
        lines.append(
            f"| {symbol} | {total_qty:.4f} | €{total_val:,.2f} | {weight:.2f}% | "
            f"€{pie_val:,.2f} | €{standalone_val:,.2f} | {pies} | {conc} | {reliability} |"
        )
    
    lines.append("")
    lines.append("*Concentration WARNING = total exposure exceeds most restrictive pie cap.*")
    lines.append("*Reliability note only shown when pie breakdown is not exact (estimated/inferred).*")
    
    return "\n".join(lines)


def generate_actionability_section(
    canonical_exposures: List[CanonicalExposure],
) -> str:
    """Generate Actionability section for the report."""
    action_view = generate_actionability_view(canonical_exposures)
    
    lines = ["## 🎯 Actionability", ""]
    lines.append("*Advisory only. Pie-held actions are pie-level; standalone actions do not modify pie weights. No automatic orders.*")
    lines.append("")
    
    if not action_view["actions"]:
        lines.append("No actionable positions.")
        return "\n".join(lines)
    
    lines.append("| Symbol | Pie Action | Standalone Action | Scope | Add via Pie | Reduce via Pie | Buy Standalone | Sell Standalone | Estimated Alloc | Reason |")
    lines.append("|--------|------------|-------------------|-------|-------------|----------------|----------------|-----------------|-----------------|--------|")
    
    for action in action_view["actions"]:
        symbol = action["symbol"]
        pie_action = action["pie_action"]
        standalone_action = action["standalone_action"]
        scope = action["default_action_scope"]
        can_add = "✅" if action["can_add_via_pie"] else "❌"
        can_reduce = "✅" if action["can_reduce_via_pie"] else "❌"
        can_buy = "✅" if action["can_buy_standalone"] else "❌"
        can_sell = "✅" if action["can_sell_standalone"] else "❌"
        estimated = "⚠️ YES" if action["per_pie_breakdown_reliability"] == "estimated" else "No"
        reason = action.get("explanation", "")
        
        lines.append(
            f"| {symbol} | {pie_action} | {standalone_action} | {scope} | "
            f"{can_add} | {can_reduce} | {can_buy} | {can_sell} | {estimated} | {reason} |"
        )
    
    lines.append("")
    lines.append("**Constraints:**")
    lines.append("- Pie constituents cannot be rendered as plain SELL instructions")
    lines.append("- If per-pie split estimated (⚠️), pie actions restricted to REVIEW_PIE_ALLOCATION")
    lines.append("- Daily Div: pie-only, long-term; no standalone technical actions by default")
    lines.append("- Tech Pie: standalone adds require reconciliation PASS + validated signal + cap checks")
    lines.append("- Savings: never receives trading action suggestions")
    
    return "\n".join(lines)


def generate_historical_ledger_section(
    performance: Optional[PerformanceResult],
    reconciliation: Optional[ReconciliationResult] = None,
) -> str:
    """Generate Historical Trading Ledger section for the report."""
    lines = ["## 📜 Historical Trading Ledger", ""]
    
    if performance is None:
        lines.append("**Historical transaction ledger: not imported.**")
        lines.append("")
        lines.append("*To enable: export Trading 212 activity CSV and cash CSV, place in local ignored folder, run with --ledger flag.*")
        return "\n".join(lines)
    
    if performance.data_quality == "UNAVAILABLE":
        lines.append("**Historical transaction ledger: not imported.**")
        return "\n".join(lines)
    
    if performance.data_quality == "PARTIAL":
        lines.append("**Historical transaction ledger: partial; values below are not account total return.**")
        lines.append("")
    
    # Use the detailed reporting function
    detailed_report = generate_historical_ledger_report(performance, reconciliation)
    lines.append(detailed_report)
    
    return "\n".join(lines)


def generate_full_integrated_report(
    regime_result: Any,
    t212_data: Optional[Dict[str, Any]] = None,
    ai_recs: Optional[Dict[str, Any]] = None,
    canonical_exposures: Optional[List[CanonicalExposure]] = None,
    total_equity_eur: Optional[float] = None,
    historical_performance: Optional[PerformanceResult] = None,
    historical_reconciliation: Optional[ReconciliationResult] = None,
    earnings: Optional[Dict[str, Any]] = None,
    technicals: Optional[Dict[str, Any]] = None,
    names: Optional[Dict[str, str]] = None,
    known_clean: Optional[set] = None,
    canonical: Optional[Dict[str, Any]] = None,
    yahoo_map: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate complete integrated report with all sections.
    
    Section order:
    1. Regime Header & Summary (existing)
    2. Key Levels (existing)
    3. News (existing)
    4. Portfolio Exposure (NEW)
    5. Actionability (NEW)
    6. T212 Portfolio — unified holdings (exactly one holdings table)
    7. Historical Trading Ledger (NEW, optional)
    8. AI Recommendations (existing)
    9. Earnings Radar — month-grouped timeline (existing, refactored)
    10. Warnings (existing)
    """
    from investment_engine.reporting.regime_report import RegimeReportGenerator
    
    generator = RegimeReportGenerator()
    
    # Use existing regime report sections
    sections = [
        generator._header(regime_result),
        generator._regime_summary(regime_result),
        generator._key_levels(regime_result),
        generator._news_section(regime_result),
    ]
    
    # NEW: Portfolio Exposure
    if canonical_exposures and total_equity_eur is not None:
        sections.append(generate_portfolio_exposure_section(canonical_exposures, total_equity_eur))
    
    # NEW: Actionability
    if canonical_exposures:
        sections.append(generate_actionability_section(canonical_exposures))
    
    # Unified holdings: exactly one T212 Portfolio table (authoritative snapshot).
    if t212_data:
        sections.append(generator._t212_portfolio(t212_data, ai_recs, technicals, names, known_clean, canonical, None, yahoo_map))
    
    # NEW: Historical Trading Ledger (optional)
    sections.append(generate_historical_ledger_section(historical_performance, historical_reconciliation))
    
    # Existing: AI Recommendations
    if ai_recs:
        sections.append(generator._ai_recommendations(ai_recs, known_clean, canonical, names))
    
    # Earnings Radar (strict month-grouped timeline, inline tags only).
    sections.append(generator._earnings_calendar(regime_result, earnings, names))
    
    # Existing: Warnings
    warnings_section = generator._warnings(regime_result)
    if warnings_section:
        sections.append(warnings_section)
    
    return "\n\n".join(sections)