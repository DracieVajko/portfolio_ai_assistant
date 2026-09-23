"""Reporting module for historical transaction ledger.

Generates human-readable Markdown reports from ledger performance data.
Clearly separates historical ledger from current portfolio account return.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from investment_engine.accounting.models import LedgerSummary
from investment_engine.accounting.performance import PerformanceResult
from investment_engine.accounting.reconciliation import ReconciliationResult


def format_eur(amount: Decimal) -> str:
    """Format EUR amount with 2 decimal places and € sign."""
    return f"€{amount:,.2f}"


def format_pct(value: Decimal) -> str:
    """Format percentage with 2 decimal places."""
    return f"{value:.2f}%"


def format_decimal(value: Decimal, decimals: int = 2) -> str:
    """Format decimal with specified decimal places."""
    return f"{value:.{decimals}f}"


def generate_historical_ledger_report(
    performance: PerformanceResult,
    reconciliation: Optional[ReconciliationResult] = None,
    include_definitions: bool = True,
) -> str:
    """Generate complete historical ledger report in Markdown.
    
    Sections:
    1. Historical Trading Ledger
    2. Realized Performance Before Explicit Costs
    3. Explicit Costs
    4. Net Realized Performance After Known Costs
    5. Dividends and Taxes
    6. Data Completeness and Reconciliation
    7. Important Limits
    """
    lines: List[str] = []
    summary = performance.summary
    fee_breakdown = performance.fee_breakdown
    trade_stats = performance.trade_stats
    
    # Header with data quality warning
    if performance.data_quality == "PARTIAL":
        lines.append("⚠️ **PARTIAL DATA** — Historical transaction ledger is incomplete; values below are not account total return.")
        lines.append("")
    elif performance.data_quality == "UNAVAILABLE":
        lines.append("ℹ️ **NO DATA** — Historical transaction ledger not imported.")
        lines.append("")
        return "\n".join(lines)
    else:
        lines.append("✅ **RECONCILED** — Historical transaction ledger fully reconciled.")
        lines.append("")
    
    lines.append("## Historical Trading Ledger")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accounting Status | {summary.accounting_status} |")
    lines.append(f"| Period | {summary.first_event_date.date() if summary.first_event_date else 'N/A'} to {summary.last_event_date.date() if summary.last_event_date else 'N/A'} |")
    lines.append(f"| Total Events | {summary.total_events} |")
    lines.append(f"| Unknown/Unclassified Events | {summary.unknown_event_count} |")
    lines.append(f"| Unmatched Sells | {summary.unmatched_sell_count} |")
    lines.append("")
    
    # Section 2: Realized Performance Before Explicit Costs
    lines.append("## Realized Performance Before Explicit Costs")
    lines.append("")
    if include_definitions:
        lines.append("*Gross realized P/L = Sum of (sell proceeds - cost basis) for all closed trades, before fees and taxes.*")
        lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Gross Realized P/L | {format_eur(summary.gross_realized_pnl_eur)} |")
    lines.append(f"| Profitable Closed Trades | {trade_stats.profitable_count} |")
    lines.append(f"| Losing Closed Trades | {trade_stats.losing_count} |")
    lines.append(f"| Break-even Trades | {trade_stats.breakeven_count} |")
    lines.append(f"| Win Rate | {format_pct(trade_stats.win_rate) if trade_stats.win_rate else 'N/A'} |")
    lines.append(f"| Profit Factor | {format_decimal(trade_stats.profit_factor) if trade_stats.profit_factor and trade_stats.profit_factor != Decimal('Infinity') else '∞ (all profitable)'} |")
    lines.append(f"| Gross Profit | {format_eur(trade_stats.gross_profit_eur)} |")
    lines.append(f"| Gross Loss | {format_eur(trade_stats.gross_loss_eur)} |")
    lines.append(f"| Average Winner | {format_eur(trade_stats.avg_winner_eur) if trade_stats.avg_winner_eur else 'N/A'} |")
    lines.append(f"| Average Loser | {format_eur(trade_stats.avg_loser_eur) if trade_stats.avg_loser_eur else 'N/A'} |")
    lines.append(f"| Expectancy (after known costs) | {format_eur(trade_stats.expectancy_eur) if trade_stats.expectancy_eur else 'N/A'} |")
    lines.append("")
    
    # Section 3: Explicit Costs
    lines.append("## Explicit Costs")
    lines.append("")
    if include_definitions:
        lines.append("*All costs below are known explicit costs from the imported transaction ledger. They do not include implicit costs (spread, slippage) or unavailable historical fees.*")
        lines.append("")
    lines.append("| Cost Category | Amount (EUR) |")
    lines.append("|---------------|--------------|")
    lines.append(f"| Trading Commissions | {format_eur(fee_breakdown.trading_commission_eur)} |")
    lines.append(f"| FX Conversion Fees | {format_eur(fee_breakdown.fx_conversion_eur)} |")
    lines.append(f"| Stamp Duty (UK) | {format_eur(fee_breakdown.stamp_duty_eur)} |")
    lines.append(f"| Financial Transaction Tax (FR/IT/etc.) | {format_eur(fee_breakdown.financial_transaction_tax_eur)} |")
    lines.append(f"| Dividend Withholding Tax | {format_eur(fee_breakdown.dividend_withholding_tax_eur)} |")
    lines.append(f"| Other Known Fees | {format_eur(fee_breakdown.other_eur)} |")
    lines.append(f"| **Total Known Explicit Costs** | **{format_eur(fee_breakdown.total())}** |")
    lines.append("")
    
    # Cost ratios
    if summary.gross_realized_pnl_eur != 0:
        cost_pct_of_gross = fee_breakdown.total() / abs(summary.gross_realized_pnl_eur) * Decimal("100")
        lines.append(f"Costs as % of Gross Realized P/L: **{format_pct(cost_pct_of_gross)}**")
    if performance.gross_turnover_eur != 0:
        cost_pct_of_turnover = fee_breakdown.total() / performance.gross_turnover_eur * Decimal("100")
        lines.append(f"Costs as % of Gross Turnover: **{format_pct(cost_pct_of_turnover)}**")
    if trade_stats.profitable_count + trade_stats.losing_count > 0:
        avg_cost_per_sale = fee_breakdown.total() / Decimal(str(trade_stats.profitable_count + trade_stats.losing_count))
        lines.append(f"Average Explicit Cost per Completed Sale: **{format_eur(avg_cost_per_sale)}**")
    if performance.cost_drag:
        lines.append(f"Cost Drag (known_costs / max(|gross|, ε)): **{format_decimal(performance.cost_drag, 4)}**")
    lines.append("")
    
    # Section 4: Net Realized Performance After Known Costs
    lines.append("## Net Realized Performance After Known Costs")
    lines.append("")
    if include_definitions:
        lines.append("*Net realized P/L after known explicit costs = Gross realized P/L − Total known explicit costs.*")
        lines.append("*This is NOT the account total return. Account total return = broker equity − net deposits (or broker-provided account return).*")
        lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Gross Realized P/L | {format_eur(summary.gross_realized_pnl_eur)} |")
    lines.append(f"| Total Known Explicit Costs | {format_eur(fee_breakdown.total())} |")
    lines.append(f"| **Net Realized P/L After Known Costs** | **{format_eur(summary.net_realized_pnl_after_known_costs_eur)}** |")
    lines.append("")
    
    # Section 5: Dividends and Taxes
    lines.append("## Dividends and Taxes")
    lines.append("")
    if include_definitions:
        lines.append("*Dividends gross = cash dividends received before withholding tax.*")
        lines.append("*Dividends net = gross dividends − withholding tax.*")
        lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Dividends Gross | {format_eur(summary.dividends_gross_eur)} |")
    lines.append(f"| Dividend Withholding Tax | {format_eur(summary.dividend_withholding_tax_eur)} |")
    lines.append(f"| **Dividends Net** | **{format_eur(summary.dividends_net_eur)}** |")
    lines.append(f"| Cash Interest | {format_eur(summary.cash_interest_eur)} |")
    lines.append("")
    
    # Section 6: Data Completeness and Reconciliation
    lines.append("## Data Completeness and Reconciliation")
    lines.append("")
    
    if performance.completeness_warnings:
        lines.append("**Warnings:**")
        for w in performance.completeness_warnings:
            lines.append(f"- {w}")
        lines.append("")
    
    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    lines.append(f"| Ledger Net Deposits | {format_eur(summary.net_deposits_eur)} |")
    lines.append(f"| Ledger Open Lots Cost Basis | {format_eur(summary.open_lot_cost_basis_eur)} |")
    
    if reconciliation:
        if reconciliation.broker_net_deposits_eur is not None:
            diff = reconciliation.deposit_difference_eur
            status = "✅ RECONCILED" if reconciliation.deposit_reconciled else "❌ MISMATCH"
            lines.append(f"| Broker Net Deposits | {format_eur(reconciliation.broker_net_deposits_eur)} |")
            lines.append(f"| Deposit Difference | {format_eur(diff) if diff else 'N/A'} |")
            lines.append(f"| Deposit Reconciliation | {status} |")
        else:
            lines.append("| Broker Net Deposits | Not provided |")
        
        if reconciliation.broker_positions_value_eur is not None:
            lines.append(f"| Broker Positions Value | {format_eur(reconciliation.broker_positions_value_eur)} |")
            if reconciliation.position_difference_eur is not None:
                lines.append(f"| Position Difference (vs cost basis) | {format_eur(reconciliation.position_difference_eur)} |")
        else:
            lines.append("| Broker Positions Value | Not provided |")
        
        if reconciliation.broker_account_return_eur is not None:
            lines.append(f"| Broker Account Return | {format_eur(reconciliation.broker_account_return_eur)} |")
            if reconciliation.performance_difference_eur is not None:
                lines.append(f"| Performance Difference (vs net realized) | {format_eur(reconciliation.performance_difference_eur)} |")
        else:
            lines.append("| Broker Account Return | Not provided |")
        
        lines.append(f"| **Overall Reconciliation Status** | **{reconciliation.status}** |")
        if reconciliation.notes:
            lines.append("")
            lines.append("**Notes:**")
            for note in reconciliation.notes:
                lines.append(f"- {note}")
    else:
        lines.append("| Broker Data | Not provided for reconciliation |")
    
    lines.append("")
    
    # Section 7: Important Limits
    lines.append("## Important Limits")
    lines.append("")
    lines.append("1. **This is not account total return.** Net realized P/L after known costs excludes:")
    lines.append("   - Unrealized P/L on open positions")
    lines.append("   - Implicit trading costs (spread, slippage, market impact)")
    lines.append("   - Fees not captured in the transaction export (e.g., custody fees, inactivity fees)")
    lines.append("   - Tax implications beyond explicit withholding tax")
    lines.append("2. **Account total return** remains broker-authoritative: `account equity − net deposits` (or broker-provided return).")
    lines.append("3. **Data quality:** " + (
        "PARTIAL — some events unknown/unmatched; figures are lower bounds."
        if performance.data_quality == "PARTIAL" else
        "RECONCILED — all events classified and matched."
    ))
    lines.append("4. **No forward-looking fee schedules used.** Historical fees from actual CSV only.")
    lines.append("5. **FIFO accounting** used for cost basis. Other methods (LIFO, average) would yield different results.")
    
    return "\n".join(lines)


def generate_fee_analytics_summary(performance: PerformanceResult) -> Dict[str, Any]:
    """Generate structured fee analytics for programmatic use."""
    summary = performance.summary
    fee_breakdown = performance.fee_breakdown
    trade_stats = performance.trade_stats
    
    return {
        "gross_realized_pnl_eur": str(summary.gross_realized_pnl_eur),
        "net_realized_pnl_after_known_costs_eur": str(summary.net_realized_pnl_after_known_costs_eur),
        "known_explicit_costs_eur": str(fee_breakdown.total()),
        "cost_breakdown_eur": fee_breakdown.to_dict(),
        "cost_as_pct_of_gross_realized": str(
            fee_breakdown.total() / abs(summary.gross_realized_pnl_eur) * Decimal("100")
        ) if summary.gross_realized_pnl_eur != 0 else None,
        "cost_as_pct_of_turnover": str(
            fee_breakdown.total() / performance.gross_turnover_eur * Decimal("100")
        ) if performance.gross_turnover_eur != 0 else None,
        "avg_explicit_cost_per_sale_eur": str(
            fee_breakdown.total() / Decimal(str(trade_stats.profitable_count + trade_stats.losing_count))
        ) if (trade_stats.profitable_count + trade_stats.losing_count) > 0 else None,
        "cost_drag": str(performance.cost_drag) if performance.cost_drag else None,
        "trade_statistics": trade_stats.to_dict(),
        "dividends": {
            "gross_eur": str(summary.dividends_gross_eur),
            "withholding_tax_eur": str(summary.dividend_withholding_tax_eur),
            "net_eur": str(summary.dividends_net_eur),
        },
        "cash_flows": {
            "deposits_eur": str(summary.deposits_eur),
            "withdrawals_eur": str(summary.withdrawals_eur),
            "net_deposits_eur": str(summary.net_deposits_eur),
            "interest_eur": str(summary.cash_interest_eur),
        },
        "open_positions": {
            "cost_basis_eur": str(summary.open_lot_cost_basis_eur),
        },
        "data_quality": performance.data_quality,
        "warnings": performance.completeness_warnings,
    }