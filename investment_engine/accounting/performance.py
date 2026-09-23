"""Historical performance calculations from ledger data.

Calculates realized P&L, costs, dividends, and trade statistics from
matched trades and ledger events. All calculations in EUR using Decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from investment_engine.accounting.models import (
    LedgerEvent,
    MatchedTrade,
    LedgerSummary,
    EventType,
)
from investment_engine.accounting.fees import FeeBreakdown, aggregate_fees
from investment_engine.accounting.lot_matching import MatchingResult


def _convert_to_eur(amount: Decimal, currency: str, exchange_rate: Optional[Decimal],
                    default_fx_rates: Dict[str, Decimal]) -> Decimal:
    """Convert amount to EUR using event's exchange rate or default rates."""
    if currency == "EUR":
        return amount
    if exchange_rate is not None and exchange_rate != 0:
        return amount * exchange_rate
    rate = default_fx_rates.get(currency)
    if rate is not None:
        return amount * rate
    return amount  # No conversion available


@dataclass
class TradeStatistics:
    """Trade-level statistics for closed trades."""
    profitable_count: int = 0
    losing_count: int = 0
    breakeven_count: int = 0
    gross_profit_eur: Decimal = Decimal("0")
    gross_loss_eur: Decimal = Decimal("0")
    profit_factor: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    avg_winner_eur: Optional[Decimal] = None
    avg_loser_eur: Optional[Decimal] = None
    expectancy_eur: Optional[Decimal] = None  # after known costs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profitable_count": self.profitable_count,
            "losing_count": self.losing_count,
            "breakeven_count": self.breakeven_count,
            "gross_profit_eur": str(self.gross_profit_eur),
            "gross_loss_eur": str(self.gross_loss_eur),
            "profit_factor": str(self.profit_factor) if self.profit_factor is not None else None,
            "win_rate": str(self.win_rate) if self.win_rate is not None else None,
            "avg_winner_eur": str(self.avg_winner_eur) if self.avg_winner_eur is not None else None,
            "avg_loser_eur": str(self.avg_loser_eur) if self.avg_loser_eur is not None else None,
            "expectancy_eur": str(self.expectancy_eur) if self.expectancy_eur is not None else None,
        }


@dataclass
class PerformanceResult:
    """Complete historical performance result."""
    summary: LedgerSummary
    fee_breakdown: FeeBreakdown
    trade_stats: TradeStatistics
    gross_turnover_eur: Decimal = Decimal("0")
    cost_drag: Optional[Decimal] = None  # known_costs / max(|gross_realized|, epsilon)
    completeness_warnings: List[str] = field(default_factory=list)
    data_quality: str = "PARTIAL"  # RECONCILED | PARTIAL | UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "fee_breakdown": self.fee_breakdown.to_dict(),
            "trade_stats": self.trade_stats.to_dict(),
            "gross_turnover_eur": str(self.gross_turnover_eur),
            "cost_drag": str(self.cost_drag) if self.cost_drag is not None else None,
            "completeness_warnings": self.completeness_warnings,
            "data_quality": self.data_quality,
        }


def calculate_trade_statistics(matched_trades: List[MatchedTrade]) -> TradeStatistics:
    """Calculate trade-level statistics from matched trades."""
    stats = TradeStatistics()
    pnls = []
    
    for trade in matched_trades:
        pnl = trade.realized_pnl_eur
        pnls.append(pnl)
        if pnl > Decimal("0.01"):  # profitable (with small epsilon)
            stats.profitable_count += 1
            stats.gross_profit_eur += pnl
        elif pnl < Decimal("-0.01"):  # losing
            stats.losing_count += 1
            stats.gross_loss_eur += abs(pnl)
        else:
            stats.breakeven_count += 1
    
    total_closed = stats.profitable_count + stats.losing_count + stats.breakeven_count
    if total_closed > 0:
        stats.win_rate = Decimal(str(stats.profitable_count)) / Decimal(str(total_closed)) * Decimal("100")
    
    if stats.gross_loss_eur > 0:
        stats.profit_factor = stats.gross_profit_eur / stats.gross_loss_eur
    elif stats.gross_profit_eur > 0:
        # All trades profitable - profit factor is effectively infinite
        stats.profit_factor = Decimal("Infinity")
    
    if stats.profitable_count > 0:
        stats.avg_winner_eur = stats.gross_profit_eur / Decimal(str(stats.profitable_count))
    
    if stats.losing_count > 0:
        stats.avg_loser_eur = stats.gross_loss_eur / Decimal(str(stats.losing_count))
    
    # Expectancy after known costs (average P&L per trade including fees)
    if total_closed > 0:
        total_pnl = sum(t.realized_pnl_eur for t in matched_trades)
        stats.expectancy_eur = total_pnl / Decimal(str(total_closed))
    
    return stats


def calculate_performance(
    events: List[LedgerEvent],
    matching_result: MatchingResult,
    manual_adjustments: Optional[Any] = None,  # ManualAdjustments type
    default_fx_rates: Optional[Dict[str, Decimal]] = None,
) -> PerformanceResult:
    """Calculate complete historical performance from ledger events.
    
    This is the main entry point for historical performance reporting.
    """
    # Convert amounts to EUR if not already done
    if default_fx_rates:
        default_fx_rates_str = {str(k): v for k, v in default_fx_rates.items()}
        for event in events:
            if event.gross_amount_eur is None and event.gross_amount != 0:
                event.gross_amount_eur = _convert_to_eur(event.gross_amount, event.currency, event.exchange_rate, default_fx_rates_str)
            if event.fee_amount_eur is None and event.fee_amount != 0:
                event.fee_amount_eur = _convert_to_eur(event.fee_amount, event.currency, event.exchange_rate, default_fx_rates_str)
            if event.tax_amount_eur is None and event.tax_amount != 0:
                event.tax_amount_eur = _convert_to_eur(event.tax_amount, event.currency, event.exchange_rate, default_fx_rates_str)
            if event.fx_fee_amount_eur is None and event.fx_fee_amount != 0:
                event.fx_fee_amount_eur = _convert_to_eur(event.fx_fee_amount, event.currency, event.exchange_rate, default_fx_rates_str)
            if event.net_amount_eur is None and event.net_amount != 0:
                event.net_amount_eur = _convert_to_eur(event.net_amount, event.currency, event.exchange_rate, default_fx_rates_str)
    
    warnings: List[str] = []
    
    # Aggregate fees from fee events
    fee_breakdown = aggregate_fees(events)
    
    # Add fees from matched trades (already in EUR)
    for trade in matching_result.matched_trades:
        if trade.fees_eur > 0:
            fee_breakdown.other_eur += trade.fees_eur
    
    # Calculate trade statistics
    trade_stats = calculate_trade_statistics(matching_result.matched_trades)
    
    # Gross realized P&L (before explicit costs)
    gross_realized_pnl_eur = sum(t.realized_pnl_eur for t in matching_result.matched_trades)
    
    # Known explicit costs
    known_explicit_costs_eur = fee_breakdown.total()
    
    # Net realized P&L after known costs
    net_realized_pnl_after_known_costs_eur = gross_realized_pnl_eur - known_explicit_costs_eur
    
    # Dividends
    dividends_gross_eur = Decimal("0")
    dividend_withholding_tax_eur = Decimal("0")
    for event in events:
        if event.event_type == EventType.DIVIDEND:
            amt = event.gross_amount_eur if event.gross_amount_eur is not None else event.gross_amount
            dividends_gross_eur += amt
        elif event.event_type == EventType.DIVIDEND_WITHHOLDING_TAX:
            amt = event.tax_amount_eur if event.tax_amount_eur is not None else event.tax_amount
            dividend_withholding_tax_eur += amt
    
    dividends_net_eur = dividends_gross_eur - dividend_withholding_tax_eur
    
    # Cash flows
    deposits_eur = Decimal("0")
    withdrawals_eur = Decimal("0")
    cash_interest_eur = Decimal("0")
    for event in events:
        if event.event_type == EventType.CASH_DEPOSIT:
            amt = event.net_amount_eur if event.net_amount_eur is not None else event.net_amount
            deposits_eur += amt
        elif event.event_type == EventType.CASH_WITHDRAWAL:
            amt = event.net_amount_eur if event.net_amount_eur is not None else event.net_amount
            withdrawals_eur += abs(amt)
        elif event.event_type == EventType.CASH_INTEREST:
            amt = event.net_amount_eur if event.net_amount_eur is not None else event.net_amount
            cash_interest_eur += amt
    
    net_deposits_eur = deposits_eur - withdrawals_eur
    
    # Open lot cost basis
    open_lot_cost_basis_eur = Decimal("0")
    for lots in matching_result.open_lots.values():
        for lot in lots:
            if lot.cost_basis_eur_per_share is not None:
                open_lot_cost_basis_eur += lot.cost_basis_eur_per_share * lot.remaining_quantity
    
    # Gross turnover (sum of absolute trade values)
    gross_turnover_eur = Decimal("0")
    for trade in matching_result.matched_trades:
        if trade.sell_price_eur is not None:
            gross_turnover_eur += trade.sell_price_eur * trade.quantity_sold
        else:
            gross_turnover_eur += trade.proceeds_eur
    
    # Cost drag
    cost_drag = None
    epsilon = Decimal("0.01")
    divisor = max(abs(gross_realized_pnl_eur), epsilon)
    if divisor > 0:
        cost_drag = known_explicit_costs_eur / divisor
    
    # Date range
    trade_events = [e for e in events if e.is_trade]
    first_event_date = min((e.timestamp for e in trade_events), default=None)
    last_event_date = max((e.timestamp for e in trade_events), default=None)
    
    # Count unknown events
    unknown_count = sum(1 for e in events if e.event_type == EventType.UNKNOWN)
    
    # Determine accounting status
    data_quality = "RECONCILED"
    if matching_result.short_sells:
        data_quality = "PARTIAL"
        warnings.append(f"Short sells detected: {len(matching_result.short_sells)} sell(s) exceeded available lots")
    if matching_result.unmatched_sells:
        data_quality = "PARTIAL"
        warnings.append(f"Unmatched sells: {len(matching_result.unmatched_sells)} sell(s) with no buy lots")
    if unknown_count > 0:
        data_quality = "PARTIAL"
        warnings.append(f"Unknown event types: {unknown_count} event(s) could not be classified")
    if not events:
        data_quality = "UNAVAILABLE"
        warnings.append("No events imported")
    
    # Apply manual adjustments if provided
    if manual_adjustments:
        if manual_adjustments.starting_cash_eur is not None:
            net_deposits_eur += manual_adjustments.starting_cash_eur
            warnings.append(f"Applied manual starting cash adjustment: {manual_adjustments.starting_cash_eur} EUR")
        for adj in manual_adjustments.historical_fee_adjustments:
            # Would apply fee adjustments – implementation depends on structure
            pass
        for dep in manual_adjustments.external_deposits:
            # Would add external deposits
            pass
        for wd in manual_adjustments.external_withdrawals:
            # Would add external withdrawals
            pass
    
    summary = LedgerSummary(
        gross_realized_pnl_eur=gross_realized_pnl_eur,
        realized_trading_fees_eur=fee_breakdown.trading_commission_eur,
        fx_conversion_fees_eur=fee_breakdown.fx_conversion_eur,
        stamp_duty_eur=fee_breakdown.stamp_duty_eur,
        financial_transaction_tax_eur=fee_breakdown.financial_transaction_tax_eur,
        commissions_eur=fee_breakdown.trading_commission_eur,
        other_known_fees_eur=fee_breakdown.other_eur,
        known_explicit_costs_eur=known_explicit_costs_eur,
        net_realized_pnl_after_known_costs_eur=net_realized_pnl_after_known_costs_eur,
        dividends_gross_eur=dividends_gross_eur,
        dividend_withholding_tax_eur=dividend_withholding_tax_eur,
        dividends_net_eur=dividends_net_eur,
        cash_interest_eur=cash_interest_eur,
        deposits_eur=deposits_eur,
        withdrawals_eur=withdrawals_eur,
        net_deposits_eur=net_deposits_eur,
        open_lot_cost_basis_eur=open_lot_cost_basis_eur,
        unmatched_sell_count=len(matching_result.unmatched_sells),
        unknown_event_count=unknown_count,
        accounting_status=data_quality,
        first_event_date=first_event_date,
        last_event_date=last_event_date,
        total_events=len(events),
    )
    
    return PerformanceResult(
        summary=summary,
        fee_breakdown=fee_breakdown,
        trade_stats=trade_stats,
        gross_turnover_eur=gross_turnover_eur,
        cost_drag=cost_drag,
        completeness_warnings=warnings,
        data_quality=data_quality,
    )