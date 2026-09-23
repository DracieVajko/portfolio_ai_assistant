"""Ledger orchestration – builds complete historical ledger from imported events.

This module ties together import, lot matching, fee categorization,
performance calculation, and reconciliation into a single workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from investment_engine.accounting.models import (
    LedgerEvent,
    LedgerSummary,
    ManualAdjustments,
    EventType,
)
from investment_engine.accounting.importers.t212_csv import (
    import_t212_activity_csv,
    import_t212_cash_csv,
    detect_csv_type,
)
from investment_engine.accounting.lot_matching import (
    match_sell_fifo,
    MatchingResult,
)
from investment_engine.accounting.performance import (
    calculate_performance,
    PerformanceResult,
)
from investment_engine.accounting.reconciliation import (
    reconcile_ledger,
    ReconciliationResult,
)


@dataclass
class Ledger:
    """Complete historical ledger with all derived calculations."""
    events: List[LedgerEvent]
    matching_result: MatchingResult
    performance: PerformanceResult
    reconciliation: Optional[ReconciliationResult] = None
    manual_adjustments: Optional[ManualAdjustments] = None
    
    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        return {
            "events": [e.to_dict(include_raw=include_raw) for e in self.events],
            "matching": self.matching_result.to_dict(),
            "performance": self.performance.to_dict(),
            "reconciliation": self.reconciliation.to_dict() if self.reconciliation else None,
            "manual_adjustments": self.manual_adjustments.to_dict() if self.manual_adjustments else None,
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report for user-facing output."""
        return {
            "historical_ledger_summary": self.performance.summary.to_dict(),
            "fee_breakdown": self.performance.fee_breakdown.to_dict(),
            "trade_statistics": self.performance.trade_stats.to_dict(),
            "gross_turnover_eur": str(self.performance.gross_turnover_eur),
            "cost_drag": str(self.performance.cost_drag) if self.performance.cost_drag else None,
            "data_quality": self.performance.data_quality,
            "completeness_warnings": self.performance.completeness_warnings,
            "reconciliation": self.reconciliation.to_dict() if self.reconciliation else None,
        }


def build_ledger(
    activity_csv_path: Optional[str | Path] = None,
    cash_csv_path: Optional[str | Path] = None,
    manual_adjustments: Optional[ManualAdjustments] = None,
    default_fx_rates: Optional[Dict[str, Decimal]] = None,
    broker_net_deposits_eur: Optional[Decimal] = None,
    broker_positions_value_eur: Optional[Decimal] = None,
    broker_account_return_eur: Optional[Decimal] = None,
) -> Ledger:
    """Build complete historical ledger from CSV files.
    
    This is the main entry point for historical accounting.
    
    Args:
        activity_csv_path: Path to T212 transaction/activity export CSV
        cash_csv_path: Path to T212 cash transactions export CSV
        manual_adjustments: Optional manual adjustments for gaps
        default_fx_rates: Fallback FX rates for conversion to EUR
        broker_net_deposits_eur: Optional broker net deposits for reconciliation
        broker_positions_value_eur: Optional broker positions value for reconciliation
        broker_account_return_eur: Optional broker account return for reconciliation
    
    Returns:
        Ledger with all events, matching, performance, and reconciliation.
    """
    all_events: List[LedgerEvent] = []
    
    # Import activity CSV
    if activity_csv_path:
        activity_events = import_t212_activity_csv(activity_csv_path)
        all_events.extend(activity_events)
    
    # Import cash CSV
    if cash_csv_path:
        cash_events = import_t212_cash_csv(cash_csv_path)
        all_events.extend(cash_events)
    
    # If neither provided, return empty ledger
    if not all_events:
        from investment_engine.accounting.performance import PerformanceResult, TradeStatistics
        from investment_engine.accounting.fees import FeeBreakdown
        from investment_engine.accounting.lot_matching import MatchingResult
        
        empty_performance = PerformanceResult(
            summary=LedgerSummary(accounting_status="UNAVAILABLE", total_events=0),
            fee_breakdown=FeeBreakdown(),
            trade_stats=TradeStatistics(),
            data_quality="UNAVAILABLE",
            completeness_warnings=["No CSV files imported"],
        )
        return Ledger(
            events=[],
            matching_result=MatchingResult(),
            performance=empty_performance,
        )
    
    # Convert amounts to EUR where exchange rates available
    if default_fx_rates:
        default_fx_rates_str = {str(k): v for k, v in default_fx_rates.items()}
        for event in all_events:
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
    
    # Match sells to buys (FIFO)
    matching_result = match_sell_fifo(all_events, default_fx_rates)
    
    # Calculate performance
    performance = calculate_performance(
        all_events,
        matching_result,
        manual_adjustments,
        default_fx_rates,
    )
    
    # Reconcile if broker data provided
    reconciliation = None
    if any(v is not None for v in [broker_net_deposits_eur, broker_positions_value_eur, broker_account_return_eur]):
        reconciliation = reconcile_ledger(
            performance,
            broker_net_deposits_eur,
            broker_positions_value_eur,
            broker_account_return_eur,
        )
    
    return Ledger(
        events=all_events,
        matching_result=matching_result,
        performance=performance,
        reconciliation=reconciliation,
        manual_adjustments=manual_adjustments,
    )


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


def preview_csv(file_path: str | Path, max_rows: int = 5) -> Dict[str, Any]:
    """Preview a CSV file to verify column mapping before import."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        import csv
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    
    # Detect column mapping
    from investment_engine.accounting.importers.t212_csv import _detect_columns
    col_map = _detect_columns(headers)
    
    return {
        "file": str(path),
        "headers": headers,
        "column_mapping": col_map,
        "detected_type": detect_csv_type(path),
        "sample_rows": rows,
        "row_count_estimate": sum(1 for _ in Path(file_path).open("r", encoding="utf-8-sig")) - 1,
    }