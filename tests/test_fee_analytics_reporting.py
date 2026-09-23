"""Tests for fee analytics and cost-aware reporting."""

import pytest
from decimal import Decimal
from pathlib import Path

from investment_engine.accounting import build_ledger
from investment_engine.accounting.reporting import (
    generate_historical_ledger_report,
    generate_fee_analytics_summary,
)
from investment_engine.accounting.performance import PerformanceResult
from investment_engine.accounting.models import LedgerSummary
from investment_engine.accounting.fees import FeeBreakdown
from investment_engine.accounting.performance import TradeStatistics


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "t212_transactions_sanitized.csv"


def test_generate_historical_ledger_report_basic():
    """Report generation should work for a basic ledger."""
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    
    report = generate_historical_ledger_report(ledger.performance, ledger.reconciliation)
    
    assert "## Historical Trading Ledger" in report
    assert "## Realized Performance Before Explicit Costs" in report
    assert "## Explicit Costs" in report
    assert "## Net Realized Performance After Known Costs" in report
    assert "## Dividends and Taxes" in report
    assert "## Data Completeness and Reconciliation" in report
    assert "## Important Limits" in report
    assert "PARTIAL DATA" in report  # Fixture has unknown events


def test_generate_historical_ledger_report_unavailable():
    """Report should handle UNAVAILABLE data quality."""
    from investment_engine.accounting.performance import PerformanceResult
    from investment_engine.accounting.models import LedgerSummary
    from investment_engine.accounting.fees import FeeBreakdown
    from investment_engine.accounting.performance import TradeStatistics
    
    perf = PerformanceResult(
        summary=LedgerSummary(accounting_status="UNAVAILABLE", total_events=0),
        fee_breakdown=FeeBreakdown(),
        trade_stats=TradeStatistics(),
        data_quality="UNAVAILABLE",
        completeness_warnings=["No CSV files imported"],
    )
    
    report = generate_historical_ledger_report(perf, None)
    
    assert "NO DATA" in report
    assert "Historical transaction ledger not imported" in report


def test_generate_historical_ledger_report_reconciled():
    """Report should show RECONCILED status when appropriate."""
    from investment_engine.accounting.performance import PerformanceResult
    from investment_engine.accounting.models import LedgerSummary
    from investment_engine.accounting.fees import FeeBreakdown
    from investment_engine.accounting.performance import TradeStatistics
    
    perf = PerformanceResult(
        summary=LedgerSummary(
            accounting_status="RECONCILED",
            total_events=10,
            gross_realized_pnl_eur=Decimal("1000"),
            net_realized_pnl_after_known_costs_eur=Decimal("900"),
            known_explicit_costs_eur=Decimal("100"),
        ),
        fee_breakdown=FeeBreakdown(),
        trade_stats=TradeStatistics(),
        data_quality="RECONCILED",
        completeness_warnings=[],
    )
    
    report = generate_historical_ledger_report(perf, None)
    
    assert "RECONCILED" in report


def test_generate_fee_analytics_summary():
    """Fee analytics summary should return structured data."""
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    
    summary = generate_fee_analytics_summary(ledger.performance)
    
    assert "gross_realized_pnl_eur" in summary
    assert "net_realized_pnl_after_known_costs_eur" in summary
    assert "known_explicit_costs_eur" in summary
    assert "cost_breakdown_eur" in summary
    assert "cost_as_pct_of_gross_realized" in summary
    assert "trade_statistics" in summary
    assert "dividends" in summary
    assert "cash_flows" in summary
    assert "data_quality" in summary
    assert "warnings" in summary
    
    # All monetary values should be strings
    for key in ["gross_realized_pnl_eur", "net_realized_pnl_after_known_costs_eur", "known_explicit_costs_eur"]:
        assert isinstance(summary[key], str)


def test_report_contains_correct_definitions():
    """Report should contain correct definitions and disclaimers."""
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    
    report = generate_historical_ledger_report(ledger.performance, ledger.reconciliation)
    
    # Check key disclaimers
    assert "NOT the account total return" in report
    assert "account equity − net deposits" in report
    assert "broker-authoritative" in report
    assert "No forward-looking fee schedules used" in report
    assert "FIFO accounting" in report


def test_report_format_eur():
    """EUR formatting should work correctly."""
    from investment_engine.accounting.reporting import format_eur, format_pct, format_decimal
    
    assert format_eur(Decimal("1234.56")) == "€1,234.56"
    assert format_eur(Decimal("-100.00")) == "€-100.00"
    assert format_eur(Decimal("0")) == "€0.00"
    assert format_pct(Decimal("12.34")) == "12.34%"
    assert format_decimal(Decimal("1.2345"), 2) == "1.23"
    assert format_decimal(Decimal("1.2345"), 4) == "1.2345"


def test_report_with_reconciliation():
    """Report should include reconciliation data when provided."""
    from investment_engine.accounting.reconciliation import ReconciliationResult
    from investment_engine.accounting.performance import PerformanceResult
    from investment_engine.accounting.models import LedgerSummary
    from investment_engine.accounting.fees import FeeBreakdown
    from investment_engine.accounting.performance import TradeStatistics
    
    perf = PerformanceResult(
        summary=LedgerSummary(
            accounting_status="RECONCILED",
            total_events=10,
            gross_realized_pnl_eur=Decimal("1000"),
            net_realized_pnl_after_known_costs_eur=Decimal("900"),
            known_explicit_costs_eur=Decimal("100"),
            net_deposits_eur=Decimal("5000"),
            open_lot_cost_basis_eur=Decimal("2000"),
        ),
        fee_breakdown=FeeBreakdown(),
        trade_stats=TradeStatistics(),
        data_quality="RECONCILED",
        completeness_warnings=[],
    )
    
    recon = ReconciliationResult(
        ledger_net_deposits_eur=Decimal("5000"),
        broker_net_deposits_eur=Decimal("5000"),
        deposit_difference_eur=Decimal("0"),
        deposit_reconciled=True,
        ledger_open_lots_cost_basis_eur=Decimal("2000"),
        broker_positions_value_eur=Decimal("2500"),
        position_difference_eur=Decimal("500"),
        ledger_net_realized_pnl_eur=Decimal("900"),
        broker_account_return_eur=Decimal("1200"),
        performance_difference_eur=Decimal("300"),
        status="RECONCILED",
        notes=["Deposits match", "Positions differ due to unrealized P/L"],
    )
    
    report = generate_historical_ledger_report(perf, recon)
    
    assert "RECONCILED" in report
    assert "Deposits match" in report
    assert "Positions differ due to unrealized P/L" in report
    assert "€5,000.00" in report  # net deposits
    assert "€5,000.00" in report  # broker net deposits


def test_report_cost_ratios():
    """Report should calculate cost ratios correctly."""
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    
    report = generate_historical_ledger_report(ledger.performance, ledger.reconciliation)
    
    # Check cost ratios appear
    assert "Costs as % of Gross Realized P/L" in report
    assert "Costs as % of Gross Turnover" in report
    assert "Average Explicit Cost per Completed Sale" in report
    assert "Cost Drag" in report