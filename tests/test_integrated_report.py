"""Tests for integrated report generation with pie exposure and historical ledger."""

import pytest
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

from investment_engine.reporting.integrated_report import (
    generate_portfolio_exposure_section,
    generate_actionability_section,
    generate_historical_ledger_section,
    generate_full_integrated_report,
)
from investment_engine.accounting import build_ledger, PerformanceResult
from investment_engine.accounting.models import LedgerSummary
from investment_engine.accounting.fees import FeeBreakdown
from investment_engine.accounting.performance import TradeStatistics
from investment_engine.accounting.reconciliation import ReconciliationResult
from investment_engine.portfolio.exposure import CanonicalExposure, PieExposure, StandaloneExposure, ExecutionCapabilities


def _make_mock_exposure(symbol: str, total_val: float, pie_val: float, standalone_val: float, 
                        pies: list, conc_status: str = "OK", reliability: str = "exact") -> CanonicalExposure:
    """Create a mock CanonicalExposure for testing."""
    pie_exposures = []
    for i, pie_id in enumerate(pies):
        pie_exposures.append(PieExposure(
            pie_id=pie_id,
            quantity=1.0,
            value_eur=pie_val / len(pies) if pies else 0,
            weight_in_pie=10.0,
            source="api_exact",
            allocation_source="api_exact",
            allocation_confidence="high",
            allocation_is_estimated=False,
            allocation_note="Per-pie allocation from broker",
            per_pie_breakdown_reliability="exact",
            match_method="exact_normalized_symbol",
            match_confidence="high",
        ))
    
    standalone = StandaloneExposure(
        quantity=0.5 if standalone_val > 0 else 0,
        value_eur=standalone_val,
        source="api_standalone" if standalone_val > 0 else "api_none",
    )
    
    return CanonicalExposure(
        symbol=symbol,
        pie_exposures=pie_exposures,
        standalone_exposure=standalone,
        total_quantity=1.5,
        total_value_eur=total_val,
        total_weight_of_account_pct=total_val / 10000 * 100,
        pie_count=len(pies),
        execution_capabilities=ExecutionCapabilities(
            can_add_via_pie=len(pies) > 0,
            can_reduce_via_pie=len(pies) > 0 and pie_val > 0,
            can_buy_standalone=standalone_val == 0,
            can_sell_standalone=standalone_val > 0,
        ),
        default_action_scope="mixed" if pies and standalone_val > 0 else ("pie" if pies else "standalone"),
        concentration_status=conc_status,
        concentration_cap_pct=7.0 if "Tech" in str(pies) else (6.0 if pies else 10.0),
        data_validation_status="PASS",
        data_validation_reason="",
        pies_involved=pies,
        horizon="short_term" if "Tech" in str(pies) else "long_term",
        notes=[],
        per_pie_breakdown_reliability=reliability,
    )


def test_generate_portfolio_exposure_section():
    """Portfolio exposure section should render correctly."""
    exposures = [
        _make_mock_exposure("AAPL_US_EQ", 5000, 4000, 1000, ["TechPieShort", "DailyDivPieLongShort"], "WARNING", "estimated"),
        _make_mock_exposure("NVDA_US_EQ", 3000, 3000, 0, ["TechPieShort"], "OK", "inferred"),
        _make_mock_exposure("MSFT_US_EQ", 2000, 0, 2000, [], "OK", "unavailable"),
    ]
    
    section = generate_portfolio_exposure_section(exposures, 10000)
    
    assert "## 📊 Portfolio Exposure" in section
    assert "AAPL_US_EQ" in section
    assert "NVDA_US_EQ" in section
    assert "MSFT_US_EQ" in section
    assert "€5,000.00" in section  # AAPL total
    assert "€3,000.00" in section  # NVDA total
    assert "€2,000.00" in section  # MSFT total
    assert "⚠️ WARNING" in section  # AAPL concentration
    assert "TechPieShort, DailyDivPieLongShort" in section
    assert "estimated" in section.lower() or "inferred" in section.lower()


def test_generate_actionability_section():
    """Actionability section should render correctly."""
    exposures = [
        _make_mock_exposure("AAPL_US_EQ", 5000, 4000, 1000, ["TechPieShort", "DailyDivPieLongShort"], "WARNING", "estimated"),
        _make_mock_exposure("NVDA_US_EQ", 3000, 3000, 0, ["TechPieShort"], "OK", "inferred"),
        _make_mock_exposure("MSFT_US_EQ", 2000, 0, 2000, [], "OK", "unavailable"),
    ]
    
    section = generate_actionability_section(exposures)
    
    assert "## 🎯 Actionability" in section
    assert "AAPL_US_EQ" in section
    assert "NVDA_US_EQ" in section
    assert "MSFT_US_EQ" in section
    assert "REVIEW_PIE_ALLOCATION" in section  # AAPL has estimated allocation
    assert "HOLD_PIE" in section  # NVDA single pie
    assert "WATCH" in section or "BUY_STANDALONE" in section  # MSFT standalone
    assert "Pie constituents cannot be rendered as plain SELL" in section
    assert "Daily Div: pie-only" in section
    assert "Tech Pie: standalone adds require" in section
    assert "Savings: never receives" in section


def test_generate_historical_ledger_section_none():
    """Historical ledger section should handle None performance."""
    section = generate_historical_ledger_section(None, None)
    
    assert "## 📜 Historical Trading Ledger" in section
    assert "not imported" in section
    assert "To enable: export Trading 212 activity CSV" in section


def test_generate_historical_ledger_section_unavailable():
    """Historical ledger section should handle UNAVAILABLE performance."""
    perf = PerformanceResult(
        summary=LedgerSummary(accounting_status="UNAVAILABLE", total_events=0),
        fee_breakdown=FeeBreakdown(),
        trade_stats=TradeStatistics(),
        data_quality="UNAVAILABLE",
        completeness_warnings=[],
    )
    
    section = generate_historical_ledger_section(perf, None)
    
    assert "not imported" in section


def test_generate_historical_ledger_section_partial():
    """Historical ledger section should show PARTIAL warning."""
    perf = PerformanceResult(
        summary=LedgerSummary(
            accounting_status="PARTIAL",
            total_events=10,
            gross_realized_pnl_eur=Decimal("1000"),
            net_realized_pnl_after_known_costs_eur=Decimal("900"),
            known_explicit_costs_eur=Decimal("100"),
        ),
        fee_breakdown=FeeBreakdown(),
        trade_stats=TradeStatistics(),
        data_quality="PARTIAL",
        completeness_warnings=["Unknown event types: 2"],
    )
    
    section = generate_historical_ledger_section(perf, None)
    
    assert "partial" in section.lower()
    assert "not account total return" in section.lower()
    assert "Unknown event types: 2" in section


def test_generate_historical_ledger_section_reconciled():
    """Historical ledger section should show RECONCILED status."""
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
    
    section = generate_historical_ledger_section(perf, None)
    
    assert "RECONCILED" in section


def test_generate_historical_ledger_section_with_reconciliation():
    """Historical ledger section should include reconciliation data."""
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
    
    section = generate_historical_ledger_section(perf, recon)
    
    assert "RECONCILED" in section
    assert "Deposits match" in section
    assert "Positions differ due to unrealized P/L" in section


def test_generate_full_integrated_report():
    """Full integrated report should include all sections."""
    # Mock regime result
    regime_result = Mock()
    regime_result.regime = "NEUTRAL"
    regime_result.confidence = 0.75
    regime_result.generated_at = "2024-01-15T12:00:00Z"
    regime_result.primary_signal = "Test signal"
    regime_result.implications = {
        "portfolio_action": "HOLD",
        "tech_allocation": "5%",
        "crypto_allocation": "0%",
        "dca_multiplier": 1.0,
        "cash_target_pct": 10,
        "message": "Test guidance",
    }
    regime_result.price_structure = Mock()
    regime_result.price_structure.nearest_support = 100.0
    regime_result.price_structure.nearest_resistance = 110.0
    regime_result.price_structure.peaks = []
    regime_result.price_structure.valleys = []
    regime_result.price_structure.current_trend = "NEUTRAL"
    regime_result.price_structure.trend_quality = 0.5
    regime_result.price_structure.key_levels = {"support": [], "resistance": []}
    regime_result.timeframes = {
        "daily": {"indicators": {"CLOSE": 105.0, "RSI_14": 50, "SMA_200": 100.0, "SMA_50": 102.0}, "trend": "NEUTRAL"}
    }
    regime_result.news_sentiment = {"sentiment": "neutral", "score": 50, "count": 5, "key_topics": ["test"]}
    regime_result.warnings = []
    
    # Mock exposures
    exposures = [
        _make_mock_exposure("AAPL_US_EQ", 5000, 4000, 1000, ["TechPieShort", "DailyDivPieLongShort"], "WARNING", "estimated"),
        _make_mock_exposure("NVDA_US_EQ", 3000, 3000, 0, ["TechPieShort"], "OK", "inferred"),
    ]
    
    # Mock T212 data (self-consistent: positions 5000+3000+2000=10000,
    # available cash 2000+500=2500, derived 12500 == total 12500 → PASS).
    t212_data = {
        "status": "success",
        "account_summary": {
            "total_equity": 12500,
            "derived_holdings_plus_available_cash": 12500,
            "reconciliation_difference": 0,
            "reconciliation_threshold": 12.5,
            "reconciliation_status": "PASS",
            "unrealized_pnl": 100,
            "unrealized_pnl_calc": 99,
        },
        "cash": {"free": 2000, "pie_cash": 500, "invested": 7500, "blocked": 0},
        "positions": [
            {"symbol": "AAPL_US_EQ", "quantity": 10, "value_eur": 5000, "pnl_eur": 100, "pnl_pct": 2.0, "is_pie_constituent": True},
            {"symbol": "NVDA_US_EQ", "quantity": 5, "value_eur": 3000, "pnl_eur": 50, "pnl_pct": 1.7, "is_pie_constituent": True},
            {"symbol": "MSFT_US_EQ", "quantity": 2, "value_eur": 2000, "pnl_eur": -20, "pnl_pct": -1.0, "is_pie_constituent": False},
        ],
    }
    
    # Historical performance
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "t212_transactions_sanitized.csv"
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    
    report = generate_full_integrated_report(
        regime_result=regime_result,
        t212_data=t212_data,
        ai_recs=None,
        canonical_exposures=exposures,
        total_equity_eur=10000,
        historical_performance=ledger.performance,
        historical_reconciliation=ledger.reconciliation,
    )
    
    # Check all major sections present
    assert "EXI2 Market Regime" in report
    assert "Regime Summary" in report
    assert "Key Levels to Watch" in report
    assert "News" in report
    assert "Portfolio Exposure" in report
    assert "Actionability" in report
    assert "Trading212 Portfolio" in report
    assert "T212 Portfolio" in report  # unified holdings table (exactly one)
    assert "Historical Trading Ledger" in report
    assert "Earnings Radar" in report  # strict month-grouped timeline (replaces calendar table)
    
    # Check specific content
    assert "AAPL_US_EQ" in report
    assert "NVDA_US_EQ" in report
    assert "REVIEW_PIE_ALLOCATION" in report  # AAPL multi-pie estimated
    assert "HOLD_PIE" in report  # NVDA single pie
# T212 portfolio reconciliation (authoritative: positions + cash = derived)
    assert "**Reconciliation:** positions" in report
    assert "→ PASS" in report


def test_report_without_historical_ledger():
    """Report should work without historical ledger data."""
    regime_result = Mock()
    regime_result.regime = "NEUTRAL"
    regime_result.confidence = 0.75
    regime_result.generated_at = "2024-01-15T12:00:00Z"
    regime_result.primary_signal = "Test signal"
    regime_result.implications = {
        "portfolio_action": "HOLD",
        "tech_allocation": "5%",
        "crypto_allocation": "0%",
        "dca_multiplier": 1.0,
        "cash_target_pct": 10,
        "message": "Test guidance",
    }
    regime_result.price_structure = Mock()
    regime_result.price_structure.nearest_support = 100.0
    regime_result.price_structure.nearest_resistance = 110.0
    regime_result.price_structure.peaks = []
    regime_result.price_structure.valleys = []
    regime_result.price_structure.current_trend = "NEUTRAL"
    regime_result.price_structure.trend_quality = 0.5
    regime_result.price_structure.key_levels = {"support": [], "resistance": []}
    regime_result.timeframes = {
        "daily": {"indicators": {"CLOSE": 105.0, "RSI_14": 50, "SMA_200": 100.0, "SMA_50": 102.0}, "trend": "NEUTRAL"}
    }
    regime_result.news_sentiment = {"sentiment": "neutral", "score": 50, "count": 5, "key_topics": ["test"]}
    regime_result.warnings = []
    
    t212_data = {
        "status": "success",
        "account_summary": {
            "total_equity": 10000,
            "derived_holdings_plus_available_cash": 9999,
            "reconciliation_difference": 1,
            "reconciliation_threshold": 10,
            "reconciliation_status": "PASS",
        },
        "cash": {"free": 2000, "pie_cash": 500, "invested": 7500, "blocked": 0},
        "positions": [],
    }
    
    report = generate_full_integrated_report(
        regime_result=regime_result,
        t212_data=t212_data,
        ai_recs=None,
        canonical_exposures=None,
        total_equity_eur=None,
        historical_performance=None,
        historical_reconciliation=None,
    )
    
    assert "EXI2 Market Regime" in report
    assert "Trading212 Portfolio" in report
    assert "Historical Trading Ledger" in report
    assert "not imported" in report  # Should show not imported message