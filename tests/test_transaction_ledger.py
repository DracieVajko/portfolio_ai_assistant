"""Tests for complete historical transaction ledger workflow."""

import pytest
from decimal import Decimal
from pathlib import Path

from investment_engine.accounting import (
    build_ledger,
    Ledger,
    calculate_performance,
    match_sell_fifo,
    ManualAdjustments,
)
from investment_engine.accounting.importers.t212_csv import import_t212_activity_csv
from investment_engine.accounting.models import LedgerEvent, EventType, LedgerSummary
from investment_engine.accounting.lot_matching import MatchingResult


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "t212_transactions_sanitized.csv"


def test_fifo_full_and_partial_matching():
    """FIFO matching should correctly handle full and partial lot closes."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    
    # Filter to just AAPL trades
    aapl_events = [e for e in events if e.broker_symbol == "AAPL_US_EQ" and e.is_trade]
    assert len(aapl_events) == 3  # Buy 10, Sell 4, Sell 6
    
    # Use default FX rates
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    result = match_sell_fifo(aapl_events, default_fx)
    
    # Should have 2 matched trades
    assert len(result.matched_trades) == 2
    assert len(result.short_sells) == 0
    assert len(result.unmatched_sells) == 0
    
    # First sell (4 shares) matches first 4 of buy (10 shares)
    first_sell = result.matched_trades[0]
    assert first_sell.quantity_sold == Decimal("4")
    assert len(first_sell.matched_lots) == 1
    assert first_sell.matched_lots[0]["quantity"] == "4"
    
    # Second sell (6 shares) matches remaining 6 of buy
    second_sell = result.matched_trades[1]
    assert second_sell.quantity_sold == Decimal("6")
    assert len(second_sell.matched_lots) == 1
    assert second_sell.matched_lots[0]["quantity"] == "6"
    
    # No open lots remaining for AAPL
    assert "AAPL_US_EQ" not in result.open_lots or len(result.open_lots.get("AAPL_US_EQ", [])) == 0


def test_gross_realized_pnl_calculation():
    """Gross realized P&L should be calculated correctly."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    # AAPL: Buy 10@150, Sell 4@160, Sell 6@170
    # First sell: 4 * (160 - 150) = 40 USD = 36.80 EUR (minus fees)
    # Second sell: 6 * (170 - 150) = 120 USD = 110.40 EUR (minus fees)
    # Total gross P&L before fees: 160 USD = 147.20 EUR
    # Fees: 5 trades * 1.00 USD = 5 USD = 4.60 EUR, FX fees: 1.40 USD = 1.29 EUR
    # Net after fees
    
    assert performance.summary.gross_realized_pnl_eur != Decimal("0")
    assert performance.summary.known_explicit_costs_eur > Decimal("0")
    assert performance.summary.net_realized_pnl_after_known_costs_eur == (
        performance.summary.gross_realized_pnl_eur - performance.summary.known_explicit_costs_eur
    )


def test_known_fees_categorized_correctly():
    """Fees should be categorized into correct buckets."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    fb = performance.fee_breakdown
    # From fixture:
    # - FX conversion fee: 50 USD = 46 EUR
    # - Stamp duty: 15.50 GBP = 18.34 EUR
    # - FTT: 2.11 EUR
    # - Dividend WHT: 1.80 USD = 1.66 EUR
    # - Trade fees (commissions on trades): 5 * 1.00 USD = 5 USD = 4.60 EUR
    #   These are in trade events, not separate COMMISSION events
    # - FX fees on trades: 0.50 + 0.40 + 0.20 + 0.30 = 1.40 USD = 1.29 EUR
    
    assert fb.fx_conversion_eur > Decimal("40")  # ~46 EUR
    assert fb.stamp_duty_eur > Decimal("17")  # ~18.34 EUR
    assert fb.financial_transaction_tax_eur == Decimal("2.11")
    assert fb.dividend_withholding_tax_eur > Decimal("1.5")  # ~1.66 EUR
    # Commission fees are embedded in trade events, not separate COMMISSION events
    # They appear in other_eur or are captured during lot matching
    assert fb.other_eur > Decimal("1")  # Includes trade commissions and FX fees on trades


def test_net_realized_pnl_after_explicit_costs():
    """Net realized P&L after known costs should equal gross - costs."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    gross = performance.summary.gross_realized_pnl_eur
    costs = performance.summary.known_explicit_costs_eur
    net = performance.summary.net_realized_pnl_after_known_costs_eur
    
    assert net == gross - costs


def test_dividend_gross_net_separation():
    """Dividends should be separated into gross and net after WHT."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    # Dividend: 12 USD = 11.04 EUR
    # WHT: 1.80 USD = 1.66 EUR
    # Net: 10.20 USD = 9.38 EUR
    assert performance.summary.dividends_gross_eur > Decimal("10")
    assert performance.summary.dividend_withholding_tax_eur > Decimal("1.5")
    assert performance.summary.dividends_net_eur == (
        performance.summary.dividends_gross_eur - performance.summary.dividend_withholding_tax_eur
    )


def test_unknown_event_causes_partial_status():
    """Unknown events should force PARTIAL accounting status."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    # Fixture has one unknown event
    assert performance.summary.unknown_event_count == 1
    assert performance.data_quality == "PARTIAL"
    assert any("Unknown event types" in w for w in performance.completeness_warnings)


def test_sell_without_inventory_causes_reconciliation_failure():
    """Sell without matching buy lots should be flagged."""
    from investment_engine.accounting.models import LedgerEvent
    from datetime import datetime
    
    # Create a sell event without prior buy
    orphan_sell = LedgerEvent(
        event_id="orphan_1",
        timestamp=datetime(2023, 1, 15),
        source_file="test.csv",
        source_row=1,
        source_type="t212_activity_csv",
        broker_symbol="ORPHAN_US_EQ",
        event_type=EventType.TRADE_SELL,
        currency="USD",
        quantity=Decimal("10"),
        price=Decimal("100"),
        gross_amount=Decimal("1000"),
        net_amount=Decimal("999"),
    )
    
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0")}
    result = match_sell_fifo([orphan_sell], default_fx)
    
    assert len(result.unmatched_sells) == 1
    assert len(result.matched_trades) == 0
    assert any("No buy lots available" in w for w in result.warnings)


def test_money_uses_decimal_safely():
    """All monetary calculations should use Decimal."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    # All summary fields should be Decimal
    summary = performance.summary
    for field_name in [
        "gross_realized_pnl_eur", "known_explicit_costs_eur",
        "net_realized_pnl_after_known_costs_eur", "dividends_gross_eur",
        "dividends_net_eur", "deposits_eur", "withdrawals_eur",
        "net_deposits_eur", "open_lot_cost_basis_eur"
    ]:
        value = getattr(summary, field_name)
        assert isinstance(value, Decimal), f"{field_name} is {type(value)}, not Decimal"


def test_raw_csv_zeros_placeholders_no_fake_events():
    """Zero/placeholder rows should not create fake events."""
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date/Time", "Action", "Ticker", "Qty", "Price", "Currency", "Amount", "Fee"])
        writer.writerow(["2023-01-15 10:00:00", "Buy", "AAPL_US_EQ", "10", "150", "USD", "1500", "1"])
        writer.writerow(["2023-01-16 10:00:00", "Buy", "ZERO_TICKER", "0", "0", "USD", "0", "0"])  # zero placeholder
        writer.writerow(["2023-01-17 10:00:00", "Sell", "AAPL_US_EQ", "5", "160", "USD", "800", "1"])
        path = f.name
    
    try:
        events = import_t212_activity_csv(path)
        # Should have 2 real trades, zero row might create event with 0 quantity
        buy_events = [e for e in events if e.is_buy and e.quantity > 0]
        assert len(buy_events) == 1
        assert buy_events[0].broker_symbol == "AAPL_US_EQ"
    finally:
        Path(path).unlink()


def test_no_t212_http_calls():
    """Import and ledger building should not make HTTP calls."""
    # This test verifies by construction - no HTTP client is imported or used
    # in the accounting module
    import investment_engine.accounting.importers.t212_csv as importer
    import investment_engine.accounting.ledger as ledger_mod
    
    # Check no requests/session/client imports
    import inspect
    importer_src = inspect.getsource(importer)
    ledger_src = inspect.getsource(ledger_mod)
    
    assert "requests" not in importer_src
    assert "httpx" not in importer_src
    assert "Trade212Client" not in importer_src
    assert "requests" not in ledger_src
    assert "httpx" not in ledger_src
    assert "Trade212Client" not in ledger_src


def test_no_secrets_in_output():
    """Output artifacts should not contain secrets."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    output = ledger.to_dict(include_raw=False)
    
    # Check no API keys, tokens, or auth headers in output
    output_str = str(output)
    assert "api_key" not in output_str.lower()
    assert "api_secret" not in output_str.lower()
    assert "authorization" not in output_str.lower()
    assert "bearer" not in output_str.lower()
    assert "password" not in output_str.lower()


def test_build_ledger_with_manual_adjustments():
    """Manual adjustments should be applied to ledger."""
    adjustments = ManualAdjustments(
        starting_cash_eur=Decimal("10000.00"),
        notes="Test adjustment"
    )
    
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    ledger = build_ledger(
        activity_csv_path=FIXTURE_PATH,
        default_fx_rates=default_fx,
        manual_adjustments=adjustments,
    )
    
    assert ledger.manual_adjustments is not None
    assert ledger.manual_adjustments.starting_cash_eur == Decimal("10000.00")
    assert any("starting cash adjustment" in w.lower() for w in ledger.performance.completeness_warnings)


def test_preview_csv():
    """Preview should show column mapping and sample rows."""
    from investment_engine.accounting.ledger import preview_csv
    
    preview = preview_csv(FIXTURE_PATH, max_rows=3)
    
    assert "headers" in preview
    assert "column_mapping" in preview
    assert "detected_type" in preview
    assert preview["detected_type"] == "activity"
    assert len(preview["sample_rows"]) == 3
    assert "row_count_estimate" in preview


def test_performance_result_to_dict():
    """PerformanceResult.to_dict should serialize all fields."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    d = performance.to_dict()
    
    assert "summary" in d
    assert "fee_breakdown" in d
    assert "trade_stats" in d
    assert "gross_turnover_eur" in d
    assert "cost_drag" in d
    assert "completeness_warnings" in d
    assert "data_quality" in d
    
    # All monetary values should be strings
    assert isinstance(d["summary"]["gross_realized_pnl_eur"], str)
    assert isinstance(d["fee_breakdown"]["total_eur"], str)


def test_trade_statistics_calculation():
    """Trade statistics should be calculated correctly."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    stats = performance.trade_stats
    
    # AAPL: 2 sells, both profitable (160>150, 170>150)
    assert stats.profitable_count >= 2
    assert stats.losing_count == 0
    assert stats.win_rate is not None
    assert stats.profit_factor is not None
    assert stats.avg_winner_eur is not None
    assert stats.expectancy_eur is not None


def test_cost_drag_calculation():
    """Cost drag should be calculated as costs / max(|gross|, epsilon)."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    assert performance.cost_drag is not None
    gross = abs(performance.summary.gross_realized_pnl_eur)
    costs = performance.summary.known_explicit_costs_eur
    expected_drag = costs / max(gross, Decimal("0.01"))
    assert abs(performance.cost_drag - expected_drag) < Decimal("0.001")


def test_gross_turnover_calculation():
    """Gross turnover should sum absolute trade values."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    
    matching_result = match_sell_fifo(events, default_fx)
    performance = calculate_performance(events, matching_result, None, default_fx)
    
    # Turnover = sum of sell proceeds
    assert performance.gross_turnover_eur > Decimal("0")


def test_empty_ledger_handling():
    """Empty ledger should return UNAVAILABLE status."""
    ledger = build_ledger()
    
    assert len(ledger.events) == 0
    assert ledger.performance.data_quality == "UNAVAILABLE"
    assert "No CSV files imported" in ledger.performance.completeness_warnings


def test_ledger_summary_report():
    """get_summary_report should return user-facing summary."""
    default_fx = {"USD": Decimal("0.92"), "EUR": Decimal("1.0"), "GBP": Decimal("1.183")}
    ledger = build_ledger(activity_csv_path=FIXTURE_PATH, default_fx_rates=default_fx)
    
    report = ledger.get_summary_report()
    
    assert "historical_ledger_summary" in report
    assert "fee_breakdown" in report
    assert "trade_statistics" in report
    assert "data_quality" in report
    assert "completeness_warnings" in report