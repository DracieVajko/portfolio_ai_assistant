"""Tests for Trading 212 CSV transaction import."""

import pytest
from decimal import Decimal
from pathlib import Path

from investment_engine.accounting.importers.t212_csv import (
    import_t212_activity_csv,
    import_t212_cash_csv,
    detect_csv_type,
)
from investment_engine.accounting.models import EventType, ParseStatus


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "t212_transactions_sanitized.csv"


def test_detect_csv_type_activity():
    """Activity CSV should be detected as 'activity' type."""
    assert detect_csv_type(FIXTURE_PATH) == "activity"


def test_import_activity_csv_parses_all_rows():
    """All rows in fixture should be parsed into events."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    # Fixture has 13 data rows (header + 13 rows, but one is cash deposit, one withdrawal)
    # Actually all rows are in activity format
    assert len(events) == 13


def test_import_activity_csv_buy_event():
    """Buy event should be parsed correctly."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    buy_events = [e for e in events if e.event_type == EventType.TRADE_BUY]
    assert len(buy_events) == 3  # AAPL, MSFT, VWSBd
    
    aapl_buy = next(e for e in buy_events if e.broker_symbol == "AAPL_US_EQ")
    assert aapl_buy.quantity == Decimal("10")
    assert aapl_buy.price == Decimal("150.00")
    assert aapl_buy.currency == "USD"
    assert aapl_buy.gross_amount == Decimal("1500.00")
    assert aapl_buy.fee_amount == Decimal("1.00")
    assert aapl_buy.fx_fee_amount == Decimal("0.50")
    assert aapl_buy.exchange_rate == Decimal("0.92")
    assert aapl_buy.isin == "US0378331005"


def test_import_activity_csv_sell_event():
    """Sell events should be parsed correctly."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    sell_events = [e for e in events if e.event_type == EventType.TRADE_SELL]
    assert len(sell_events) == 2  # Two AAPL sells
    
    partial_sell = next(e for e in sell_events if e.quantity == Decimal("4"))
    assert partial_sell.broker_symbol == "AAPL_US_EQ"
    assert partial_sell.price == Decimal("160.00")
    assert partial_sell.gross_amount == Decimal("640.00")


def test_import_activity_csv_dividend_and_wht():
    """Dividend and withholding tax should be separate events."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    dividends = [e for e in events if e.event_type == EventType.DIVIDEND]
    wht = [e for e in events if e.event_type == EventType.DIVIDEND_WITHHOLDING_TAX]
    
    assert len(dividends) == 1
    assert len(wht) == 1
    assert dividends[0].gross_amount == Decimal("12.00")
    assert wht[0].tax_amount == Decimal("1.80")
    assert wht[0].net_amount == Decimal("-1.80")


def test_import_activity_csv_stamp_duty_and_ftt():
    """Stamp duty and financial transaction tax should be parsed."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    stamp = [e for e in events if e.event_type == EventType.STAMP_DUTY]
    ftt = [e for e in events if e.event_type == EventType.FINANCIAL_TRANSACTION_TAX]
    
    assert len(stamp) == 1
    assert len(ftt) == 1
    assert stamp[0].tax_amount == Decimal("15.50")
    assert stamp[0].currency == "GBP"
    assert ftt[0].tax_amount == Decimal("2.11")
    assert ftt[0].currency == "EUR"


def test_import_activity_csv_fx_fee():
    """FX conversion fee should be parsed."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    fx_fees = [e for e in events if e.event_type == EventType.FX_CONVERSION_FEE]
    assert len(fx_fees) == 1
    assert fx_fees[0].fx_fee_amount == Decimal("50.00")
    assert fx_fees[0].currency == "USD"


def test_import_activity_csv_unknown_action():
    """Unknown action types should become UNKNOWN events with WARNING status."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    unknown = [e for e in events if e.event_type == EventType.UNKNOWN]
    assert len(unknown) == 1
    assert unknown[0].parse_status == ParseStatus.WARNING
    assert "Unrecognized action type" in unknown[0].parse_warning


def test_import_activity_csv_raw_row_preserved():
    """Raw CSV row should be preserved in event for diagnostics."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    for event in events:
        assert "_raw_row" in dir(event) or hasattr(event, "_raw_row")
        assert isinstance(event._raw_row, dict)
        assert len(event._raw_row) > 0


def test_import_activity_csv_net_amount_calculated():
    """Net amount should be gross - fee - tax - fx_fee."""
    events = import_t212_activity_csv(FIXTURE_PATH)
    aapl_buy = next(e for e in events if e.broker_symbol == "AAPL_US_EQ" and e.is_buy)
    expected_net = Decimal("1500.00") - Decimal("1.00") - Decimal("0.00") - Decimal("0.50")
    assert aapl_buy.net_amount == expected_net


def test_cash_csv_import():
    """Test importing a cash transactions CSV."""
    # Create a minimal cash CSV
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date/Time", "Action", "Currency", "Amount", "Balance", "Reference"])
        writer.writerow(["2023-01-15 10:00:00", "Deposit", "EUR", "5000.00", "5000.00", "REF001"])
        writer.writerow(["2023-01-20 10:00:00", "Withdrawal", "EUR", "-1000.00", "4000.00", "REF002"])
        writer.writerow(["2023-01-31 10:00:00", "Interest", "EUR", "5.25", "4005.25", "REF003"])
        cash_path = f.name
    
    try:
        events = import_t212_cash_csv(cash_path)
        assert len(events) == 3
        
        deposit = next(e for e in events if e.event_type == EventType.CASH_DEPOSIT)
        assert deposit.net_amount == Decimal("5000.00")
        assert deposit.currency == "EUR"
        
        withdrawal = next(e for e in events if e.event_type == EventType.CASH_WITHDRAWAL)
        assert withdrawal.net_amount == Decimal("-1000.00")
        
        interest = next(e for e in events if e.event_type == EventType.CASH_INTEREST)
        assert interest.net_amount == Decimal("5.25")
    finally:
        Path(cash_path).unlink()


def test_detect_csv_type_cash():
    """Cash CSV should be detected as 'cash' type."""
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date/Time", "Action", "Currency", "Amount", "Balance", "Reference"])
        writer.writerow(["2023-01-15 10:00:00", "Deposit", "EUR", "5000.00", "5000.00", "REF001"])
        cash_path = f.name
    
    try:
        assert detect_csv_type(cash_path) == "cash"
    finally:
        Path(cash_path).unlink()


def test_empty_rows_skipped():
    """Empty rows in CSV should be skipped."""
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date/Time", "Action", "Ticker", "Qty", "Price", "Currency", "Amount"])
        writer.writerow(["2023-01-15 10:00:00", "Buy", "AAPL_US_EQ", "10", "150", "USD", "1500"])
        writer.writerow(["", "", "", "", "", "", ""])  # empty row
        writer.writerow(["2023-01-16 10:00:00", "Sell", "AAPL_US_EQ", "5", "160", "USD", "800"])
        path = f.name
    
    try:
        events = import_t212_activity_csv(path)
        assert len(events) == 2
    finally:
        Path(path).unlink()


def test_event_id_stable():
    """Event IDs should be stable for same file/row."""
    events1 = import_t212_activity_csv(FIXTURE_PATH)
    events2 = import_t212_activity_csv(FIXTURE_PATH)
    for e1, e2 in zip(events1, events2):
        assert e1.event_id == e2.event_id


def test_timestamp_parsing():
    """Various timestamp formats should be parsed."""
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date/Time", "Action", "Ticker", "Qty", "Price", "Currency", "Amount"])
        writer.writerow(["2023-01-15 10:30:00", "Buy", "AAPL_US_EQ", "10", "150", "USD", "1500"])
        writer.writerow(["2023-01-15T10:30:00", "Buy", "MSFT_US_EQ", "5", "250", "USD", "1250"])
        writer.writerow(["2023-01-15T10:30:00Z", "Buy", "GOOGL_US_EQ", "2", "2800", "USD", "5600"])
        path = f.name
    
    try:
        events = import_t212_activity_csv(path)
        assert len(events) == 3
        for e in events:
            assert e.timestamp is not None
    finally:
        Path(path).unlink()