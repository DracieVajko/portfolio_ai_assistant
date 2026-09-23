"""Broker-first reconciliation tests: dedupe, aliases, pies, GBX/FX, fallback.

Uses the real 2026-09-11 user transactions (17 Renewables Pie market buys +
2x French transaction tax) as the import fixture, plus synthetic broker
snapshots. No network access.
"""

import pytest

from investment_engine.portfolio.broker_first import (
    KNOWN_ISINS,
    AccountSnapshot,
    BrokerInstrument,
    BrokerPosition,
    BrokerTransaction,
    CashBalance,
    build_account_snapshot,
    build_diagnostic_rows,
    dedupe_key,
    deduplicate_positions,
    diagnostic_summary,
    implied_rate_from_pnl,
    normalize_minor_unit,
    reconcile_snapshot,
    resolve_fx_rate,
    write_diagnostic_csv,
    write_diagnostic_markdown,
)

# Real user transactions, 2026-09-11 (~35 EUR into Renewables Pie).
USER_BUYS_2026_09_11 = [
    ("AI", "FR0000120073", 1.91), ("SU", "FR0000121972", 1.55),
    ("ETN", "IE00B8KQN827", 1.36), ("ENR", "DE000ENER6Y0", 1.83),
    ("NEE", "US65339F1012", 3.40), ("LITM", "IE000WDG5795", 4.22),
    ("HTHIY", "US4335785071", 1.37), ("VWSB", "DK0061539921", 2.46),
    ("APD", "US0091581068", 1.79), ("IQQH", "IE00B1XNHC34", 3.50),
    ("RWE", "DE0007037129", 1.83), ("IBE", "ES0144580Y14", 1.68),
    ("GEV", "US36828A1016", 1.63), ("UEC", "US9168961038", 1.70),
    ("C7A0", "CNE100006WS8", 3.07), ("LIN", "IE000S9YS762", 1.63),
]


def _raw_pos(ticker, qty, avg, cur, ppl=0.0, fx_ppl=0.0, pie_qty=0.0):
    return {"ticker": ticker, "quantity": qty, "averagePrice": avg,
            "currentPrice": cur, "ppl": ppl, "fxPpl": fx_ppl,
            "pieQuantity": pie_qty}


def _mkpos(ticker, isin=None, value=10.0):
    return BrokerPosition(
        instrument=BrokerInstrument(broker_instrument_id=ticker, broker_ticker=ticker,
                                    display_symbol=ticker.split("_")[0], isin=isin),
        quantity=1.0, broker_market_value_eur=value)


# --- 1. ISIN ground truth ----------------------------------------------------

def test_known_isins_cover_user_transactions():
    for display, isin, _amount in USER_BUYS_2026_09_11:
        assert KNOWN_ISINS.get(display) == isin, display


# --- 2. Dedupe by ISIN ---------------------------------------------------------

def test_dedupe_by_isin_same_instrument_two_tickers():
    a = _mkpos("VWSBd_EQ", isin="DK0061539921", value=30.0)
    b = _mkpos("VWSB", isin="DK0061539921", value=30.0)
    unique, duplicates = deduplicate_positions([a, b])
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert sum(p.broker_market_value_eur for p in unique) == pytest.approx(30.0)
    assert duplicates[0].duplicate_of != ""
    assert "not summed" in duplicates[0].exclusion_reason


def test_dedupe_falls_back_to_broker_id_without_isin():
    a = _mkpos("AAA_US_EQ", value=10.0)
    b = _mkpos("AAA_US_EQ", value=10.0)
    c = _mkpos("BBB_US_EQ", value=5.0)
    unique, duplicates = deduplicate_positions([a, b, c])
    assert len(unique) == 2 and len(duplicates) == 1
    assert dedupe_key("default", None, "AAA_US_EQ") == ("default", "AAA_US_EQ")


# --- 3. Ticker alias nesmie vytvoriť druhú pozíciu ------------------------------

def test_ticker_alias_never_creates_second_position():
    # Rovnaký instrument (ISIN DK0061539921) pod dvoma broker tickermi sa
    # započíta práve raz; duplikát nesie konkrétny dôvod.
    snap = build_account_snapshot(
        {"free": 0.0, "pieCash": 0.0, "blocked": 0.0, "invested": 60.0, "total": 60.0, "ppl": 0.0},
        [_raw_pos("VWSBd_EQ", 1.0, 27.0, 28.0), _raw_pos("VWSB", 1.0, 27.0, 28.0)],
        None, live_rates={"EUR": 1.0},
    )
    unique, duplicates = deduplicate_positions(snap.positions)
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert "not summed" in duplicates[0].exclusion_reason
    assert duplicates[0].duplicate_of != ""
    recon = reconcile_snapshot(snap)
    assert recon.sum_deduplicated_broker_position_values_eur == pytest.approx(28.0)
    assert recon.sum_excluded_values_eur == pytest.approx(0.0)  # VWSB mala UNRESOLVED menu -> hodnota None


# --- 4. Pie total + komponenty sa nesmú sčítať -----------------------------------

def test_pie_total_and_components_never_double_counted():
    snap = build_account_snapshot(
        {"free": 10.0, "pieCash": 0.0, "blocked": 0.0, "invested": 100.0, "total": 110.0, "ppl": 0.0},
        [_raw_pos("AAPL_US_EQ", 1.0, 100.0, 100.0, pie_qty=1.0),
         _raw_pos("MSFT_US_EQ", 2.0, 50.0, 50.0, pie_qty=2.0)],
        [{"id": "pie1", "instruments": [
            {"ticker": "AAPL_US_EQ", "ownedQuantity": 1.0},
            {"ticker": "MSFT_US_EQ", "ownedQuantity": 2.0}]}],
        live_rates={"USD": 1.0, "EUR": 1.0},
    )
    assert len(snap.pies) == 1
    assert snap.pies[0].member_instrument_ids == ["AAPL_US_EQ", "MSFT_US_EQ"]
    recon = reconcile_snapshot(snap)
    # 1x100 + 2x50 = 200, pie total nikde (endpoint ho ani nedáva).
    assert recon.sum_deduplicated_broker_position_values_eur == pytest.approx(200.0)


# --- 5. GBX normalizácia ----------------------------------------------------------

def test_gbx_divides_by_100_before_fx():
    price, factor = normalize_minor_unit(1350.0, "GBX")
    assert price == pytest.approx(13.5)
    assert factor == 100
    # EGTl_EQ: 1151.9 ks x 13.50 GBX -> GBP -> EUR pri kurze 1.1651
    qty, gbx, rate = 1151.9, 13.5, 1.1651
    assert qty * (gbx / 100.0) * rate == pytest.approx(181.18, abs=0.05)


# --- 6. USD/EUR sa nesmie aplikovať na broker EUR hodnotu ----------------------------

def test_no_fx_on_broker_eur_values():
    snap = build_account_snapshot(
        {"free": 0.0, "pieCash": 0.0, "blocked": 0.0, "invested": 100.0, "total": 100.0, "ppl": 0.0},
        [_raw_pos("IBEe_EQ", 1.0, 20.0, 20.0)],
        None, live_rates={"USD": 0.5, "EUR": 1.0},
    )
    pos = snap.positions[0]
    assert pos.instrument.currency == "EUR"
    assert pos.broker_market_value_eur == pytest.approx(20.0)
    assert pos.valuation_source == "broker_value"
    assert pos.fx_audit.rate == pytest.approx(1.0)


def test_broker_implied_rate_calibration():
    # AAPL 2026-09-11: r1 = (8.43 - 0.22) / (0.5214 x 18.2647) ≈ 0.8621
    rate = implied_rate_from_pnl(0.5214, 310.3853, 328.65, 8.43, 0.22)
    assert rate == pytest.approx(0.8621, abs=0.002)
    # Degenerate: nulový pohyb ceny -> None (šum zo zaokrúhlenia).
    assert implied_rate_from_pnl(1.0, 100.0, 100.0, 0.01, 0.0) is None


# --- 7. Fallback nikdy nepremôže live API dáta ---------------------------------------

def test_fallback_never_overrides_live_api():
    snap_live = build_account_snapshot(
        {"free": 1.0, "pieCash": 0.0, "blocked": 0.0, "invested": 10.0, "total": 11.0, "ppl": 0.0},
        [_raw_pos("AAPL_US_EQ", 1.0, 100.0, 110.0)],
        None, live_rates={"USD": 0.86, "EUR": 1.0},
    )
    live_val = snap_live.positions[0].broker_market_value_eur
    snap_fallback = build_account_snapshot(
        {"free": 1.0, "pieCash": 0.0, "blocked": 0.0, "invested": 10.0, "total": 11.0, "ppl": 0.0},
        [_raw_pos("AAPL_US_EQ", 1.0, 100.0, 110.0)],
        None, live_rates={},
    )
    # Bez live kurzu: static fallback s flagom, nikdy ticho rovnaká hodnota.
    assert snap_fallback.positions[0].mapping_status == "FX_STATIC_FALLBACK"
    assert live_val == pytest.approx(94.6)


# --- 8. Reconciliation: expected 2412.62, delta +89.93 ----------------------------------

def test_expected_positions_value_from_user_numbers():
    snap = AccountSnapshot(
        retrieved_at="t",
        cash=__import__("investment_engine.portfolio.broker_first", fromlist=["CashBalance"]).CashBalance(
            free_cash_eur=833.00, pie_cash_eur=0.60, blocked_cash_eur=31.04,
            broker_total_equity_eur=3246.22),
        positions=[],
    )
    recon = reconcile_snapshot(snap)
    # expected = total - free - pieCash - blocked = 3246.22 - 833.00 - 0.60 - 31.04 = 2381.58
    assert recon.expected_open_positions_value_eur == pytest.approx(2381.58)
    # Tolerancia: min(0.1 % z 3246.22, 2.00 EUR) ≈ 2.00 EUR.
    assert recon.tolerance_eur == pytest.approx(2.00)


def test_delta_plus_89_93_is_detected_and_never_a_loss():
    from investment_engine.portfolio.broker_first import CashBalance
    snap = AccountSnapshot(
        retrieved_at="t",
        cash=CashBalance(free_cash_eur=833.00, pie_cash_eur=0.60,
                         broker_total_equity_eur=3246.22),
        positions=[_mkpos("X", value=2502.55)],
    )
    recon = reconcile_snapshot(snap)
    assert recon.reconciliation_delta_eur == pytest.approx(89.93)
    assert recon.reconciliation_status == "FAIL"
    assert recon.data_quality == "DATA_QUALITY_FAIL"
    assert "never a reported loss" in " ".join(recon.notes)


# --- 9. User transakcie 11.9. ako import test --------------------------------------------

def test_user_september_buys_import():
    from investment_engine.portfolio.broker_first import BrokerTransaction
    txns = [BrokerTransaction(date="2026-09-11", flow_type="BUY", ticker=d, isin=i,
                              original_amount=a, original_currency="EUR",
                              amount_eur=a, fx_to_eur=1.0, verified=True)
            for d, i, a in USER_BUYS_2026_09_11]
    assert len(txns) == 16
    assert sum(t.original_amount for t in txns) == pytest.approx(34.92, abs=0.05)
    assert all(t.isin and t.verified for t in txns)
    taxes = [BrokerTransaction(date="2026-09-11", flow_type="FEE", ticker=d, isin=i,
                               original_amount=0.01, original_currency="EUR",
                               amount_eur=0.01, fx_to_eur=1.0, verified=True)
             for d, i in (("AI", "FR0000120073"), ("SU", "FR0000121972"))]
    assert sum(t.original_amount for t in taxes) == pytest.approx(0.02)
    # ISIN každého nákupu sedí s ground-truth mapou.
    for t in txns:
        assert KNOWN_ISINS.get(t.ticker) == t.isin, t.ticker


# --- 10. Diagnostický report ---------------------------------------------------------------

def test_catalog_currency_beats_heuristic():
    # UEC_US_EQ má fxPpl=0 (heuristika by dala EUR), katalóg hovorí USD.
    catalog = {"UEC_US_EQ": {"isin": "US9168961038", "currencyCode": "USD"}}
    snap = build_account_snapshot(
        {"free": 0.0, "pieCash": 0.0, "blocked": 0.0, "invested": 20.0, "total": 20.0, "ppl": 0.0},
        [_raw_pos("UEC_US_EQ", 1.1204, 11.8349, 11.11, ppl=-0.70, fx_ppl=0.0)],
        None, live_rates={"USD": 0.86, "EUR": 1.0}, catalog=catalog,
    )
    pos = snap.positions[0]
    assert pos.instrument.currency == "USD"
    assert pos.broker_market_value_eur == pytest.approx(1.1204 * 11.11 * 0.86, abs=0.05)
    assert "HEURISTIC" not in pos.mapping_status


def test_gbx_rate_lookup_uses_gbp():
    from investment_engine.portfolio.broker_first import resolve_fx_rate
    rate, source = resolve_fx_rate("GBX", {"GBP": 1.1651, "USD": 0.86, "EUR": 1.0})
    assert rate == pytest.approx(1.1651)
    assert source == "live-fx"
    # Bez live kurzu padne na označený static fallback (nie ticho None).
    rate2, source2 = resolve_fx_rate("GBX", {})
    assert rate2 == pytest.approx(1.183)
    assert source2 == "static-fallback"
    rate3, _ = resolve_fx_rate("XXX", {})
    assert rate3 is None


def test_parse_position_gbx_uses_gbp_rate():
    from trading212_portfolio import parse_position
    catalog = {"EGTl_EQ": {"isin": "GB00BPVG5407", "currencyCode": "GBX"}}
    parsed = parse_position(
        {"ticker": "EGTl_EQ", "quantity": 100.0, "averagePrice": 14.0,
         "currentPrice": 13.5, "ppl": -0.5, "fxPpl": -0.1, "pieQuantity": 0.0},
        fx_rates={"GBP": 1.1651, "USD": 0.86, "EUR": 1.0}, catalog=catalog)
    # 100 x (13.5/100) x 1.1651 = 15.73 (nie x1.0 ani x100).
    assert parsed["value_eur"] == pytest.approx(15.73, abs=0.02)
    assert parsed["quote_currency"] == "GBX"
    assert parsed["currency_source"] == "broker_metadata"
    assert parsed["isin"] == "GB00BPVG5407"


def test_broker_internal_consistency_section():
    from investment_engine.portfolio.broker_first import AccountSnapshot, CashBalance
    snap = AccountSnapshot(
        retrieved_at="t",
        cash=CashBalance(free_cash_eur=833.00, pie_cash_eur=0.60,
                         broker_invested_eur=2407.47, broker_total_equity_eur=3246.22),
        positions=[],
    )
    rows, summary = build_diagnostic_rows(snap)
    bic = summary["broker_internal_consistency"]
    assert bic["invested_plus_free_plus_pie_minus_total"] == pytest.approx(-5.15)
    assert "separate" in bic["note"]


def test_empty_snapshot_never_passes():
    from investment_engine.portfolio.broker_first import AccountSnapshot, CashBalance
    snap = AccountSnapshot(retrieved_at="t", cash=CashBalance(), positions=[])
    recon = reconcile_snapshot(snap)
    assert recon.reconciliation_status == "UNKNOWN"
    assert recon.data_quality == "NO_BROKER_DATA"
    assert recon.sum_deduplicated_broker_position_values_eur == pytest.approx(0.0)


def test_catalog_isin_lookup():
    from investment_engine.portfolio.broker_first import lookup_instrument
    catalog = {"AAPL_US_EQ": {"isin": "US0378331005", "currencyCode": "USD"}}
    assert lookup_instrument("AAPL_US_EQ", catalog)["isin"] == "US0378331005"
    assert lookup_instrument("ZZZ_NOT_EXIST_EQ", catalog) == {}
    # Bez katalógu sa použije lokálna cache (ak existuje), inak prázdno.
    assert isinstance(lookup_instrument("ZZZ_NOT_EXIST_EQ", {}), dict)


def test_diagnostic_csv_and_markdown(tmp_path):
    from investment_engine.portfolio.broker_first import CashBalance
    snap = AccountSnapshot(
        retrieved_at="t",
        cash=CashBalance(free_cash_eur=833.00, pie_cash_eur=0.60,
                         broker_total_equity_eur=3246.22),
        positions=[_mkpos("AAPL_US_EQ", isin="US0378331005", value=158.01),
                   _mkpos("VWSBd_EQ", isin="DK0061539921", value=30.33)],
    )
    rows, summary = build_diagnostic_rows(snap)
    assert len(rows) == 2
    assert set(rows[0].keys()) >= {"dedupe_key", "mapping_status", "valuation_source",
                                   "included_in_position_total", "exclusion_reason"}
    csv_path = write_diagnostic_csv(rows, tmp_path / "diag.csv")
    md_path = write_diagnostic_markdown(rows, summary, tmp_path / "diag.md")
    assert csv_path.exists() and md_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "dedupe_key" in header and "fx_source" in header
    text = md_path.read_text(encoding="utf-8")
    for key in ("broker_total_equity_eur", "expected_open_positions_value_eur",
                "sum_deduplicated_broker_position_values_eur",
                "reconciliation_delta_eur", "reconciliation_status"):
        assert key in text
