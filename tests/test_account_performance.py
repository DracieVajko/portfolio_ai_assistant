"""Focused tests: account performance, multi-currency, reconciliation interplay.

Covers the F.1-F.6 matrix. Offline and deterministic: network fetchers are
stubbed; the yfinance historical-rate path is injected, never called live.
"""

from datetime import date

import pytest

from investment_engine.accounting import cashflows as cf


def _rate_stub(rates):
    def _fetch(ccy, day):
        if ccy in rates:
            return rates[ccy], f"stub {ccy} {day}"
        return None, "unresolved"
    return _fetch


RATES = {"USD": 0.92, "GBP": 1.18}


# --- F.1 manual baseline ------------------------------------------------------

def test_manual_baseline_pnl_and_return():
    perf = cf.build_account_performance(
        3214.24, 863.93, 0.02, 0.0, 3201.51,
        "Manual verified baseline", "MANUAL")
    assert perf["net_pnl_after_costs_eur"] == pytest.approx(12.73)
    assert perf["return_pct"] == pytest.approx(0.3976, abs=1e-3)
    assert perf["performance_status"] == "Available"
    assert perf["reported_cash_eur"] == pytest.approx(863.95)
    assert perf["cash_allocation_pct"] == pytest.approx(863.95 / 3214.24 * 100, abs=0.01)


def test_no_fee_double_count():
    base = dict(broker_total_equity_eur=3214.24, reported_free_cash_eur=863.93,
                pie_cash_eur=0.02, blocked_cash_eur=0.0, net_deposits_eur=3201.51,
                cash_flow_source="Manual verified baseline", cash_flow_status="MANUAL")
    a = cf.build_account_performance(**base, fees_eur=0.0)
    b = cf.build_account_performance(**base, fees_eur=50.0, interest_eur=3.0, tax_eur=1.0)
    assert a["net_pnl_after_costs_eur"] == b["net_pnl_after_costs_eur"] == pytest.approx(12.73)
    assert b["fees_eur"] == pytest.approx(50.0)  # display-only, retained


def test_manual_baseline_file():
    from pathlib import Path
    manual = cf.load_manual_baseline(Path("data/portfolio_performance.toml"))
    assert manual is not None
    assert manual["net_deposits_eur"] == pytest.approx(3201.51)


# --- F.2 API ledger flows ------------------------------------------------------

def test_deposit_withdrawal_net_and_ignored_types():
    raws = [
        {"type": "DEPOSIT", "amount": 1000.0, "currency": "EUR", "date": "2026-01-05"},
        {"type": "deposit", "amount": 500.0, "currency": "EUR", "date": "2026-02-05"},
        {"type": "WITHDRAW", "amount": 200.0, "currency": "EUR", "date": "2026-03-05"},
        {"type": "BUY", "amount": 999.0, "currency": "EUR", "date": "2026-03-06"},
        {"type": "SELL", "amount": 999.0, "currency": "EUR", "date": "2026-03-07"},
        {"type": "FEE", "amount": 2.5, "currency": "EUR", "date": "2026-03-08"},
        {"type": "MYSTERY_TYPE", "amount": 777.0, "currency": "EUR", "date": "2026-03-09"},
    ]
    items = [cf.normalize_cashflow_item(r, rate_fetcher=_rate_stub(RATES)) for r in raws]
    summary = cf.summarize_cashflows(items)
    assert summary["total_deposits_eur"] == pytest.approx(1500.0)
    assert summary["total_withdrawals_eur"] == pytest.approx(200.0)
    assert summary["net_deposits_eur"] == pytest.approx(1300.0)
    assert summary["fees_eur"] == pytest.approx(2.5)
    assert summary["unknown_count"] == 3  # MYSTERY_TYPE + BUY/SELL-shaped unknowns
    assert set(summary) >= {"total_deposits_eur", "net_deposits_eur", "fees_eur"}


def test_unknown_type_never_silent_deposit():
    items = [cf.normalize_cashflow_item(
        {"type": "SOMETHING_NEW", "amount": 5000.0, "currency": "EUR", "date": "2026-01-01"},
        rate_fetcher=_rate_stub(RATES))]
    assert items[0]["type"] == "unknown"
    summary = cf.summarize_cashflows(items)
    assert summary["net_deposits_eur"] == pytest.approx(0.0)
    assert summary["unknown_count"] == 1


# --- F.3 source priority --------------------------------------------------------

def _api_result(status, net=1500.0, dep=1700.0, wd=200.0):
    return {"status": status, "pages": 2, "exhausted": status == "VERIFIED",
            "item_count": 3, "summary": {
                "total_deposits_eur": dep, "total_withdrawals_eur": wd,
                "net_deposits_eur": net, "fees_eur": 1.0, "interest_eur": 0.5,
                "unresolved_count": 0, "unknown_count": 0, "item_count": 3}}


def test_source_priority():
    manual = {"net_deposits_eur": 3201.51}
    full = cf.resolve_net_deposits(_api_result("VERIFIED"), manual)
    assert full["net_deposits_eur"] == pytest.approx(1500.0)
    assert full["net_deposits_source"] == cf.API_SOURCE
    partial = cf.resolve_net_deposits(_api_result("PARTIAL"), manual)
    assert partial["net_deposits_eur"] == pytest.approx(3201.51)
    assert partial["net_deposits_source"] == cf.MANUAL_SOURCE
    none = cf.resolve_net_deposits({"status": "UNAVAILABLE"}, None)
    assert none["net_deposits_eur"] is None
    assert none["net_deposits_source"] == cf.UNAVAILABLE_SOURCE
    perf = cf.build_account_performance(
        3214.24, 0.0, 0.0, 0.0, none["net_deposits_eur"],
        none["net_deposits_source"], none["net_deposits_status"])
    assert perf["performance_status"] == "Unavailable"
    assert perf["net_pnl_after_costs_eur"] is None


def test_coverage_downgrade_partial_window():
    api = dict(_api_result("VERIFIED"))
    api["items"] = [{"date": "2026-07-30", "type": "DEPOSIT", "original_amount": 1.0,
                     "original_currency": "EUR", "fx_to_eur": 1.0, "fx_rate_source": "EUR",
                     "amount_eur": 1.0, "source": "t212_api", "verified": True,
                     "conversion_status": "ok"}]
    covered = cf.resolve_net_deposits(api, {"net_deposits_eur": 3201.51}, account_start="2026-07-30")
    assert covered["net_deposits_source"] == cf.API_SOURCE
    gapped = cf.resolve_net_deposits(api, {"net_deposits_eur": 3201.51}, account_start="2026-04-10")
    assert gapped["net_deposits_source"] == cf.MANUAL_SOURCE
    assert gapped["net_deposits_eur"] == pytest.approx(3201.51)


# --- F.4 currency handling -------------------------------------------------------

def test_currency_conversions():
    eur = cf.convert_to_eur(100.0, "EUR", "2026-01-05", rate_fetcher=_rate_stub(RATES))
    assert eur[0] == pytest.approx(100.0) and eur[1] == pytest.approx(1.0) and eur[3] == "ok"
    usd = cf.convert_to_eur(100.0, "USD", "2026-01-05", rate_fetcher=_rate_stub(RATES))
    assert usd[0] == pytest.approx(92.0)
    gbp = cf.convert_to_eur(100.0, "GBP", "2026-01-05", rate_fetcher=_rate_stub(RATES))
    assert gbp[0] == pytest.approx(118.0)
    gbx = cf.convert_to_eur(1000.0, "GBX", "2026-01-05", rate_fetcher=_rate_stub(RATES))
    assert gbx[0] == pytest.approx(11.8)  # /100 first, then GBP rate
    missing = cf.convert_to_eur(100.0, "JPY", "2026-01-05", rate_fetcher=_rate_stub(RATES))
    assert missing[3] == "unresolved" and missing[0] is None


def test_mixed_currency_never_sums_raw():
    raws = [
        {"type": "DEPOSIT", "amount": 1000.0, "currency": "EUR", "date": "2026-01-05"},
        {"type": "DEPOSIT", "amount": 1000.0, "currency": "USD", "date": "2026-01-05"},
        {"type": "DEPOSIT", "amount": 50000.0, "currency": "GBX", "date": "2026-01-05"},
    ]
    items = [cf.normalize_cashflow_item(r, rate_fetcher=_rate_stub(RATES)) for r in raws]
    summary = cf.summarize_cashflows(items)
    # 1000 + 920 + 590 — never 1000+1000+50000.
    assert summary["total_deposits_eur"] == pytest.approx(2510.0)


def test_unresolved_fx_blocks_verified():
    raws = [{"type": "DEPOSIT", "amount": 100.0, "currency": "JPY", "date": "2026-01-05"}]
    result = cf.ingest_history_pages([{"items": raws}], True, rate_fetcher=_rate_stub(RATES))
    assert result["status"] == "PARTIAL"
    resolved = cf.resolve_net_deposits(result, {"net_deposits_eur": 3201.51})
    assert resolved["net_deposits_source"] == cf.MANUAL_SOURCE


def test_safe_ledger_fields_only():
    item = cf.normalize_cashflow_item(
        {"type": "DEPOSIT", "amount": 10.0, "currency": "EUR", "date": "2026-01-01",
         "reference": "SECRET-REF", "accountId": "12345", "id": "abc"},
        rate_fetcher=_rate_stub(RATES))
    assert set(item) <= cf.SAFE_LEDGER_FIELDS
    blob = str(item)
    assert "SECRET-REF" not in blob and "12345" not in blob


# --- F.5 reconciliation interplay --------------------------------------------------

def test_fail_keeps_performance_but_blocks_trades():
    perf = cf.build_account_performance(
        3215.0, 863.93, 0.02, 0.0, 3201.51, cf.MANUAL_SOURCE, "MANUAL")
    assert perf["performance_status"] == "Available"
    assert perf["net_pnl_after_costs_eur"] == pytest.approx(13.49)
    assert "not a reported portfolio loss" in cf.FAIL_PERFORMANCE_NOTE


def test_reconciliation_delta_is_not_loss():
    from investment_engine.reporting.regime_report import RegimeReportGenerator
    from unittest.mock import Mock
    gen = RegimeReportGenerator()
    rr = Mock()
    rr.regime = "NEUTRAL"
    rr.confidence = 0.5
    rr.generated_at = "2026-09-09T12:00:00Z"
    rr.primary_signal = "x"
    rr.implications = {"portfolio_action": "H", "tech_allocation": "t",
                       "crypto_allocation": "c", "dca_multiplier": 1.0,
                       "cash_target_pct": 10, "message": "m"}
    rr.price_structure = Mock()
    rr.price_structure.nearest_support = 1
    rr.price_structure.nearest_resistance = 2
    rr.price_structure.peaks = []
    rr.price_structure.valleys = []
    rr.timeframes = {"daily": {"indicators": {"CLOSE": 1, "RSI_14": 50}, "trend": "N"}}
    rr.news_sentiment = {"count": 0}
    rr.warnings = []
    data = {"status": "ok",
            "account_summary": {"total_equity": 3215.0, "all_positions": [
                {"symbol": "AAPL_US_EQ", "quantity": 1.0, "average_price_eur": 100.0,
                 "current_price_eur": 110.0, "value_eur": 110.0, "pnl_eur": 10.0, "pnl_pct": 10.0}],
                                "cash_free": 863.93, "cash_pie": 0.02},
            "cash": {"free": 863.93, "pie_cash": 0.02}}
    perf = cf.build_account_performance(
        3215.0, 863.93, 0.02, 0.0, 3201.51, cf.MANUAL_SOURCE, "MANUAL")
    section = gen._t212_portfolio(data, {}, {}, {}, None, None, None, None, perf)
    assert "Net P&L €+13.49" in section
    assert "delta is not a reported portfolio loss" in section


# --- F.6 reporting -----------------------------------------------------------------

def test_brief_and_snapshot_carry_performance():
    from investment_engine.reporting.documents import render_brief, render_snapshot
    perf = cf.build_account_performance(
        3214.24, 863.93, 0.02, 0.0, 3201.51, cf.MANUAL_SOURCE, "MANUAL")
    recon = {"total_equity": 3214.24, "positions_value": 12.73, "implied_cash": 3201.51,
             "reported_cash": 863.95, "derived_total": 876.68, "cash_delta": -2337.56,
             "threshold": 3.21, "status": "FAIL"}
    brief = render_brief(generated_at="2026-09-10 10:00 CEST", regime_result=None,
                         recon=recon, rows=[], monitoring_items=[],
                         earnings_7d={"in_window": [], "next_after": None},
                         decision_news=[], ideas=[], cash_line="x", performance=perf)
    assert "Net P&L after costs: €+12.73" in brief
    assert "Return on net deposits: +0.3976%" in brief
    assert "Position reconciliation is FAIL; account performance uses broker total equity" in brief
    assert "| Ticker | Company | Qty |" not in brief
    snap = render_snapshot(generated_at="2026-09-10 10:00 CEST", recon=recon, cash={},
                           rows=[], monitoring_by_display={}, earnings_status={},
                           performance=perf, ledger_metadata={"selected_source": cf.MANUAL_SOURCE,
                                                              "selected_status": "MANUAL"})
    assert "| Broker total equity |" in snap and "| Position Reconciliation" not in snap
    assert "### Broker Account Performance" in snap and "### Position Reconciliation" in snap
    assert "delta is not a reported portfolio loss" in snap
    assert "Manual verified baseline" in snap


def test_report_json_sanitization():
    import json as _json
    from investment_engine.main import _sanitized_settings, _sanitized_t212_data
    from investment_engine.config.settings import EngineSettings

    settings = EngineSettings.from_defaults()
    object.__setattr__(settings, "openrouter_api_key", "sk-or-v1-SECRETVALUE123")
    clean = _sanitized_settings(settings)
    blob = _json.dumps(clean)
    assert "SECRETVALUE123" not in blob
    assert "openrouter_api_key" not in clean
    assert clean["openrouter_model"] == settings.openrouter_model

    t212 = {"status": "ok",
            "account_summary": {"total_equity": 3215.0, "account_id": "987654321",
                                "all_positions": [], "positions": []},
            "cash": {"free": 1.0}}
    clean_t = _sanitized_t212_data(t212)
    assert "account_id" not in _json.dumps(clean_t)
    assert clean_t["account_summary"]["total_equity"] == 3215.0
