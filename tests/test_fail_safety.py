

# --- FAIL-state leakage + watchlist + registry + observability ----------------

from datetime import date, timedelta
from unittest.mock import Mock

import pytest

from investment_engine.portfolio.symbols import (
    resolve_company_name,
    to_display_symbol,
    to_yahoo_symbol,
)
from investment_engine.reporting.regime_report import (
    RegimeReportGenerator,
    build_canonical_signals,
    build_unified_portfolio_rows,
    compute_reconciliation,
    ensure_canonical_defaults,
)


KNOWN_CLEAN = {
    "AAPL", "NEE", "TTWO", "TUYA", "ASGN", "EGT", "SYNL", "SYN",
    "MSFT", "GD", "CF", "BMY", "NVDA", "AMD", "INTC",
}
NAMES = {"AAPL": "Apple", "NEE": "NextEra Energy", "TTWO": "Take-Two Interactive",
         "TUYA": "Tuya Inc"}
ALIASES = {"VWSBD": "VWS.CO", "EGTL": "EGTL.L", "C7A0": "C7A0.F"}


def _pos(symbol, value, qty=1.0, pnl=0.0, pnl_pct=0.0, pie=False):
    return {
        "symbol": symbol, "quantity": qty, "average_price_eur": 10.0,
        "current_price_eur": 10.0, "value_eur": value, "cost_basis_eur": value - pnl,
        "pnl_eur": pnl, "pnl_pct": pnl_pct, "is_pie_constituent": pie,
        "validation_status": "PASS", "validation_reason": "",
    }


def _t212(all_positions, positions, total=3243.51, free=0.61, pie_cash=1.03):
    return {
        "status": "ok",
        "account_summary": {
            "total_equity": total, "all_positions": all_positions, "positions": positions,
            "cash_free": free, "cash_pie": pie_cash, "cash_blocked": 23.01,
            "reconciliation_threshold": 3.24,
        },
        "cash": {"free": free, "pie_cash": pie_cash, "invested": 0, "blocked": 23.01},
        "positions": positions, "all_positions": all_positions,
    }


FAIL_RECS = {
    "portfolio_actions": [
        {"action": "BUY", "asset": "TUYA", "reason": "DCA on losing position"},
        {"action": "BUY", "asset": "TTWO", "reason": "DCA proposal"},
        {"action": "hold", "asset": "SYNL", "reason": "keep"},
    ],
    "trades": [
        {"side": "BUY", "asset": "TUYA", "qty": 5.0, "price": 1.7,
         "stop_loss": 1.5, "take_profit": 2.0, "risk_pct": 1.0, "reason": "DCA buy"},
    ],
    "position_updates": [
        {"asset": "TUYA", "stop_loss": 1.7, "new_stop": 1.59, "take_profit": 0.0,
         "new_tp": 1.85, "reason": "Adjusted stop loss and take profit for entire position after DCA"},
        {"asset": "SYNL", "stop_loss": 0.01, "new_stop": 0.009, "take_profit": 0.02,
         "new_tp": 0.025, "reason": "Routine trailing update"},
    ],
    "cash_deployment": {"free_cash": 100.0, "deploy_amount": 20.0, "deploy_pct": 20.0,
                        "reserve": 80.0, "targets": [{"asset": "TUYA", "amount": 20.0,
                                                      "price": 1.7, "max_risk": 1.0}]},
}


def _fail_report():
    from investment_engine.reporting.regime_report import RegimeReportGenerator
    from unittest.mock import Mock
    allp = [_pos("TUYA_US_EQ", 104.0, qty=61.0, pnl=-1.3, pnl_pct=-1.2),
            _pos("TTWO_US_EQ", 324.0, qty=1.6, pnl=-11.0, pnl_pct=-3.4),
            _pos("SYNl_EQ", 64.0, qty=100.0, pnl=2.9, pnl_pct=4.8)]
    data = _t212(allp, [])
    regime_result = Mock()
    regime_result.regime = "NEUTRAL"
    regime_result.confidence = 0.5
    regime_result.generated_at = "2026-09-09T12:00:00Z"
    regime_result.primary_signal = "x"
    regime_result.implications = {"portfolio_action": "H", "tech_allocation": "t",
                                  "crypto_allocation": "c", "dca_multiplier": 1.0,
                                  "cash_target_pct": 10, "message": "m"}
    regime_result.price_structure = Mock()
    regime_result.price_structure.nearest_support = 1
    regime_result.price_structure.nearest_resistance = 2
    regime_result.price_structure.peaks = []
    regime_result.price_structure.valleys = []
    regime_result.timeframes = {"daily": {"indicators": {"CLOSE": 1, "RSI_14": 50}, "trend": "N"}}
    regime_result.news_sentiment = {"count": 0}
    regime_result.warnings = []
    gen = RegimeReportGenerator()
    parts = gen.generate_parts(regime_result, data, FAIL_RECS, None, {}, NAMES, KNOWN_CLEAN, None)
    return parts


def test_fail_no_owned_buy_dca_or_position_mgmt():
    parts = _fail_report()
    actionable = "\n".join([parts.get("portfolio", ""), parts.get("ai", "")])
    assert "**BUY**" not in actionable
    assert "| BUY" not in actionable
    assert "DCA on" not in actionable and "after DCA" not in actionable
    assert "Position Management" not in parts.get("ai", "")
    assert "### Cash Deployment" not in parts.get("ai", "")
    assert "Keep Reserve" not in parts.get("ai", "")
    assert "Free Cash" not in parts.get("ai", "")
    assert "withheld because account reconciliation failed" in parts.get("ai", "")


def test_fail_watchlist_sweep():
    from investment_engine.main import _apply_fail_watchlist_sweep
    summary = ("## Summary\n- **Top Opportunities:**\n"
               "- **Top Opportunities**:\n"
               "  - **BUY** on **NVDA** (room to run).\n"
               "  - **SU** (**BUY** swing).\n")
    out, _ = _apply_fail_watchlist_sweep(summary, [])
    assert "**Top Opportunities:**" not in out and "**Top Opportunities**:" not in out
    assert "Watchlist Candidates — WATCH" in out
    assert "**BUY**" not in out and "(BUY" not in out
    assert "Research-only watchlist; no deployment is authorized while account reconciliation is failing." in out


def test_priority_tickers_use_display_forms():
    from investment_engine.main import _portfolio_tickers_for_priority
    data = _t212([_pos("EGTl_EQ", 1.0), _pos("TTWO_US_EQ", 2.0)], [])
    tickers = _portfolio_tickers_for_priority(data, KNOWN_CLEAN)
    assert "EGT" in tickers and "TTWO" in tickers
    assert not any("_EQ" in t for t in tickers)


def test_monitoring_block_review_format():
    from investment_engine.main import _build_monitoring_block
    rows = [
        {"display_symbol": "TTWO", "ticker": "TTWO", "signal": "HOLD",
         "market_value": 324.0, "weight": 10.1, "current_price": 195.0,
         "support": 208.0, "resistance": 249.0},
        {"display_symbol": "AAPL", "ticker": "AAPL", "signal": "HOLD",
         "market_value": 150.0, "weight": 4.7, "current_price": 289.0,
         "support": None, "resistance": None, "pnl_pct": 0.5},
    ]
    out = _build_monitoring_block(rows, {}, {})
    assert out.startswith("- **Portfolio Monitoring:**")
    assert "**TTWO** — REVIEW: weight >= 5%; canonical holding action remains HOLD. No transaction authorized." in out
    assert "WATCH" not in out
    assert out.rstrip().endswith("rest **HOLD**.")


def test_sweep_replaces_actionable_tickers():
    from investment_engine.main import _apply_fail_watchlist_sweep
    rows = [
        {"display_symbol": "EGT", "ticker": "EGT", "signal": "HOLD",
         "market_value": 173.0, "weight": 5.4, "current_price": 0.15,
         "support": None, "resistance": None, "pnl_pct": -10.3},
    ]
    summary = "## Summary\n- **Actionable Tickers:** **EGT** (WATCH); rest **HOLD**."
    out, _ = _apply_fail_watchlist_sweep(summary, [], rows, {}, {})
    assert "Actionable Tickers" not in out
    assert "**EGT** — REVIEW:" in out and "canonical holding action remains HOLD" in out
    assert "(WATCH)" not in out
    # No Actionable Tickers line at all: monitoring bullet is appended.
    out2, _ = _apply_fail_watchlist_sweep("## Summary\n- **Portfolio State:** HOLD all.", [], rows, {}, {})
    assert "- **Portfolio Monitoring:**" in out2 and "**EGT** — REVIEW:" in out2
    # Owned HOLD tickers must never read as WATCH/BUY in FAIL prose;
    # non-owned research names keep WATCH.
    rows2 = rows + [
        {"display_symbol": "TTWO", "ticker": "TTWO", "signal": "HOLD",
         "market_value": 324.0, "weight": 10.1, "current_price": 195.0,
         "support": 208.0, "resistance": 249.0, "pnl_pct": -3.4},
        {"display_symbol": "SYNL", "ticker": "SYNL", "signal": "HOLD",
         "market_value": 64.0, "weight": 2.0, "current_price": 0.01,
         "support": None, "resistance": None, "pnl_pct": 4.8},
    ]
    out3, _ = _apply_fail_watchlist_sweep(
        "## Summary\n- **Focus:** **TTWO** (WATCH) and **SYNL** (WATCH) are the only "
        "**explicit BUY** signals; **KASUSDT** (WATCH) stays on watch.",
        [], rows2, {}, {})
    assert "**TTWO** (HOLD)" in out3 and "**SYNL** (HOLD)" in out3
    assert "**KASUSDT** (WATCH)" in out3
    assert "BUY" not in out3
    # Bold-wrapped variant: "**TTWO (WATCH):**" normalizes the same way.
    out4, _ = _apply_fail_watchlist_sweep(
        "## Summary\n- **Watchlist Candidates — WATCH:**\n  - **TTWO (WATCH):** swing idea.",
        [], rows2, {}, {})
    assert "**TTWO** (HOLD):" in out4 and "(WATCH):" not in out4


def test_us_stock_coverage():
    from investment_engine.portfolio.symbols import support_state
    for sym in ["IBM", "MSFT", "JNJ", "CVX", "JPM", "WMT", "C"]:
        assert support_state(sym) == "SUPPORTED", sym
        assert support_state(sym, "earnings") == "SUPPORTED", sym


def test_us_symbols_pass_gate(monkeypatch):
    import sys
    import types
    from investment_engine.research import market_data
    seen = []

    class _Ticker:
        def __init__(self, sym):
            seen.append(sym)

        def history(self, *a, **k):
            import pandas as pd
            return pd.DataFrame()

    fake = types.ModuleType("yfinance")
    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    for sym in ["IBM", "MSFT", "JNJ", "CVX", "JPM", "WMT", "C"]:
        market_data.fetch_technical_indicators(sym)
    for sym in ["IBM", "MSFT", "JNJ", "CVX", "JPM", "WMT", "C"]:
        assert sym in seen, sym


def test_earnings_certainty_source_types():
    from investment_engine.research.market_data import classify_earnings_status
    assert classify_earnings_status("ASML", "2026-10-14", "official_ir") == "confirmed"
    assert classify_earnings_status("INTC", "2026-10-22", "exchange_notice") == "confirmed"
    for src in (None, "", "yfinance", "aggregator", "calendar", "OTHER"):
        assert classify_earnings_status("ASML", "2026-10-14", src) == "estimated", src
        assert classify_earnings_status("INTC", "2020-01-01", src) == "estimated", src


def test_support_registry_states():
    from investment_engine.portfolio.symbols import support_state
    for good in ["VRT", "QCOM", "AAPL", "NVDA", "INTC", "AVGO", "NEE", "C7A0.F", "VWS.CO", "EGT.L", "IBE.MC", "RWE.DE", "LIT", "SUI-USD", "TAO-USD", "RENDER-USD", "KAS-USD"]:
        assert support_state(good) == "SUPPORTED", good
    for bad in ["HY9H", "C7A0", "VWSB", "SYNL", "IQQH"]:
        assert support_state(bad) == "UNRESOLVED", bad
    for off in ["XEON", None, "UNKNOWN", ""]:
        assert support_state(off) in ("UNSUPPORTED", "UNRESOLVED"), off
    assert support_state("SUI-USD", "earnings") == "UNSUPPORTED"
    assert support_state("TAO-USD", "earnings") == "UNSUPPORTED"
    assert support_state(None) == "UNSUPPORTED"


def test_unresolved_symbols_zero_yfinance_requests(monkeypatch):
    import sys
    import types
    from investment_engine.research import market_data
    from investment_engine.portfolio.symbols import to_display_symbol as _td, to_yahoo_symbol as _ty

    def _boom(sym, *a, **k):
        raise AssertionError(f"yfinance called with {sym!r}")

    fake = types.ModuleType("yfinance")
    fake.Ticker = _boom
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    raws = ["HY9Hd_EQ", "EGTl_EQ", "LITMm_EQ", "IQQHd_EQ", "C7A0d_EQ", "VWSBd_EQ",
            "IBEe_EQ", "RWEd_EQ", "SYNl_EQ", "SUIUSDT", "TAOUSDT"]
    for raw in raws:
        disp = _td(raw, KNOWN_CLEAN)
        yahoo = _ty(raw, disp, ALIASES)
        assert market_data.recent_earnings_date(yahoo) is None
        assert market_data.fetch_technical_indicators(yahoo) is None
        assert market_data.analyst_consensus(yahoo) is None


def test_unresolved_rows_marked_broker_valuation():
    # Use symbols that are still UNRESOLVED
    allp = [_pos("HY9Hd_EQ", 173.0, qty=1151.0, pnl=-19.0, pnl_pct=-10.3),
            _pos("SYNl_EQ", 64.0, qty=100.0, pnl=2.9, pnl_pct=4.8)]
    data = _t212(allp, [])
    rows = build_unified_portfolio_rows(data, {}, {}, NAMES, KNOWN_CLEAN, None,
                                        {"HY9HD": "HY9HD", "SYNL": "SYN.L"})
    by_disp = {r["display_symbol"]: r for r in rows}
    assert by_disp["HY9HD"]["market_data_state"] == "UNRESOLVED"
    assert "Market-data mapping unavailable; broker valuation retained." in by_disp["HY9HD"]["notes"]
    assert "possibly delisted" not in by_disp["HY9HD"]["notes"].lower()


def test_reconciliation_diagnostic_shape():
    allp = [_pos("A_POS", 3000.00), _pos("B_POS", 379.14)]
    data = _t212(allp, [])
    rows = build_unified_portfolio_rows(data, {}, {}, {}, KNOWN_CLEAN)
    recon = compute_reconciliation(data, rows)
    assert set(["total_equity", "positions_value", "implied_cash", "reported_cash",
                "cash_delta", "threshold", "status"]) <= set(recon)
    assert all(isinstance(recon[k], float) for k in
               ("total_equity", "positions_value", "implied_cash", "reported_cash", "cash_delta", "threshold"))
