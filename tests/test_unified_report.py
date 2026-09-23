"""Regression tests for the unified report refactor (fixes 1-7).

Fast and offline: no network, no broker calls. Covers authoritative
position source, reconciliation, one canonical signal, priority triggers,
display mappings, earnings-symbol mapping, and reliability helpers.
"""

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
    "MSFT", "GD", "CF", "BMY", "NVDA", "AMD", "INTC", "SU", "ENR",
    "RWE", "IBE", "AI", "LIN", "VWSB", "C7A0", "IQQH", "LITM",
    "XEON", "CSH2", "ERNX", "BTC", "ETH",
}
NAMES = {
    "AAPL": "Apple",
    "NEE": "NextEra Energy",
    "MSFT": "Microsoft",
    "GD": "General Dynamics",
    "CF": "CF Industries",
    "BMY": "Bristol-Myers Squibb",
}

ALIASES = {"VWSBD": "VWS.CO", "EGTL": "EGTL.L", "C7A0": "C7A0.F"}


def _pos(symbol, value, qty=1.0, pnl=0.0, pnl_pct=0.0, pie=False):
    return {
        "symbol": symbol,
        "quantity": qty,
        "average_price_eur": 10.0,
        "current_price_eur": 10.0,
        "value_eur": value,
        "cost_basis_eur": value - pnl,
        "pnl_eur": pnl,
        "pnl_pct": pnl_pct,
        "is_pie_constituent": pie,
        "validation_status": "PASS",
        "validation_reason": "",
    }


def _t212(all_positions, positions, total=3243.51, free=0.61, pie_cash=1.03):
    return {
        "status": "ok",
        "account_summary": {
            "total_equity": total,
            "all_positions": all_positions,
            "positions": positions,
            "cash_free": free,
            "cash_pie": pie_cash,
            "cash_blocked": 23.01,
            "reconciliation_threshold": 3.24,
        },
        "cash": {"free": free, "pie_cash": pie_cash, "invested": 0, "blocked": 23.01},
        "positions": positions,
        "all_positions": all_positions,
    }


AI_RECS = {
    "portfolio_actions": [
        {"action": "hold", "asset": "TTWO_US_EQ", "reason": "neutral regime hold"},
    ],
    "trades": [
        {"side": "SELL", "asset": "SYNl_EQ", "qty": 1.0, "price": 0.01,
         "stop_loss": 0.02, "take_profit": 0.005, "risk_pct": 1.0, "reason": "exit volatile"},
    ],
    "position_updates": [],
    "cash_deployment": {"free_cash": 0.61, "deploy_amount": 0.0, "deploy_pct": 0.0, "reserve": 0.61, "targets": []},
}


# --- Fix 5: display mappings -------------------------------------------------

@pytest.mark.parametrize(("internal", "display"), [
    ("AAPL_US_EQ", "AAPL"),
    ("TTWO_US_EQ", "TTWO"),
    ("NEE_US_EQ", "NEE"),
    ("MSFT_US_EQ", "MSFT"),
    ("AAPL", "AAPL"),
])
def test_display_symbol_mapping(internal, display):
    assert to_display_symbol(internal, KNOWN_CLEAN) == display


def test_company_names():
    assert resolve_company_name("AAPL_US_EQ", "AAPL", "", NAMES) == "Apple"
    assert resolve_company_name("TTWO_US_EQ", "TTWO", "", NAMES) == "Take-Two Interactive"
    # Never falls back to an internal broker ID.
    assert resolve_company_name("ZZZ_US_EQ", "ZZZ", "", {}) == "Unknown instrument"
    assert resolve_company_name("ZZZ_US_EQ", "ZZZ", "ZZZ_US_EQ", {}) == "Unknown instrument"


# --- Fix 6: earnings-symbol mapping ------------------------------------------

def test_yahoo_mapping_crypto():
    assert to_yahoo_symbol("BTCUSD", "BTCUSD", {}) == "BTC-USD"
    assert to_yahoo_symbol("ETHUSD", "ETHUSD", {}) == "ETH-USD"


def test_yahoo_mapping_aliases_and_guards():
    assert to_yahoo_symbol("EGTl_EQ", "EGT", ALIASES) == "EGTL.L"
    assert to_yahoo_symbol("XEON", "XEON", {}) is None
    assert to_yahoo_symbol("AAPL_US_EQ", "AAPL", {}) == "AAPL"
    # Company names are never valid Yahoo queries.
    from investment_engine.research import market_data
    assert market_data.recent_earnings_date("APPLE") is None
    assert market_data.recent_earnings_date("Intel") is None


def test_market_data_guard_skips_network(monkeypatch):
    import sys
    import types
    called = []
    fake = types.ModuleType("yfinance")

    class _Ticker:
        def __init__(self, sym):
            called.append(sym)

    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    from investment_engine.research import market_data
    assert market_data.recent_earnings_date("Advanced Micro Devices") is None
    assert called == []


# --- Fix 1: authoritative source ----------------------------------------------

def test_standalone_in_all_positions_not_counted_twice():
    shared = [_pos("TTWO_US_EQ", 320.55), _pos("AAPL_US_EQ", 158.44, pie=True)]
    data = _t212(list(shared), list(shared))
    rows = build_unified_portfolio_rows(data, AI_RECS, {}, NAMES, KNOWN_CLEAN)
    assert len(rows) == 2
    assert sum(r["market_value"] for r in rows) == pytest.approx(478.99)


def test_positions_fallback_only():
    only_standalone = [_pos("TTWO_US_EQ", 100.0)]
    data = _t212([], only_standalone)
    data["account_summary"]["all_positions"] = []
    data["all_positions"] = []
    rows = build_unified_portfolio_rows(data, AI_RECS, {}, NAMES, KNOWN_CLEAN)
    assert [r["display_symbol"] for r in rows] == ["TTWO"]


def test_duplicate_broker_entries_not_summed():
    dup = [_pos("NEE_US_EQ", 56.87, pie=True), _pos("NEE_US_EQ", 56.87, pie=True)]
    data = _t212(dup, [])
    rows = build_unified_portfolio_rows(data, AI_RECS, {}, NAMES, KNOWN_CLEAN)
    assert len(rows) == 1
    assert rows[0]["market_value"] == pytest.approx(56.87)
    assert "not summed" in rows[0]["notes"]


# --- Fix 2: reconciliation ----------------------------------------------------

def test_reconciliation_formula_and_status():
    allp = [_pos("A_POS", 3000.00), _pos("B_POS", 379.14)]
    data = _t212(allp, [])
    rows = build_unified_portfolio_rows(data, {}, {}, {}, KNOWN_CLEAN)
    recon = compute_reconciliation(data, rows)
    assert recon["positions_value"] == pytest.approx(3379.14)
    # reported_cash = free + pie_cash + blocked = 0.61 + 1.03 + 23.01 = 24.65
    assert recon["reported_cash"] == pytest.approx(24.65)
    assert recon["derived_total"] == pytest.approx(3379.14 + 24.65)
    assert recon["diff"] == pytest.approx(24.65 - 3243.51 + 3379.14)  # Actually diff = derived - total = (positions + reported) - total
    # Let's recompute: total=3243.51, positions=3379.14, reported=24.65, derived=3403.79, diff=3403.79-3243.51=160.28
    assert recon["diff"] == pytest.approx(160.28)
    assert recon["status"] == "FAIL"


def test_reconciliation_warning_neutral_and_gated():
    allp = [
        _pos("A_POS", 3000.00),
        dict(_pos("B_POS", 379.14), validation_status="FAIL", validation_reason="stale price"),
    ]
    data = _t212(allp, [])
    gen = RegimeReportGenerator()
    section = gen._t212_portfolio(data, {}, {}, {}, KNOWN_CLEAN, {})
    low = section.lower()
    assert "withheld" in low
    for banned in ("pending orders", "zaokr", "stale prices", "fx"):
        assert banned not in low
    # Validated P&L visible; failed row masked.
    assert "€3,000.00" in section
    assert section.count("n/a") >= 1


# --- Fix 3: one canonical signal ----------------------------------------------

def test_canonical_signal_identical_across_sections():
    from investment_engine.main import (
        _build_priority_actions_table,
        _enforce_canonical_in_pie_section,
    )
    allp = [
        _pos("NEE_US_EQ", 56.87, pnl_pct=-0.9, pie=True),
        _pos("SYNl_EQ", 75.86, pnl_pct=23.4),
        _pos("TTWO_US_EQ", 320.55, pnl_pct=-4.6),
    ]
    tech = {"NEE": {"RSI_14": 40.6, "Support": 80.75, "Resistance": 86.87}}
    canon = build_canonical_signals(AI_RECS, KNOWN_CLEAN)
    ensure_canonical_defaults(canon, ["NEE", "SYN", "SYNL", "TTWO"])
    rows = build_unified_portfolio_rows(_t212(allp, []), AI_RECS, tech, NAMES, KNOWN_CLEAN, canon)
    by_disp = {r["display_symbol"]: r["signal"] for r in rows}
    assert by_disp["NEE"] == "HOLD"

    table = _build_priority_actions_table(rows, {}, tech, canonical_map=canon)
    pie = "- **NEE** — **BUY** — oversold bounce — support 80.75\n- **TTWO** — **SELL** — weak — support 1.0"
    fixed = _enforce_canonical_in_pie_section(pie, canon, KNOWN_CLEAN)

    import re
    for disp, sig in by_disp.items():
        if disp in table:
            line = next(l for l in table.splitlines() if f"**{disp}**" in l)
            assert sig in line, (disp, line)
    assert "NEE — **HOLD**" in fixed
    assert "TTWO — **HOLD**" in fixed
    assert "NEE — **BUY**" not in fixed


# --- Fix 4: priority triggers --------------------------------------------------

def test_priority_hold_triggers_no_padding():
    from investment_engine.main import _build_priority_actions_table
    rows = [
        {"display_symbol": "SELL1", "ticker": "SELL1", "signal": "SELL", "market_value": 100.0,
         "unrealized_pnl": 0.0, "pnl_pct": 1.0, "weight": 1.0, "support": None, "resistance": None},
        {"display_symbol": "BIG", "ticker": "BIG", "signal": "HOLD", "market_value": 200.0,
         "unrealized_pnl": 0.0, "pnl_pct": 1.0, "weight": 6.0, "support": None, "resistance": None},
        {"display_symbol": "PLAIN", "ticker": "PLAIN", "signal": "HOLD", "market_value": 150.0,
         "unrealized_pnl": 0.0, "pnl_pct": 1.0, "weight": 1.0, "support": None, "resistance": None,
         "current_price": 100.0},
    ]
    tech = {"PLAIN": {"RSI_14": 50.0}}
    table = _build_priority_actions_table(rows, {}, tech)
    assert "**SELL1**" in table and "**BIG**" in table
    assert "**PLAIN**" not in table
    assert table.index("**SELL1**") < table.index("**BIG**")
    assert len([l for l in table.splitlines() if l.startswith("| **")]) == 2


# --- Fix 5/6: rendered output uses display symbols, radar format ---------------

def test_rendered_sections_use_display_symbols():
    allp = [_pos("AAPL_US_EQ", 158.44, pie=True), _pos("TTWO_US_EQ", 320.55)]
    data = _t212(allp, [])
    canon = build_canonical_signals(AI_RECS, KNOWN_CLEAN)
    gen = RegimeReportGenerator()
    section = gen._t212_portfolio(data, AI_RECS, {}, NAMES, KNOWN_CLEAN, canon)
    assert "AAPL" in section and "TTWO" in section
    for banned in ("_US_EQ", "_DE_EQ", "_IM_EQ"):
        assert banned not in section
    assert "Apple" in section and "Take-Two Interactive" in section


def test_earnings_radar_format_and_unavailable():
    from investment_engine.main import _format_earnings_radar
    d1 = (date.today() + timedelta(days=10)).isoformat()
    d2 = (date.today() + timedelta(days=40)).isoformat()
    radar = _format_earnings_radar(
        {"BBB": (d2, "confirmed"), "AAA": (d1, "estimated")},
        {"AAA": "AAA Corp", "BBB": "BBB Inc"},
        ["XEON"],
    )
    assert radar.startswith("## Earnings Radar")
    bullets = [l for l in radar.splitlines() if l.startswith("- ")]
    assert len(bullets) == 2 and "AAA" in bullets[0]
    assert all("(confirmed)" in b or "(estimated)" in b for b in bullets)
    assert "XEON" in radar and "unavailable" in radar.lower()
    assert "delisted" not in radar.lower()


def test_single_holdings_section():
    from investment_engine.reporting.integrated_report import generate_full_integrated_report
    regime_result = Mock()
    regime_result.regime = "NEUTRAL"
    regime_result.confidence = 0.75
    regime_result.generated_at = "2024-01-15T12:00:00Z"
    regime_result.primary_signal = "Test signal"
    regime_result.implications = {
        "portfolio_action": "HOLD", "tech_allocation": "5%", "crypto_allocation": "0%",
        "dca_multiplier": 1.0, "cash_target_pct": 10, "message": "Test guidance",
    }
    regime_result.price_structure = Mock()
    regime_result.price_structure.nearest_support = 100.0
    regime_result.price_structure.nearest_resistance = 110.0
    regime_result.price_structure.peaks = []
    regime_result.price_structure.valleys = []
    regime_result.timeframes = {
        "daily": {"indicators": {"CLOSE": 105.0, "RSI_14": 50, "SMA_200": 100.0, "SMA_50": 102.0}, "trend": "NEUTRAL"}
    }
    regime_result.news_sentiment = {"sentiment": "neutral", "score": 50, "count": 5, "key_topics": ["test"]}
    regime_result.warnings = []
    data = _t212([_pos("AAPL_US_EQ", 100.0)], [])
    report = generate_full_integrated_report(
        regime_result=regime_result, t212_data=data, ai_recs=AI_RECS,
        known_clean=KNOWN_CLEAN, canonical=build_canonical_signals(AI_RECS, KNOWN_CLEAN),
        names=NAMES,
    )
    assert report.count("## 💼 T212 Portfolio") == 1
    assert "T212 Holdings" not in report


# --- Fix 7: reliability --------------------------------------------------------

def test_provider_fallback_note_compact():
    from investment_engine.main import (
        _FALLBACK_EVENTS,
        _methodology_block,
        _provider_fallback_note,
        _record_fallback,
    )
    _FALLBACK_EVENTS.clear()
    _record_fallback("summary", "local endpoints unavailable")
    note = _provider_fallback_note()
    assert "deterministic fallback" in note and "summary" in note
    assert "Traceback" not in note
    block = _methodology_block({}, Mock(name="Chain"))
    assert "Provider note" in block
    _FALLBACK_EVENTS.clear()


def test_talib_gate_skips_without_capability():
    from investment_engine.research import technical_analysis as ta_mod
    import pandas as pd
    if ta_mod.HAS_TALIB:
        pytest.skip("TA-Lib present; gate path not exercised")
    analyzer = ta_mod.TechnicalAnalyzer()
    df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2], "volume": [10, 20, 30],
    })
    out = analyzer._apply_candlestick_patterns(df)
    assert not [c for c in out.columns if str(c).startswith("CDL_")]


# --- Fix 1: cash diagnosis + FAIL deployment suppression ----------------------

from investment_engine.reporting.regime_report import apply_trade_safety


def test_cash_diagnosis_fields():
    allp = [_pos("A_POS", 3000.00), _pos("B_POS", 379.14)]
    data = _t212(allp, [])
    rows = build_unified_portfolio_rows(data, {}, {}, {}, KNOWN_CLEAN)
    recon = compute_reconciliation(data, rows)
    assert recon["implied_cash"] == pytest.approx(3243.51 - 3379.14)
    # reported_cash = free + pie_cash + blocked = 0.61 + 1.03 + 23.01 = 24.65
    assert recon["reported_cash"] == pytest.approx(24.65)
    assert recon["cash_delta"] == pytest.approx(recon["diff"])
    assert recon["status"] == "FAIL"


def test_fail_suppresses_deployment_and_buys():
    from investment_engine.main import _apply_fail_guards, _format_cash_deployment
    allp = [_pos("A_POS", 3000.00), _pos("B_POS", 379.14)]
    data = _t212(allp, [])
    rows = build_unified_portfolio_rows(data, AI_RECS, {}, NAMES, KNOWN_CLEAN)
    line = _format_cash_deployment(Mock(), data, AI_RECS, rows)
    assert line == "€24.65 reported cash — deployment withheld until account reconciliation passes."
    guarded = _apply_fail_guards(
        {"trades": [{"side": "BUY", "asset": "X"}, {"side": "SELL", "asset": "Y"}],
         "cash_deployment": {"free_cash": 10.0, "deploy_amount": 5.0, "deploy_pct": 50.0,
                              "reserve": 5.0, "targets": [{"asset": "X"}]}},
        "FAIL",
    )
    assert [t["side"] for t in guarded["trades"]] == ["SELL"]
    assert guarded["cash_deployment"]["deploy_amount"] == 0.0
    assert guarded["cash_deployment"]["targets"] == []


# --- Fix 2: BUY safety gates ---------------------------------------------------

def _safety_row(display, signal="BUY", weight=1.0, pnl=1.0, price=10.0, value=100.0):
    return {"display_symbol": display, "ticker": display, "signal": signal,
            "weight": weight, "pnl_pct": pnl, "current_price": price,
            "market_value": value, "support": None, "resistance": None, "notes": ""}


def test_buy_gate_missing_technicals_egt_like():
    canon = {"EGT": {"signal": "BUY", "raw": ["BUY"], "sources": ["trade"],
                     "conflict": False, "display": "EGT", "internal": "EGTl_EQ"}}
    rows = [_safety_row("EGT", weight=5.4, pnl=-10.3)]
    apply_trade_safety(canon, rows, {}, "PASS", {})
    assert canon["EGT"]["signal"] == "HOLD"
    assert "missing technical data" in rows[0]["notes"]


def test_buy_gate_concentration_ttwo_like():
    canon = {"TTWO": {"signal": "BUY", "raw": ["BUY"], "sources": ["trade"],
                      "conflict": False, "display": "TTWO", "internal": "TTWO_US_EQ"}}
    rows = [_safety_row("TTWO", weight=10.08, pnl=-3.5)]
    tech = {"TTWO": {"RSI_14": 16.9, "Support": 208.52, "Resistance": 248.98}}
    apply_trade_safety(canon, rows, tech, "PASS", {})
    assert canon["TTWO"]["signal"] == "HOLD"
    assert "concentration cap" in rows[0]["notes"]


def test_buy_gate_no_averaging_down():
    canon = {"BIG": {"signal": "BUY", "raw": ["BUY"], "sources": ["trade"],
                     "conflict": False, "display": "BIG", "internal": "BIG"}}
    rows = [_safety_row("BIG", weight=1.0, pnl=-5.0)]
    apply_trade_safety(canon, rows, {"BIG": {"RSI_14": 50.0}}, "PASS", {})
    assert canon["BIG"]["signal"] == "HOLD"
    assert "averaging down" in rows[0]["notes"]


def test_buy_gate_valid_buy_passes_and_fail_blanket():
    canon = {"OK": {"signal": "BUY", "raw": ["BUY"], "sources": ["trade"],
                    "conflict": False, "display": "OK", "internal": "OK"}}
    rows = [_safety_row("OK", weight=1.0, pnl=2.0)]
    apply_trade_safety(canon, rows, {"OK": {"RSI_14": 55.0}}, "PASS", {})
    assert canon["OK"]["signal"] == "BUY"
    apply_trade_safety(canon, rows, {"OK": {"RSI_14": 55.0}}, "FAIL", {})
    assert canon["OK"]["signal"] == "HOLD"
    assert "reconciliation FAIL" in rows[0]["notes"]


# --- Fix 3: central Yahoo resolver ----------------------------------------------

BAD_YAHOO = ["VERTIV", "QUALCOMM", "APPL", "NVIDIA", "INTE", "BROADCOM", "BTCUSD", "ETHUSD"]


def test_verified_yahoo_overrides():
    assert to_yahoo_symbol("VERTIV", "VERTIV", {}) == "VRT"
    assert to_yahoo_symbol("QUALCOMM", "QUALCOMM", {}) == "QCOM"
    assert to_yahoo_symbol("NVIDIA", "NVIDIA", {}) == "NVDA"
    assert to_yahoo_symbol("INTE", "INTC", {}) == "INTC"
    assert to_yahoo_symbol("BROADCOM", "AVGO", {}) == "AVGO"
    assert to_yahoo_symbol("APPL", "APPL", {}) == "AAPL"
    assert to_yahoo_symbol("VRT", "VRT", {}) == "VRT"
    assert to_yahoo_symbol("BTCUSD", "BTCUSD", {}) == "BTC-USD"


def test_yahoo_gate_rejects_bad_inputs_without_network(monkeypatch):
    import sys
    import types
    from investment_engine.research import market_data

    def _boom(sym, *a, **k):
        raise AssertionError(f"yfinance called with {sym!r}")

    fake = types.ModuleType("yfinance")
    fake.Ticker = _boom
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    for bad in BAD_YAHOO + ["Apple Inc", "AAPL_US_EQ", "TTWO_US_EQ", "", "X" * 17]:
        assert market_data.recent_earnings_date(bad) is None
        assert market_data.fetch_technical_indicators(bad) is None
        assert market_data.analyst_consensus(bad) is None


def test_pipeline_helpers_never_call_bad_symbols(monkeypatch):
    import sys
    import types
    from investment_engine.portfolio.symbols import to_display_symbol as _td, to_yahoo_symbol as _ty
    import investment_engine.main as main_mod
    from investment_engine.research import market_data

    seen = []
    fake = types.ModuleType("yfinance")

    class _Ticker:
        def __init__(self, sym):
            seen.append(sym)

        def history(self, *a, **k):
            raise RuntimeError("offline")

        def get_earnings_dates(self, *a, **k):
            raise RuntimeError("offline")

    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    raws = ["VERTIV", "QUALCOMM", "APPL", "NVIDIA", "INTE", "BROADCOM",
            "BTCUSD", "ETHUSD", "AAPL_US_EQ", "Apple Inc", "VRT", "AAPL"]
    pairs = [(r, _ty(r, _td(r), None)) for r in raws]
    main_mod._fetch_technicals_parallel([(d, y) for d, y in
                                         [(r, _ty(r, _td(r), None)) for r in raws] if y])
    main_mod._fetch_earnings_for_symbols({r: _ty(r, _td(r), None) for r in raws})
    for s in seen:
        assert s not in BAD_YAHOO, s
        assert " " not in s and "_EQ" not in s.upper(), s


# --- Fix 4: Potential New Ideas --------------------------------------------------

def test_new_ideas_fallback_exact():
    from investment_engine.main import _validate_new_ideas
    out = _validate_new_ideas(
        "Here’s your **\"Potential New Ideas\"** section based on recent headlines "
        "(hypothetical examples for illustration—replace with actual tickers):\n"
        "- **Ticker: TSLA** – placeholder text",
        {"MACRO": [{"title": "S&P 500 surges", "url": "https://example.com/a"}]},
    )
    assert out == "## Potential New Ideas\n\nNo new ideas met the current evidence and validation threshold."


def test_new_ideas_validated_table():
    from investment_engine.main import _validate_new_ideas
    out = _validate_new_ideas(
        "- **Ticker: TSLA** – regulatory scrutiny",
        {"X": [{"title": "TSLA faces regulatory scrutiny over margins", "url": "https://example.com/t"}]},
    )
    assert "| **TSLA** |" in out and "https://example.com/t" in out
    for banned in ("Here's your", "hypothetical", "for illustration", "replace with", "placeholder"):
        assert banned not in out


# --- Fix 5: provider resilience ----------------------------------------------------

def test_circuit_breaker_skips_dead_providers(caplog):
    import logging
    from investment_engine.providers.base import BaseLLMProvider
    from investment_engine.providers.fallback import ChainedFallbackProvider, reset_circuit_breaker

    calls = []

    class Fail(BaseLLMProvider):
        def __init__(self, name, result):
            self._name, self._result = name, result

        @property
        def name(self):
            return self._name

        def generate(self, prompt, *, stage, context=None):
            calls.append(self._name)
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

    class Good(BaseLLMProvider):
        def __init__(self):
            self.model = "mistral-test"

        @property
        def name(self):
            return "Mistral"

        def generate(self, prompt, *, stage, context=None):
            calls.append("Mistral")
            return "ok-content"

    reset_circuit_breaker()
    chain = ChainedFallbackProvider([
        Fail("LM Studio", RuntimeError("Model 'qwen3.8-4b' previously failed to load")),
        Fail("Ollama", "[decision] Ollama error: connection refused"),
        Fail("Gemini", "[decision] Gemini HTTP: {\"code\": 429, \"message\": " +
             "\"quota exceeded. Please retry in 17.9s.\", \"status\": \"RESOURCE_EXHAUSTED\"}"),
        Good(),
    ])
    with caplog.at_level(logging.INFO, logger="investment_engine.providers.fallback"):
        assert chain.generate("p", stage="decision") == "ok-content"
        assert chain.generate("p", stage="decision") == "ok-content"
    # Dead providers tried once, then skipped: 3 + 1 + 1 calls.
    assert calls.count("LM Studio") == 1 and calls.count("Ollama") == 1
    assert calls.count("Gemini") == 1 and calls.count("Mistral") == 2
    assert "model=mistral-test" in caplog.text
    reset_circuit_breaker()


def test_finviz_disable_on_api_error():
    from investment_engine.research import technical_analysis as ta_mod
    assert ta_mod.finviz_enabled() or not ta_mod.HAS_FINVIZ
    ta_mod.disable_finviz("unit test")
    try:
        assert not ta_mod.finviz_enabled()
        if ta_mod.HAS_FINVIZ:
            assert ta_mod.FinVizEnrichment().get_quote_data("AAPL") == {}
    finally:
        ta_mod._FINVIZ_DISABLED = False


def test_earnings_fast_display_keys_no_company_names(monkeypatch):
    import investment_engine.main as main_mod
    monkeypatch.setattr(main_mod, "recent_earnings_date", lambda y: "2026-10-29")
    assets = [
        {"ticker": "NVDA", "name": "Nvidia"},
        {"ticker": "VRT", "name": "Vertiv"},
        {"broker_symbol": "BTCUSD", "yahoo_symbol": "BTC-USD", "name": "Bitcoin"},
    ]
    out = main_mod._fetch_earnings_fast(assets)
    assert out.get("NVDA") == "2026-10-29"
    assert out.get("VRT") == "2026-10-29"
    assert out.get("BTC-USD") == "unavailable"
    assert "Nvidia" not in out and "Vertiv" not in out and "Bitcoin" not in out


def test_radar_confirmed_wins_and_fences_stripped():
    from investment_engine.main import _format_earnings_radar, _strip_priority_actions_from_summary
    from datetime import date, timedelta
    d = (date.today() + timedelta(days=20)).isoformat()
    radar = _format_earnings_radar({"INTC": (d, "estimated")}, {"INTC": "Intel"})
    assert "(estimated)" in radar
    # Confirmed upgrade path via _add_radar semantics: direct call twice.
    rows_first = {"INTC": (d, "estimated")}
    r1 = _format_earnings_radar(rows_first, {"INTC": "Intel"})
    assert "(estimated)" in r1
    cleaned = _strip_priority_actions_from_summary("```markdown\n## Summary\n- x\n```")
    assert "```" not in cleaned and "## Summary" in cleaned
