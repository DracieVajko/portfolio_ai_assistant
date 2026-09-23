"""Ticker-mapping invariants."""
import pathlib
from investment_engine.portfolio import symbols as sym
from investment_engine.portfolio.symbols import (
    DISPLAY_OVERRIDES,
    SUPPORTED,
    SUPPORTED_YAHOO,
    UNRESOLVED,
    UNRESOLVED_YAHOO,
    YAHOO_OVERRIDES,
    EXCHANGE_SUFFIX_MAP,
    propose_yahoo_candidates,
    resolve_alias,
    resolve_alias_with_source,
    support_state,
    to_display_symbol,
    to_yahoo_symbol,
)
from investment_engine.portfolio.pie_metadata import _normalize_base_symbol as pie_normalize
def test_no_supported_unresolved_overlap():
    sup_up = {str(s).strip().upper() for s in SUPPORTED_YAHOO}
    unr_up = {str(s).strip().upper() for s in UNRESOLVED_YAHOO}
    overlap = sup_up & unr_up
    assert not overlap, f"overlap: {sorted(overlap)}"
def test_no_static_mapping_to_unresolved_value():
    unr_up = {str(s).strip().upper() for s in UNRESOLVED_YAHOO}
    bad = [(k, v) for k, v in YAHOO_OVERRIDES.items() if str(v).strip().upper() in unr_up]
    assert not bad, f"static maps to UNRESOLVED: {bad}"
def test_resolve_alias_config_over_static():
    assert resolve_alias("VWSB", {}) == "VWS.CO"
    assert resolve_alias("VWSBD", {"VWSBD": "VWS.CO"}) == "VWS.CO"
    assert resolve_alias("VWSBD", {"VWSBD": "CUSTOM.X"}) == "CUSTOM.X"
    assert resolve_alias("vwsbd", {"vwsbd": "CUSTOM.X"}) == "CUSTOM.X"
    assert resolve_alias("EGT", {}) == "EGT.L"
    assert resolve_alias("EGTL", {"EGTL": "EGTL.L"}) == "EGTL.L"
    assert resolve_alias("C7A0", {"C7A0": "C7A0.F"}) == "C7A0.F"
    assert resolve_alias("UNKNOWN_KEY_XYZ", {}) is None
    assert resolve_alias("", {"A": "B"}) is None
    val, src = resolve_alias_with_source("VWSBD", {"VWSBD": "VWS.CO"})
    assert val == "VWS.CO" and src == "config"
    val2, src2 = resolve_alias_with_source("VWSB", {})
    assert val2 == "VWS.CO" and src2 == "static"
    val3, src3 = resolve_alias_with_source("NOPE_XYZ", {})
    assert val3 is None and src3 == "none"
def test_to_yahoo_uses_single_alias_layer():
    assert to_yahoo_symbol("VWSBd_EQ", "VWSB", {"VWSBD": "CUSTOM.X"}) == "CUSTOM.X"
    assert to_yahoo_symbol("VWSBd_EQ", "VWSB", {}) == "VWS.CO"
    assert to_yahoo_symbol("EGTl_EQ", "EGT", {"EGTL": "EGTL.L"}) == "EGTL.L"
    assert to_yahoo_symbol("C7A0d_EQ", "C7A0", {"C7A0": "C7A0.F"}) == "C7A0.F"
    assert to_yahoo_symbol("AAPL_US_EQ", "AAPL", {}) == "AAPL"
    assert to_yahoo_symbol("BTCUSD", "BTCUSD", {}) == "BTC-USD"
    assert to_yahoo_symbol("XEON", "XEON", {}) is None
def test_display_and_pie_wrapper_unified():
    assert to_display_symbol("VWSBd_EQ") == "VWSB"
    assert pie_normalize("VWSBd_EQ") == "VWSB"
    assert to_display_symbol("VWSBd_EQ") != "VWSBD"
    assert to_display_symbol("SUp_EQ", {"SU"}) == "SU"
    assert pie_normalize("SUp_EQ", {"SU"}) == "SU"
    assert to_display_symbol("C7A0d_EQ") == "C7A0"
    assert pie_normalize("C7A0d_EQ") == "C7A0"
    assert to_display_symbol("AIp_EQ", {"AI"}) == "AI"
    assert pie_normalize("AIp_EQ", {"AI"}) == "AI"
    assert to_display_symbol("ENRd_EQ", {"ENR"}) == "ENR"
    assert pie_normalize("ENRd_EQ", {"ENR"}) == "ENR"
    assert to_display_symbol("AAPL_US_EQ") == "AAPL"
    assert pie_normalize("AAPL_US_EQ") == "AAPL"
    from investment_engine.portfolio.pie_metadata import load_pie_universe as _loadu
    _u = _loadu("PIEs")
    assert any(pp.pie_id == "RenewablePieLong" for pp in _u.pies_for_ticker("SUp_EQ"))
    assert to_display_symbol("SUp_EQ", {"SU"}) == pie_normalize("SUp_EQ", {"SU"})
    src = pathlib.Path("investment_engine/portfolio/pie_metadata.py").read_text(encoding="utf-8")
    assert "VWSBD (kept as is" not in src
    assert "to_display_symbol" in src
    assert "def _normalize_base_symbol" in src
def test_propose_candidates_exchange_suffix_ranked():
    cands = propose_yahoo_candidates("SUp_EQ", "SU", {"currencyCode": "EUR", "isin": "FR0000121972"})
    assert isinstance(cands, list) and len(cands) >= 1
    assert all(isinstance(c, tuple) and len(c) == 2 for c in cands)
    by_cand = {c: s for c, s in cands}
    assert "SU.PA" in by_cand, f"expected SU.PA in {cands}"
    assert "exchange_suffix" in by_cand["SU.PA"] or "isin_country" in by_cand["SU.PA"] or "venue" in by_cand["SU.PA"]
    cands2 = propose_yahoo_candidates("C7A0d_EQ", "C7A0", {"currencyCode": "EUR", "isin": "CNE100006WS8"}, {"C7A0": "C7A0.F"})
    assert cands2[0][0] == "C7A0.F" and "config" in cands2[0][1]
    cands3 = propose_yahoo_candidates("VWSBd_EQ", "VWSB", {"currencyCode": "EUR", "isin": "DK0061539921"}, {"VWSBD": "VWS.CO"})
    assert cands3[0][0] == "VWS.CO" and "config" in cands3[0][1]
    assert any(c == "VWSB.DE" and "exchange_suffix:d" in s for c, s in cands3), f"missing VWSB.DE: {cands3}"
    assert EXCHANGE_SUFFIX_MAP.get("d") == ".DE"
    assert EXCHANGE_SUFFIX_MAP.get("l") == ".L"
    assert EXCHANGE_SUFFIX_MAP.get("p") == ".PA"
    assert EXCHANGE_SUFFIX_MAP.get("e") == ".MC"
    again = propose_yahoo_candidates("SUp_EQ", "SU", {"currencyCode": "EUR", "isin": "FR0000121972"})
    assert again == cands
    assert "SU.PA" not in SUPPORTED_YAHOO
def test_propose_candidates_currency_venue_labels():
    cands = propose_yahoo_candidates("ZZZl_EQ", "ZZZ", {"currencyCode": "GBX", "isin": "GB00BPVG5407"})
    by_cand = {c: s for c, s in cands}
    assert "ZZZ.L" in by_cand, f"expected ZZZ.L in {cands}"
    assert any("currency:GBX" in s for _, s in cands), f"currency source missing: {cands}"
    assert any("isin_country:GB" in s or "exchange_suffix:l" in s for _, s in cands), f"isin/exchange missing: {cands}"
    cands_fr = propose_yahoo_candidates("AIp_EQ", "AI", {"currencyCode": "EUR", "isin": "FR0000120073"})
    assert any(c == "AI.PA" for c, _ in cands_fr), f"AI.PA missing: {cands_fr}"
    cands_egt = propose_yahoo_candidates("EGTl_EQ", "EGT", {"currencyCode": "GBX", "isin": "GB00BPVG5407"})
    assert any(c in ("EGT.L", "EGTL.L") for c, _ in cands_egt)
def test_support_state_su_ai_enr_unresolved():
    for t in ("SU", "AI", "ENR"):
        assert support_state(t) == UNRESOLVED, t
        assert support_state(t, "market_data") == UNRESOLVED, t
        assert support_state(t, "earnings") == UNRESOLVED, t
        assert support_state(t, venue="US", isin="US1234567890", currency="USD") == UNRESOLVED, t
    assert support_state("C7A0") == UNRESOLVED
    assert support_state("VWSB") == UNRESOLVED
    assert support_state("C7A0.F") == SUPPORTED
    assert support_state("VWS.CO") == SUPPORTED
def test_support_state_restricted_us_pattern():
    for good in ("AAPL", "MSFT", "IBM", "JNJ", "CVX", "JPM", "WMT", "C", "NEE", "VRT"):
        assert support_state(good) == SUPPORTED, good
    assert support_state("XYZ") == UNRESOLVED
    assert support_state("FOO") == UNRESOLVED
    assert support_state("XYZ", isin="US0378331005") == SUPPORTED
    assert support_state("XYZ", currency="USD") == SUPPORTED
    assert support_state("XYZ", venue="NASDAQ") == SUPPORTED
    assert support_state("XYZ", venue="NYSE") == SUPPORTED
    assert support_state("XYZ.DE") == UNRESOLVED
    assert support_state("XYZ.DE", venue="DE", isin="DE0001234567") == UNRESOLVED
    assert support_state("XYZ", verified_symbols={"XYZ"}) == SUPPORTED
    assert support_state(None) == "UNSUPPORTED"
    assert support_state("") == "UNSUPPORTED"
    assert support_state("UNKNOWN") == "UNSUPPORTED"
    assert support_state("XEON") == "UNSUPPORTED"
def test_symbols_layer_deterministic_network_free():
    src = pathlib.Path("investment_engine/portfolio/symbols.py").read_text(encoding="utf-8")
    assert "import yfinance" not in src
    assert "import requests" not in src
    assert "def resolve_alias" in src
    assert "def propose_yahoo_candidates" in src
    assert "config" in src.lower() and "static" in src.lower()
    a = propose_yahoo_candidates("SUp_EQ", "SU", {"currencyCode": "EUR"})
    b = propose_yahoo_candidates("SUp_EQ", "SU", {"currencyCode": "EUR"})
    assert a == b
    assert to_display_symbol("VWSBd_EQ") == to_display_symbol("VWSBd_EQ")
