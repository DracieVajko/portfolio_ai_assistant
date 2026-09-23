"""Pie-aware exposure tests – advisory only, no live trading."""
from pathlib import Path
import json

from investment_engine.portfolio.pie_metadata import load_pie_universe
from investment_engine.portfolio.sidecar import load_sidecar_registry
from investment_engine.portfolio.exposure import (
    build_canonical_exposures,
    evaluate_opportunistic_standalone_add,
    generate_actionability_view,
    generate_exposure_view,
)
from trading212_portfolio import parse_position


def _fake_api_position(ticker: str, quantity: float, pie_quantity: float, current_price: float, ppl: float = 0.0, fx_ppl: float = 0.0):
    """Helper to build parse_position-like raw dict."""
    return {
        "ticker": ticker,
        "quantity": quantity,
        "averagePrice": current_price,
        "currentPrice": current_price,
        "ppl": ppl,
        "fxPpl": fx_ppl,
        "pieQuantity": pie_quantity,
        "initialFillDate": "2024-01-01T00:00:00Z",
    }


def test_csv_zero_placeholders_never_affect_account_calculations():
    """PIEs/*.csv zero placeholders must not be used for account P/L/value/quantity/reconciliation."""
    # DailyDivPie has all zeros for Invested value, Value, Owned quantity
    universe = load_pie_universe("PIEs")
    # Ensure TechPieShort and DailyDiv are loaded
    tech = next((p for p in universe.pies if p.pie_id == "TechPieShort"), None)
    daily = next((p for p in universe.pies if p.pie_id == "DailyDivPieLongShort"), None)
    assert tech is not None and daily is not None
    # DailyDiv constituents all parsing to tickers, but no value used
    assert len(daily.constituents) > 0
    # Ensure zero placeholders were ignored: constituents still have tickers, but no quantity/value
    for c in daily.constituents:
        assert c.ticker != "0"
        assert c.target_weight is None  # unknown, not invented

    # Now create API position for a symbol that is in DailyDiv but CSV has 0 quantity
    sidecar = load_sidecar_registry("PIEs/config")
    raw = _fake_api_position("CF_US_EQ", quantity=10, pie_quantity=10, current_price=50, ppl=20, fx_ppl=5)
    parsed = parse_position(raw)
    # Build exposures: API value should be used, not CSV zero
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    cf_exp = next(e for e in exps if e.symbol == "CF_US_EQ")
    # Total quantity/value must come from API, not CSV zero
    assert cf_exp.total_quantity == 10
    assert cf_exp.total_value_eur == parsed["value_eur"]
    # Total value must be >0, not zero placeholder
    assert cf_exp.total_value_eur > 0
    # CSV zero placeholder did not overwrite
    assert cf_exp.data_validation_status == parsed["validation_status"]


def test_pie_and_standalone_quantities_aggregate_correctly():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    # NVDA example from spec: 0.60 inside pie, 0.15 standalone => total 0.75
    # Simulate API: quantity 0.75, pie_quantity 0.60
    raw = _fake_api_position("NVDA_US_EQ", quantity=0.75, pie_quantity=0.60, current_price=180, ppl=5, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=20000)
    nvda = next(e for e in exps if e.symbol == "NVDA_US_EQ")
    pie_qty = sum(p.quantity for p in nvda.pie_exposures)
    standalone_qty = nvda.standalone_exposure.quantity
    assert abs(pie_qty - 0.60) < 1e-8
    assert abs(standalone_qty - 0.15) < 1e-8
    assert abs(nvda.total_quantity - 0.75) < 1e-8


def test_pie_and_standalone_value_aggregate_correctly():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    raw = _fake_api_position("NVDA_US_EQ", quantity=0.75, pie_quantity=0.60, current_price=200, ppl=10, fx_ppl=2)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=20000)
    nvda = next(e for e in exps if e.symbol == "NVDA_US_EQ")
    pie_val = sum(p.value_eur for p in nvda.pie_exposures)
    standalone_val = nvda.standalone_exposure.value_eur
    total_val = nvda.total_value_eur
    # Due to equal split across pies containing NVDA (TechPieShort only), pie_val = 0.60*price_eur
    # standalone = 0.15*price_eur, total = 0.75*price_eur
    assert abs(pie_val + standalone_val - total_val) < 1e-6
    assert abs(total_val - parsed["value_eur"]) < 1e-6


def test_standalone_buy_blocked_when_total_cap_exceeded():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    # Use TechPieShort cap 7% of account, standalone max 3%
    tech_sidecar = sidecar.get_for_pie("TechPieShort")
    assert tech_sidecar.max_total_ticker_pct_of_account == 7.0
    # Create position where total already 6.5% of 10k = 650 EUR, pie 600, standalone 50
    raw = _fake_api_position("NVDA_US_EQ", quantity=0.75, pie_quantity=0.60, current_price=180, ppl=0, fx_ppl=1)
    parsed = parse_position(raw)
    # Adjust price to get desired value: 0.75*180=135 EUR -> 1.35% of 10k, not enough to test cap
    # Create larger position: quantity 3, price 300 => 900 EUR = 9% > cap
    raw2 = _fake_api_position("AAPL_US_EQ", quantity=3, pie_quantity=3, current_price=300, ppl=0, fx_ppl=1)
    parsed2 = parse_position(raw2)
    total_equity = 10000
    exps = build_canonical_exposures([parsed2], universe, sidecar, total_equity_eur=total_equity)
    aapl = next(e for e in exps if e.symbol == "AAPL_US_EQ")
    # Already 9% > 7% cap, so concentration warning
    assert aapl.concentration_status == "WARNING"
    # Try opportunistic add: propose 0.5 shares at 300 =150 EUR, standalone pct 1.5% (under standalone cap 3%) but total after would be 10.5% >7% -> blocked
    sidecar_cfg = sidecar.get_for_pie("TechPieShort")
    result = evaluate_opportunistic_standalone_add(
        symbol="AAPL_US_EQ",
        proposed_quantity=0.5,
        proposed_price_eur=300,
        canonical=aapl,
        sidecar=sidecar_cfg,
        total_equity_eur=total_equity,
        reconciliation_status="PASS",
        signal_validated=True,
    )
    assert result["allowed"] is False
    assert "exceeds cap" in result["reason"]


def test_pie_constituent_never_rendered_as_individual_sell():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    raw = _fake_api_position("NVDA_US_EQ", quantity=0.60, pie_quantity=0.60, current_price=180, ppl=0, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    view = generate_actionability_view(exps)
    for row in view["actions"]:
        if row["symbol"] == "NVDA_US_EQ":
            assert row["pie_action"] in ["HOLD_PIE", "ADD_VIA_PIE", "REDUCE_VIA_PIE", "REVIEW_PIE", "N/A"]
            assert row["pie_action"] != "SELL"
            assert row["pie_action"] != "SELL_STANDALONE"
            # Ensure standalone sell not misattributed to pie
            if "pies_involved" in row and row["pies_involved"]:
                assert row["pie_action"] != "SELL"


def test_daily_div_ignores_daily_ticker_trading_signal_by_default():
    """DailyDiv sidecar must not allow standalone buys for daily signals."""
    sidecar = load_sidecar_registry("PIEs/config")
    daily_cfg = sidecar.get_for_pie("DailyDivPieLongShort")
    assert daily_cfg.horizon == "long_term"
    assert daily_cfg.allow_standalone_adds is False
    assert daily_cfg.default_execution_mode == "pie_only"
    assert daily_cfg.default_action_scope == "pie"

    universe = load_pie_universe("PIEs")
    # Build exposure for a DailyDiv constituent, e.g., O_US_EQ (Realty Income)
    raw = _fake_api_position("O_US_EQ", quantity=5, pie_quantity=5, current_price=60, ppl=0, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    o_exp = next(e for e in exps if e.symbol == "O_US_EQ")
    assert o_exp.horizon == "long_term"
    assert o_exp.execution_capabilities.can_buy_standalone is False
    # Opportunistic add must be blocked even with validated signal
    result = evaluate_opportunistic_standalone_add(
        symbol="O_US_EQ",
        proposed_quantity=1,
        proposed_price_eur=60,
        canonical=o_exp,
        sidecar=daily_cfg,
        total_equity_eur=10000,
        reconciliation_status="PASS",
        signal_validated=True,
    )
    assert result["allowed"] is False
    assert "disabled" in result["reason"].lower()


def test_unknown_target_weights_do_not_produce_fake_rebalance():
    universe = load_pie_universe("PIEs")
    # TechPieShort has no weight column -> has_target_weights False
    tech = next(p for p in universe.pies if p.pie_id == "TechPieShort")
    assert tech.has_target_weights is False
    assert tech.weight_sum is None
    assert tech.weights_valid is None
    # Sidecar also has target_weights None
    sidecar = load_sidecar_registry("PIEs/config")
    tech_cfg = sidecar.get_for_pie("TechPieShort")
    assert tech_cfg.target_weights is None
    # Exposure should reflect unknown and not generate rebalance
    raw = _fake_api_position("NVDA_US_EQ", quantity=0.75, pie_quantity=0.60, current_price=180, ppl=0, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    nvda = next(e for e in exps if e.symbol == "NVDA_US_EQ")
    for pe in nvda.pie_exposures:
        assert pe.weight_in_pie is None


def test_account_reconciliation_behavior_unchanged():
    """Ensure reconciliation math still uses ALL positions value + reported cash (free + pie + blocked), not CSV values."""
    from trading212_portfolio import PortfolioMonitor
    import inspect
    src = inspect.getsource(PortfolioMonitor.get_portfolio_summary)
    # Must still compute derived_holdings_plus_reported_cash = all_positions_value_eur + free_cash + pie_cash + blocked
    assert "derived_holdings_plus_reported_cash" in src
    assert "all_positions_value_eur" in src
    assert "blocked" in src
    # Must not reference PIEs/*.csv for reconciliation
    assert "PIEs" not in src
    assert "PieLoader" not in src
    # Validate that parse_position still preserves original logic
    from trading212_portfolio import parse_position as pp
    src2 = inspect.getsource(pp)
    assert "value_eur" in src2
    assert "pie_quantity" in src2


def test_no_live_trading_or_broker_write_apis_introduced():
    """Check that new portfolio modules do not introduce live order APIs."""
    import pathlib
    import re
    forbidden = ["place_order", "sell_order", "buy_order", "post_order", "Trade212Client.*order", "requests.post.*order"]
    for fpath in [
        "investment_engine/portfolio/pie_metadata.py",
        "investment_engine/portfolio/sidecar.py",
        "investment_engine/portfolio/exposure.py",
    ]:
        text = pathlib.Path(fpath).read_text(encoding="utf-8")
        for pat in forbidden:
            assert not re.search(pat, text, flags=re.IGNORECASE), f"Forbidden broker write API pattern {pat} found in {fpath}"
        # Also ensure no direct Trade212Client import for write
        assert "Trade212Client" not in text
        # Ensure advisory disclaimer present where appropriate
        if "exposure.py" in fpath:
            assert "advisory" in text.lower()
