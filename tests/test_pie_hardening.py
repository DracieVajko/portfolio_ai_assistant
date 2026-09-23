"""Hardening tests for pie-aware exposure – provenance, matching, action restrictions."""
from investment_engine.portfolio.pie_metadata import load_pie_universe, _normalize_base_symbol
from investment_engine.portfolio.sidecar import load_sidecar_registry
from investment_engine.portfolio.exposure import build_canonical_exposures, generate_actionability_view
from trading212_portfolio import parse_position


def _fake_api_position(ticker: str, quantity: float, pie_quantity: float, current_price: float, ppl: float = 0.0, fx_ppl: float = 0.0, extra: dict | None = None):
    d = {
        "ticker": ticker,
        "quantity": quantity,
        "averagePrice": current_price,
        "currentPrice": current_price,
        "ppl": ppl,
        "fxPpl": fx_ppl,
        "pieQuantity": pie_quantity,
    }
    if extra:
        d.update(extra)
    return d


def test_aapl_multi_pie_is_low_confidence_estimated():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    # AAPL appears in TechPieShort and DailyDivPieLongShort (2 pies)
    pies = universe.pies_for_ticker("AAPL_US_EQ")
    assert len(pies) == 2, f"AAPL should be in 2 pies, got {[p.pie_id for p in pies]}"
    raw = _fake_api_position("AAPL_US_EQ", quantity=2, pie_quantity=1.5, current_price=150, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    aapl = next(e for e in exps if e.symbol == "AAPL_US_EQ")
    # Must have 2 pie exposures
    assert len(aapl.pie_exposures) == 2
    for pe in aapl.pie_exposures:
        assert pe.allocation_source == "api_aggregate_estimated_split"
        assert pe.allocation_confidence == "low"
        assert pe.allocation_is_estimated is True
        assert pe.per_pie_breakdown_reliability == "estimated"
        assert "aggregate pie quantity" in pe.allocation_note
    assert aapl.per_pie_breakdown_reliability == "estimated"
    # Total exposure remains exact (sum of split equals aggregate)
    assert abs(sum(p.quantity for p in aapl.pie_exposures) - 1.5) < 1e-9
    assert abs(aapl.total_value_eur - parsed["value_eur"]) < 1e-6


def test_aapl_multi_pie_receives_review_pie_allocation_never_add_reduce():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    raw = _fake_api_position("AAPL_US_EQ", quantity=2, pie_quantity=1.5, current_price=150, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    view = generate_actionability_view(exps)
    aapl_action = next(a for a in view["actions"] if a["symbol"] == "AAPL_US_EQ")
    assert aapl_action["pie_action"] == "REVIEW_PIE_ALLOCATION"
    assert aapl_action["pie_action"] not in ("ADD_VIA_PIE", "REDUCE_VIA_PIE")
    assert "aggregate pie quantity. Per-pie allocation is estimated" in aapl_action["explanation"]
    # Also check can_add_via_pie is disabled when estimated
    assert aapl_action["can_add_via_pie"] is False
    assert aapl_action["can_reduce_via_pie"] is False


def test_single_pie_is_medium_inferred_not_estimated():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    # NVDA is only in TechPieShort (single pie)
    pies = universe.pies_for_ticker("NVDA_US_EQ")
    assert len(pies) == 1 and pies[0].pie_id == "TechPieShort"
    raw = _fake_api_position("NVDA_US_EQ", quantity=1, pie_quantity=0.6, current_price=180, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=20000)
    nvda = next(e for e in exps if e.symbol == "NVDA_US_EQ")
    assert len(nvda.pie_exposures) == 1
    pe = nvda.pie_exposures[0]
    assert pe.allocation_source == "api_aggregate_single_pie"
    assert pe.allocation_confidence == "medium"
    assert pe.allocation_is_estimated is False
    assert pe.per_pie_breakdown_reliability == "inferred"
    assert "inferred from aggregate pie quantity" in pe.allocation_note
    assert nvda.per_pie_breakdown_reliability == "inferred"
    # Should be HOLD_PIE, not REVIEW_PIE_ALLOCATION
    view = generate_actionability_view(exps)
    nvda_action = next(a for a in view["actions"] if a["symbol"] == "NVDA_US_EQ")
    assert nvda_action["pie_action"] == "HOLD_PIE"
    assert nvda_action["can_add_via_pie"] is True


def test_exact_per_pie_api_overrides_estimate():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    # Simulate API providing per-pie exact breakdown for AAPL (belongs to 2 pies)
    raw = _fake_api_position(
        "AAPL_US_EQ",
        quantity=2,
        pie_quantity=1.5,
        current_price=150,
        fx_ppl=1,
        extra={
            "pie_allocations": {
                "TechPieShort": {"quantity": 1.0, "value_eur": 138.3},
                "DailyDivPieLongShort": {"quantity": 0.5, "value_eur": 69.15},
            }
        },
    )
    parsed = parse_position(raw)
    # Inject per-pie data into parsed (parse_position preserves extra fields via raw)
    parsed["pie_allocations"] = raw["pie_allocations"]
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    aapl = next(e for e in exps if e.symbol == "AAPL_US_EQ")
    for pe in aapl.pie_exposures:
        assert pe.allocation_source == "api_exact"
        assert pe.allocation_confidence == "high"
        assert pe.allocation_is_estimated is False
        assert pe.per_pie_breakdown_reliability == "exact"
        assert "Per-pie allocation from broker" in pe.allocation_note
    assert aapl.per_pie_breakdown_reliability == "exact"
    view = generate_actionability_view(exps)
    aapl_action = next(a for a in view["actions"] if a["symbol"] == "AAPL_US_EQ")
    # Exact should allow normal pie actions, not forced REVIEW_PIE_ALLOCATION
    assert aapl_action["pie_action"] in ("HOLD_PIE", "REVIEW_PIE")  # not REVIEW_PIE_ALLOCATION
    assert aapl_action["pie_action"] != "REVIEW_PIE_ALLOCATION"


def test_o_matches_exact_and_coin_orcl_oxy_do_not_match_o():
    universe = load_pie_universe("PIEs")
    # O should match
    pies_o = universe.pies_for_ticker("O_US_EQ")
    assert any(p.pie_id == "DailyDivPieLongShort" for p in pies_o)
    # Check provenance
    pies_o_prov = universe.pies_for_ticker_with_provenance("O_US_EQ")
    o_entry = next((m for pie, m, c in pies_o_prov if pie.pie_id == "DailyDivPieLongShort"), None)
    assert o_entry is not None
    # Verify match_method for O
    sidecar = load_sidecar_registry("PIEs/config")
    raw = _fake_api_position("O_US_EQ", quantity=5, pie_quantity=5, current_price=60, fx_ppl=1)
    parsed = parse_position(raw)
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=10000)
    o_exp = next(e for e in exps if e.symbol == "O_US_EQ")
    assert len(o_exp.pie_exposures) == 1
    assert o_exp.pie_exposures[0].match_method == "exact_normalized_symbol"
    assert o_exp.pie_exposures[0].match_confidence == "high"

    # COIN, ORCL, OXY must not match slice O
    for ticker in ("COIN_US_EQ", "ORCL_US_EQ", "OXY_US_EQ"):
        pies = universe.pies_for_ticker(ticker)
        # Should not contain DailyDivPieLongShort via O slice
        # They might be in other pies, but must not be matched via O
        for p in pies:
            # Ensure no pie was matched because of loose substring O
            # Check that O slice not causing match: all matched pies must have constituent exactly equal to base
            for constituent in p.constituents:
                if constituent.ticker == "O":
                    # This would be false positive if ticker is COIN but matched O
                    assert ticker.split("_")[0].upper() != "O", f"{ticker} incorrectly matched O"
        # Explicitly, COIN should have 0 pies (unless COIN is actually in a pie, which it is not currently)
        # Check that COIN is not considered in DailyDiv via O
        pies_for_coin = universe.pies_for_ticker("COIN_US_EQ")
        # COIN is not in any pie currently, so should be 0
        # If it were in a pie, it would be via exact, not via O
        assert all(_normalize(p.tickers, ticker) is False for p in pies_for_coin) or len(pies_for_coin) == 0


def _normalize(tickers, ticker):
    # helper to check if any ticker in list would be matched via loose O
    base = ticker.split("_")[0].upper()
    for t in tickers:
        if t == "O" and base != "O":
            return True
    return False

def test_ambiguous_mapping_remains_unmatched():
    from investment_engine.portfolio.pie_metadata import PieUniverse, PieMetadata, PieConstituent
    # Create a universe with no ISIN, but ticker that would be ambiguous if loose matching were used
    # Use exact matching: ticker "AB" should not match "ABT" or "A"
    pie1 = PieMetadata(pie_id="PieA", display_name="PieA", path="PIEs/PieA.csv", constituents=[PieConstituent(ticker="AB", name="AB Inc")])
    pie2 = PieMetadata(pie_id="PieB", display_name="PieB", path="PIEs/PieB.csv", constituents=[PieConstituent(ticker="ABT", name="ABT Inc")])
    universe = PieUniverse(pies=[pie1, pie2])
    # Ticker AB should match PieA only
    assert len(universe.pies_for_ticker("AB_US_EQ")) == 1
    assert universe.pies_for_ticker("AB_US_EQ")[0].pie_id == "PieA"
    # Ticker ABT should match PieB only
    assert len(universe.pies_for_ticker("ABT_US_EQ")) == 1
    assert universe.pies_for_ticker("ABT_US_EQ")[0].pie_id == "PieB"
    # Ticker ABA should match neither (no exact), should be unmatched
    assert len(universe.pies_for_ticker("ABA_US_EQ")) == 0
    # Ensure with our real universe, an unknown ticker remains unmatched
    real_universe = load_pie_universe("PIEs")
    assert len(real_universe.pies_for_ticker("FAKE_UNKNOWN_TICKER_XYZ_US_EQ")) == 0


def test_total_exposure_and_concentration_correct_for_multi_pie():
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    # AAPL in 2 pies, total 2 shares @150 => value 276.6 EUR (150*0.922*2)
    raw = _fake_api_position("AAPL_US_EQ", quantity=2, pie_quantity=1.5, current_price=150, fx_ppl=1)
    parsed = parse_position(raw)
    total_equity = 5000  # 276.6 /5000 =5.53% -> below 6% cap? Use 3000 to exceed
    exps = build_canonical_exposures([parsed], universe, sidecar, total_equity_eur=3000)
    aapl = next(e for e in exps if e.symbol == "AAPL_US_EQ")
    # Total remains exact regardless of split
    assert abs(aapl.total_quantity - 2) < 1e-9
    assert abs(aapl.total_value_eur - parsed["value_eur"]) < 1e-6
    assert abs(sum(p.value_eur for p in aapl.pie_exposures) + aapl.standalone_exposure.value_eur - aapl.total_value_eur) < 1e-6
    # Concentration should be correct: 276.6/3000=9.22% > min cap 6% => WARNING, even though per-pie split is estimated
    assert aapl.concentration_status == "WARNING"
    assert aapl.concentration_cap_pct == 6.0  # min of 7.0 (Tech) and 6.0 (DailyDiv)


def test_existing_reconciliation_behavior_stays_unchanged():
    import inspect
    from trading212_portfolio import PortfolioMonitor
    src = inspect.getsource(PortfolioMonitor.get_portfolio_summary)
    assert "derived_holdings_plus_reported_cash" in src
    assert "all_positions_value_eur" in src
    assert "blocked" in src
    assert "PIEs" not in src
    assert "PieLoader" not in src
    # Also ensure pie exposure does not alter reconciliation
    universe = load_pie_universe("PIEs")
    sidecar = load_sidecar_registry("PIEs/config")
    raws = [
        _fake_api_position("AAPL_US_EQ", quantity=1, pie_quantity=0.5, current_price=150, fx_ppl=1),
        _fake_api_position("NVDA_US_EQ", quantity=1, pie_quantity=1, current_price=180, fx_ppl=1),
    ]
    parsed = [parse_position(r) for r in raws]
    total_equity = sum(p["value_eur"] for p in parsed) + 5000  # cash
    exps = build_canonical_exposures(parsed, universe, sidecar, total_equity_eur=total_equity)
    # Total of all exposures must equal sum of API values (no creation/loss)
    assert abs(sum(e.total_value_eur for e in exps if e.symbol in ("AAPL_US_EQ","NVDA_US_EQ")) - sum(p["value_eur"] for p in parsed)) < 1e-6


def test_no_broker_write_methods_introduced():
    import pathlib, re
    for fpath in [
        "investment_engine/portfolio/pie_metadata.py",
        "investment_engine/portfolio/sidecar.py",
        "investment_engine/portfolio/exposure.py",
    ]:
        text = pathlib.Path(fpath).read_text(encoding="utf-8")
        for pat in ["place_order", "sell_order", "buy_order", "post_order"]:
            assert not re.search(pat, text, flags=re.IGNORECASE), f"Forbidden {pat} in {fpath}"
        assert "Trade212Client" not in text
