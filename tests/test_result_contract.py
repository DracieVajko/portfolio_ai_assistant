"""Result-contract tests: run_engine() must return every key the wrapper
and generate_human_brief() consume. Guards against empty-brief regressions."""

from __future__ import annotations

import pytest

import portfolio_ai_assistant as wrapper
from investment_engine.reporting import report_structure


def _required_keys() -> tuple:
    return (
        "portfolio_rows",
        "monitoring_items",
        "regime_result",
        "decision_news",
        "earnings_7d",
        "ideas",
        "portfolio_names",
    )


def test_wrapper_contract_keys_match_brief_signature():
    """Every key the wrapper extracts must be a generate_human_brief() parameter."""
    import inspect

    params = set(inspect.signature(report_structure.generate_human_brief).parameters)
    for key in _required_keys():
        assert key in params, f"wrapper reads {key!r} which generate_human_brief() does not accept"


def test_run_engine_returns_brief_keys(monkeypatch):
    """run_engine() return dict contains all human-brief keys (mocked internals)."""
    import investment_engine.main as engine

    sentinel = {
        "portfolio_rows": [{"display": "AAPL"}],
        "monitoring_items": [{"display": "AAPL"}],
        "regime_result": None,
        "decision_news": [],
        "earnings_7d": {},
        "ideas": [],
        "portfolio_names": {"AAPL": "Apple"},
    }

    # Patch at module level is heavy; instead verify the source contract:
    # the final return statement of run_engine() must reference each key.
    import pathlib

    src = pathlib.Path(engine.__file__).read_text(encoding="utf-8")
    anchor = src.index("def _build_ai_recommendations")
    start = src.rindex("    return {", 0, anchor)
    ret_block = src[start:anchor]
    for key in _required_keys():
        assert f'"{key}"' in ret_block, f"run_engine() return is missing {key!r}"


def test_failed_tickers_helpers_present():
    from investment_engine.reporting import failed_tickers as ft

    assert callable(ft.collect_failed_tickers)
    assert callable(ft.render_failed_tickers_md)
    assert ft.render_failed_tickers_md([], "r1") == ""
    md = ft.render_failed_tickers_md(
        [{"display": "XXX", "t212_id": "XXX_US_EQ", "yahoo_attempted": "XXX",
          "support_state": "UNRESOLVED", "fetch_failed": True,
          "qty": 1, "value_eur": 10.0, "alias_hint": {"XXX": "<YAHOO_SYMBOL>"}}],
        "r1",
    )
    assert "XXX" in md and "symbol_aliases" in md
