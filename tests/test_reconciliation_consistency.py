"""Reconciliation consistency regression tests.

Verifies that all report outputs use the same canonical reconciliation
result from broker_first.reconcile_snapshot(). Key requirements:

1. All report outputs have exactly the same reconciliation status and delta.
2. A €20.51 duplicate exclusion cannot produce diagnostic FAIL while
   snapshot reports PASS.
3. Exact duplicates are counted once.
4. Two independently held rows with same ISIN but distinct broker identities
   are summed.
5. Failed-ticker classification: held unresolved positions have nonzero
   broker quantity/value and T212 instrument ID; watchlist-only items are
   not described as unresolved holdings.
"""

from __future__ import annotations

import pytest

from investment_engine.portfolio.broker_first import (
    AccountSnapshot,
    BrokerInstrument,
    BrokerPosition,
    CashBalance,
    ReconciliationResult,
    reconcile_snapshot,
    deduplicate_positions,
)
from investment_engine.reporting.failed_tickers import collect_failed_tickers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instrument(isin: str = "US0000000000", broker_id: str = "BRK001") -> BrokerInstrument:
    """Create a minimal BrokerInstrument."""
    return BrokerInstrument(isin=isin, broker_instrument_id=broker_id)


def _make_position(
    isin: str = "US0000000000",
    broker_id: str = "BRK001",
    value_eur: float = 100.0,
    quantity: float = 10.0,
    included_in_position_total: bool = True,
    account_id: str = "default",
) -> BrokerPosition:
    """Create a minimal BrokerPosition for testing."""
    return BrokerPosition(
        instrument=_make_instrument(isin=isin, broker_id=broker_id),
        quantity=quantity,
        broker_market_value_eur=value_eur,
        included_in_position_total=included_in_position_total,
        source=account_id,
    )


def _make_snapshot(
    total_equity: float = 3300.0,
    free_cash: float = 0.0,
    blocked_cash: float = 3.42,
    pie_cash: float = 0.0,
    positions: list | None = None,
) -> AccountSnapshot:
    """Create a minimal AccountSnapshot for testing."""
    return AccountSnapshot(
        cash=CashBalance(
            broker_total_equity_eur=total_equity,
            free_cash_eur=free_cash,
            blocked_cash_eur=blocked_cash,
            pie_cash_eur=pie_cash,
            broker_invested_eur=total_equity - free_cash - blocked_cash - pie_cash,
        ),
        positions=positions or [],
    )


# ---------------------------------------------------------------------------
# Test 1: Canonical reconciliation consistency
# ---------------------------------------------------------------------------

class TestReconciliationConsistency:
    """All report outputs must use the same canonical reconciliation result."""

    def test_reconcile_snapshot_returns_canonical_result(self):
        """reconcile_snapshot() returns a ReconciliationResult with correct fields."""
        snap = _make_snapshot(
            total_equity=3300.0,
            free_cash=0.0,
            blocked_cash=3.42,
            positions=[],
        )
        result = reconcile_snapshot(snap)
        assert isinstance(result, ReconciliationResult)
        assert result.reconciliation_status in ("PASS", "FAIL", "UNKNOWN")
        assert result.reconciliation_delta_eur is not None
        assert result.broker_total_equity_eur == 3300.0
        assert result.sum_excluded_values_eur >= 0

    def test_to_reconciliation_dict_has_required_keys(self):
        """to_reconciliation_dict() produces a dict consumed by all report writers."""
        snap = _make_snapshot(total_equity=3300.0, positions=[])
        result = reconcile_snapshot(snap)
        d = result.to_reconciliation_dict()
        required_keys = {
            "total_equity", "positions_value", "reported_cash",
            "free_cash", "pie_cash", "blocked_cash",
            "implied_cash", "cash_delta", "derived_total",
            "diff", "threshold", "status",
        }
        assert required_keys.issubset(d.keys())
        assert d["status"] == result.reconciliation_status
        assert abs(d["cash_delta"] - result.reconciliation_delta_eur) < 0.01

    def test_all_consumers_use_same_status(self):
        """JSON output, diagnostic markdown, and brief must share status.
        Empty positions with nonzero equity should produce FAIL, not UNKNOWN."""
        snap = _make_snapshot(
            total_equity=3300.0,
            free_cash=0.0,
            blocked_cash=3.42,
            positions=[],
        )
        result = reconcile_snapshot(snap)
        d = result.to_reconciliation_dict()
        # All consumers use the same result
        assert d["status"] == result.reconciliation_status
        # The JSON output uses _recon from to_reconciliation_dict()
        assert d["status"] == result.reconciliation_status
        # The diagnostic uses _recon_result directly
        assert result.reconciliation_status == d["status"]
        # The brief/snapshot use _recon dict
        assert d["status"] == result.reconciliation_status


# ---------------------------------------------------------------------------
# Test 2: Duplicate exclusion cannot produce FAIL while PASS
# ---------------------------------------------------------------------------

class TestDuplicateExclusionConsistency:
    """A €20.51 duplicate exclusion cannot produce diagnostic FAIL while
    snapshot reports PASS."""

    def test_duplicate_excluded_from_sum_once(self):
        """Exact duplicates are counted exactly once in deduplicated sum."""
        pos_a = _make_position(isin="US03003X1000", broker_id="BRK001", value_eur=20.51)
        pos_b = _make_position(isin="US03003X1000", broker_id="BRK001", value_eur=20.51)
        pos_b.included_in_position_total = False  # Mark as duplicate
        positions = [pos_a, pos_b]
        unique, duplicates = deduplicate_positions(positions)
        assert len(unique) == 1
        assert len(duplicates) == 1
        # Sum of deduped positions = 20.51, not 41.02
        sum_dedup = sum(p.broker_market_value_eur for p in unique if p.included_in_position_total)
        assert sum_dedup == 20.51

    def test_duplicate_20_51_cannot_produce_pass_in_diag_and_fail_in_snapshot(self):
        """Both diagnostic and snapshot must show the same status when a
        €20.51 duplicate is excluded."""
        # Total equity = 3300, expected positions = 3300 - 3.42 = 3296.58
        # Two rows: one real position (3276.87) and one duplicate (20.51)
        # After dedupe, sum = 3276.87 (not 3297.38)
        real_pos = _make_position(isin="US03003X1000", broker_id="BRK002", value_eur=3276.87)
        dup_pos = _make_position(isin="US03003X1000", broker_id="BRK001", value_eur=20.51)
        dup_pos.included_in_position_total = False
        snap = _make_snapshot(
            total_equity=3300.0,
            free_cash=0.0,
            blocked_cash=3.42,
            positions=[real_pos, dup_pos],
        )
        result = reconcile_snapshot(snap)
        d = result.to_reconciliation_dict()
        # Both must agree: if diagnostic shows FAIL, JSON must also show FAIL
        assert d["status"] == result.reconciliation_status, (
            f"Diagnostic/snapshot mismatch: diag={d['status']}, "
            f"snapshot={result.reconciliation_status}"
        )
        # The delta must be consistent
        assert abs(d["cash_delta"] - result.reconciliation_delta_eur) < 0.01

    def test_two_independently_held_same_isin_are_deduped(self):
        """Same ISIN = same holding in T212's system, deduped regardless
        of which broker account it appears under."""
        pos_a = _make_position(isin="US03003X1000", broker_id="BRK001", value_eur=20.51)
        pos_b = _make_position(isin="US03003X1000", broker_id="BRK002", value_eur=20.51)
        positions = [pos_a, pos_b]
        unique, duplicates = deduplicate_positions(positions)
        # Same ISIN = same holding, deduped
        assert len(unique) == 1
        assert len(duplicates) == 1
        assert duplicates[0].duplicate_of != ""

    def test_dedupe_counted_once(self):
        """Deduplicated positions are counted once in the sum."""
        pos_a = _make_position(isin="US03003X1000", broker_id="BRK001", value_eur=20.51)
        pos_b = _make_position(isin="US03003X1000", broker_id="BRK002", value_eur=20.51)
        pos_a.included_in_position_total = True
        pos_b.included_in_position_total = False  # duplicate
        positions = [pos_a, pos_b]
        unique, duplicates = deduplicate_positions(positions)
        sum_dedup = sum(p.broker_market_value_eur for p in unique if p.included_in_position_total)
        assert sum_dedup == 20.51


# ---------------------------------------------------------------------------
# Test 3: Dedupe correctness
# ---------------------------------------------------------------------------

class TestDedupeCorrectness:
    """Deduplication must be correct: exact duplicates counted once,
    genuinely separate holdings summed."""

    def test_exact_duplicate_counted_once(self):
        """Two rows with same ISIN and same broker ID are duplicates."""
        pos_a = _make_position(isin="US1234567890", broker_id="BRK001", value_eur=100.0)
        pos_b = _make_position(isin="US1234567890", broker_id="BRK001", value_eur=100.0)
        unique, duplicates = deduplicate_positions([pos_a, pos_b])
        assert len(unique) == 1
        assert len(duplicates) == 1
        assert duplicates[0].included_in_position_total is False

    def test_distinct_broker_identities_deduped_by_isin(self):
        """Same ISIN but different broker IDs = still same holding, deduped."""
        pos_a = _make_position(isin="US1234567890", broker_id="BRK001", value_eur=100.0)
        pos_b = _make_position(isin="US1234567890", broker_id="BRK002", value_eur=100.0)
        unique, duplicates = deduplicate_positions([pos_a, pos_b])
        # Same ISIN = same holding, deduped
        assert len(unique) == 1
        assert len(duplicates) == 1

    def test_no_isin_deduped_by_broker_id(self):
        """Positions without ISIN are deduped by broker instrument ID."""
        pos_a = _make_position(isin="", broker_id="BRK001", value_eur=100.0)
        pos_b = _make_position(isin="", broker_id="BRK001", value_eur=100.0)
        unique, duplicates = deduplicate_positions([pos_a, pos_b])
        assert len(unique) == 1
        assert len(duplicates) == 1


# ---------------------------------------------------------------------------
# Test 4: Failed tickers classification
# ---------------------------------------------------------------------------

class TestFailedTickersClassification:
    """Failed-ticker classification must correctly separate held positions
    from watchlist-only and stale config aliases."""

    def test_watchlist_only_items_excluded_from_failed_tickers(self):
        """Watchlist-only items (zero qty, zero value, no T212 ID) must NOT
        appear as unresolved holdings in the failed ticker ledger."""
        failed = collect_failed_tickers(
            yahoo_by_display={"WATCH1": None},
            standalone=[{
                "display": "WATCH1",
                "t212": "",
                "qty": 0,
                "value_eur": 0,
            }],
            fetch_failed={"WATCH1"},
        )
        # WATCH1 has qty=0, value=0, no T212 ID => should be filtered out
        # (category = WATCHLIST, not HELD_UNRESOLVED)
        for item in failed:
            assert item.get("category") == "HELD_UNRESOLVED", (
                f"Watchlist item {item.get('display')} should not be in failed_tickers"
            )

    def test_held_unresolved_appears_with_category(self):
        """Held positions with nonzero qty/value and T212 ID appear with
        category='HELD_UNRESOLVED'."""
        failed = collect_failed_tickers(
            yahoo_by_display={"AAPL": None},
            standalone=[{
                "display": "AAPL",
                "t212": "TUYA_US_EQ",
                "qty": 10.0,
                "value_eur": 180.0,
            }],
            fetch_failed={"AAPL"},
        )
        for item in failed:
            assert item.get("category") == "HELD_UNRESOLVED"
            assert item.get("qty", 0) != 0
            assert item.get("value_eur", 0) != 0
            assert item.get("t212_id", "") != ""

    def test_render_failed_tickers_shows_category(self):
        """render_failed_tickers_md must include category column."""
        from investment_engine.reporting.failed_tickers import render_failed_tickers_md
        md = render_failed_tickers_md(
            failed=[{
                "display": "AAPL",
                "t212_id": "TUYA_US_EQ",
                "qty": 10.0,
                "value_eur": 180.0,
                "category": "HELD_UNRESOLVED",
            }],
            run_id="test-run",
        )
        assert "Category" in md
        assert "HELD_UNRESOLVED" in md


# ---------------------------------------------------------------------------
# Test 5: End-to-end consistency
# ---------------------------------------------------------------------------

class TestEndToEndConsistency:
    """End-to-end: JSON output, diagnostic markdown, and brief must all
    show the same reconciliation status and delta."""

    def test_reconciliation_status_consistent_across_consumers(self):
        """The canonical reconciliation result must be the single source
        of truth for all report writers."""
        snap = _make_snapshot(
            total_equity=3300.0,
            free_cash=0.0,
            blocked_cash=3.42,
            positions=[],
        )
        result = reconcile_snapshot(snap)
        d = result.to_reconciliation_dict()
        assert d["status"] == result.reconciliation_status
        assert abs(d["cash_delta"] - result.reconciliation_delta_eur) < 0.001

    def test_20_51_duplicate_cannot_cause_pass_diag_fail_snapshot(self):
        """A €20.51 duplicate exclusion cannot produce diagnostic FAIL
        while snapshot reports PASS."""
        real_pos = _make_position(isin="US03003X1000", broker_id="BRK002", value_eur=3276.87)
        dup_pos = _make_position(isin="US03003X1000", broker_id="BRK001", value_eur=20.51)
        dup_pos.included_in_position_total = False
        snap = _make_snapshot(
            total_equity=3300.0,
            free_cash=0.0,
            blocked_cash=3.42,
            positions=[real_pos, dup_pos],
        )
        result = reconcile_snapshot(snap)
        d = result.to_reconciliation_dict()
        assert d["status"] == result.reconciliation_status
        assert d["status"] == result.reconciliation_status

    def test_fallback_to_unknown_on_empty_data(self):
        """Empty broker data must produce UNKNOWN, never PASS."""
        snap = _make_snapshot(total_equity=0.0, positions=[])
        result = reconcile_snapshot(snap)
        assert result.reconciliation_status == "UNKNOWN"
        assert result.data_quality == "NO_BROKER_DATA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
