"""Historical ledger reconciliation against broker data.

Compares ledger-derived values with broker-provided account data
for validation purposes only – never overwrites live reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from investment_engine.accounting.models import LedgerSummary
from investment_engine.accounting.performance import PerformanceResult
from investment_engine.accounting.lot_matching import MatchingResult


@dataclass
class ReconciliationResult:
    """Result of comparing ledger with broker data."""
    # Deposit reconciliation
    ledger_net_deposits_eur: Decimal = Decimal("0")
    broker_net_deposits_eur: Optional[Decimal] = None
    deposit_difference_eur: Optional[Decimal] = None
    deposit_reconciled: bool = False
    
    # Position reconciliation (diagnostic only)
    ledger_open_lots_cost_basis_eur: Decimal = Decimal("0")
    broker_positions_value_eur: Optional[Decimal] = None
    position_difference_eur: Optional[Decimal] = None
    
    # Performance reconciliation (diagnostic only)
    ledger_net_realized_pnl_eur: Decimal = Decimal("0")
    broker_account_return_eur: Optional[Decimal] = None
    performance_difference_eur: Optional[Decimal] = None
    
    # Overall
    status: str = "NO_BROKER_DATA"  # RECONCILED | PARTIAL | FAILED | NO_BROKER_DATA
    warnings: List[str] = None
    notes: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.notes is None:
            self.notes = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_net_deposits_eur": str(self.ledger_net_deposits_eur),
            "broker_net_deposits_eur": str(self.broker_net_deposits_eur) if self.broker_net_deposits_eur else None,
            "deposit_difference_eur": str(self.deposit_difference_eur) if self.deposit_difference_eur else None,
            "deposit_reconciled": self.deposit_reconciled,
            "ledger_open_lots_cost_basis_eur": str(self.ledger_open_lots_cost_basis_eur),
            "broker_positions_value_eur": str(self.broker_positions_value_eur) if self.broker_positions_value_eur else None,
            "position_difference_eur": str(self.position_difference_eur) if self.position_difference_eur else None,
            "ledger_net_realized_pnl_eur": str(self.ledger_net_realized_pnl_eur),
            "broker_account_return_eur": str(self.broker_account_return_eur) if self.broker_account_return_eur else None,
            "performance_difference_eur": str(self.performance_difference_eur) if self.performance_difference_eur else None,
            "status": self.status,
            "warnings": self.warnings,
            "notes": self.notes,
        }


def reconcile_ledger(
    performance: PerformanceResult,
    broker_net_deposits_eur: Optional[Decimal] = None,
    broker_positions_value_eur: Optional[Decimal] = None,
    broker_account_return_eur: Optional[Decimal] = None,
    deposit_tolerance_eur: Decimal = Decimal("10.0"),  # €10 tolerance
    position_tolerance_pct: Decimal = Decimal("5.0"),  # 5% tolerance
) -> ReconciliationResult:
    """Reconcile historical ledger against broker-provided data.
    
    All comparisons are diagnostic only – they do not affect the live
    account reconciliation in trading212_portfolio.py.
    
    Args:
        performance: PerformanceResult from calculate_performance()
        broker_net_deposits_eur: Total deposits - withdrawals from broker records
        broker_positions_value_eur: Current market value of all positions from broker
        broker_account_return_eur: Broker's reported account return (equity - net deposits)
        deposit_tolerance_eur: Acceptable difference for deposit reconciliation
        position_tolerance_pct: Acceptable % difference for position reconciliation
    
    Returns:
        ReconciliationResult with differences and status.
    """
    result = ReconciliationResult(
        ledger_net_deposits_eur=performance.summary.net_deposits_eur,
        ledger_open_lots_cost_basis_eur=performance.summary.open_lot_cost_basis_eur,
        ledger_net_realized_pnl_eur=performance.summary.net_realized_pnl_after_known_costs_eur,
    )
    
    # Deposit reconciliation
    if broker_net_deposits_eur is not None:
        result.broker_net_deposits_eur = broker_net_deposits_eur
        result.deposit_difference_eur = abs(performance.summary.net_deposits_eur - broker_net_deposits_eur)
        result.deposit_reconciled = result.deposit_difference_eur <= deposit_tolerance_eur
        if not result.deposit_reconciled:
            result.warnings.append(
                f"Deposit mismatch: ledger={performance.summary.net_deposits_eur} EUR, "
                f"broker={broker_net_deposits_eur} EUR, diff={result.deposit_difference_eur} EUR"
            )
        else:
            result.notes.append("Net deposits reconciled within tolerance")
    else:
        result.notes.append("Broker net deposits not provided – deposit reconciliation skipped")
    
    # Position reconciliation (cost basis vs current market value – different metrics)
    if broker_positions_value_eur is not None:
        result.broker_positions_value_eur = broker_positions_value_eur
        # Note: ledger tracks cost basis, broker reports market value – not directly comparable
        # This is a diagnostic comparison only
        if performance.summary.open_lot_cost_basis_eur > 0:
            result.position_difference_eur = broker_positions_value_eur - performance.summary.open_lots_cost_basis_eur
            pct_diff = abs(result.position_difference_eur) / performance.summary.open_lot_cost_basis_eur * Decimal("100")
            if pct_diff > position_tolerance_pct:
                result.warnings.append(
                    f"Position value vs cost basis difference: {pct_diff:.1f}% "
                    f"(cost basis={performance.summary.open_lot_cost_basis_eur} EUR, "
                    f"market value={broker_positions_value_eur} EUR)"
                )
        result.notes.append("Position comparison: ledger shows cost basis, broker shows market value")
    else:
        result.notes.append("Broker positions value not provided – position reconciliation skipped")
    
    # Performance reconciliation
    if broker_account_return_eur is not None:
        result.broker_account_return_eur = broker_account_return_eur
        result.performance_difference_eur = abs(
            performance.summary.net_realized_pnl_after_known_costs_eur - broker_account_return_eur
        )
        result.notes.append(
            f"Performance comparison: ledger net realized P/L after known costs = "
            f"{performance.summary.net_realized_pnl_after_known_costs_eur} EUR, "
            f"broker account return = {broker_account_return_eur} EUR"
        )
        result.notes.append(
            "Note: Ledger net realized P/L excludes unrealized P/L and unknown costs. "
            "Broker account return = equity - net deposits (includes unrealized)."
        )
    else:
        result.notes.append("Broker account return not provided – performance reconciliation skipped")
    
    # Overall status
    if broker_net_deposits_eur is None and broker_positions_value_eur is None and broker_account_return_eur is None:
        result.status = "NO_BROKER_DATA"
    elif result.deposit_reconciled is False:
        result.status = "FAILED"
    elif matching_issues(performance):
        result.status = "PARTIAL"
    else:
        result.status = "RECONCILED"
    
    # Add warnings from performance
    result.warnings.extend(performance.completeness_warnings)
    
    return result


def matching_issues(performance: PerformanceResult) -> bool:
    """Check if performance result has matching/data quality issues."""
    return (
        performance.summary.unmatched_sell_count > 0
        or performance.summary.unknown_event_count > 0
        or performance.data_quality != "RECONCILED"
    )