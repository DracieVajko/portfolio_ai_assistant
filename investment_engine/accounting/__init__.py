"""Historical Transaction Ledger – Read-Only Accounting Layer.

This module imports manually exported Trading 212 transaction/activity CSV files
and calculates historical performance without corrupting or replacing live
account reconciliation.

Key principles:
- Read-only: never calls Trading 212 API or any external service
- Source priority: actual CSV > manual adjustments > broker API (if verified)
- Decimal for all monetary calculations
- Timezone-aware timestamps
- Unknown events force PARTIAL status unless explicitly ignored
- Raw CSV rows preserved locally only, never in user-facing reports

This module is independent from the live portfolio reconciliation in
trading212_portfolio.py and investment_engine/portfolio/.
"""

from investment_engine.accounting.models import (
    EventType,
    ParseStatus,
    LedgerEvent,
    Lot,
    MatchedTrade,
    LedgerSummary,
    ManualAdjustments,
)

from investment_engine.accounting.importers.t212_csv import (
    import_t212_activity_csv,
    import_t212_cash_csv,
    detect_csv_type,
)

from investment_engine.accounting.ledger import (
    Ledger,
    build_ledger,
)

from investment_engine.accounting.lot_matching import (
    match_sell_fifo,
    MatchingResult,
)

from investment_engine.accounting.fees import (
    categorize_fee,
    FeeCategory,
)

from investment_engine.accounting.performance import (
    calculate_performance,
    PerformanceResult,
)

from investment_engine.accounting.reconciliation import (
    reconcile_ledger,
    ReconciliationResult,
)

__all__ = [
    # Models
    "EventType",
    "ParseStatus",
    "LedgerEvent",
    "Lot",
    "MatchedTrade",
    "LedgerSummary",
    "ManualAdjustments",
    # Importers
    "import_t212_activity_csv",
    "import_t212_cash_csv",
    "detect_csv_type",
    # Ledger
    "Ledger",
    "build_ledger",
    # Lot matching
    "match_sell_fifo",
    "MatchingResult",
    # Fees
    "categorize_fee",
    "FeeCategory",
    # Performance
    "calculate_performance",
    "PerformanceResult",
    # Reconciliation
    "reconcile_ledger",
    "ReconciliationResult",
]