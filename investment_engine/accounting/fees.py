"""Fee categorization and cost accounting for historical ledger.

Provides canonical fee categories and mapping from event types to categories.
All fee amounts tracked in EUR for consistent reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from investment_engine.accounting.models import LedgerEvent, EventType


class FeeCategory(str, Enum):
    """Canonical fee categories for reporting."""
    TRADING_COMMISSION = "trading_commission"
    FX_CONVERSION = "fx_conversion"
    STAMP_DUTY = "stamp_duty"
    FINANCIAL_TRANSACTION_TAX = "financial_transaction_tax"
    DIVIDEND_WITHHOLDING_TAX = "dividend_withholding_tax"
    OTHER = "other"


# Mapping from EventType to FeeCategory
EVENT_TYPE_TO_FEE_CATEGORY = {
    EventType.COMMISSION: FeeCategory.TRADING_COMMISSION,
    EventType.FX_CONVERSION_FEE: FeeCategory.FX_CONVERSION,
    EventType.STAMP_DUTY: FeeCategory.STAMP_DUTY,
    EventType.FINANCIAL_TRANSACTION_TAX: FeeCategory.FINANCIAL_TRANSACTION_TAX,
    EventType.DIVIDEND_WITHHOLDING_TAX: FeeCategory.DIVIDEND_WITHHOLDING_TAX,
    # Unknown fees go to OTHER
}


@dataclass
class FeeBreakdown:
    """Breakdown of fees by category in EUR."""
    trading_commission_eur: Decimal = Decimal("0")
    fx_conversion_eur: Decimal = Decimal("0")
    stamp_duty_eur: Decimal = Decimal("0")
    financial_transaction_tax_eur: Decimal = Decimal("0")
    dividend_withholding_tax_eur: Decimal = Decimal("0")
    other_eur: Decimal = Decimal("0")

    def total(self) -> Decimal:
        return (
            self.trading_commission_eur
            + self.fx_conversion_eur
            + self.stamp_duty_eur
            + self.financial_transaction_tax_eur
            + self.dividend_withholding_tax_eur
            + self.other_eur
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "trading_commission_eur": str(self.trading_commission_eur),
            "fx_conversion_eur": str(self.fx_conversion_eur),
            "stamp_duty_eur": str(self.stamp_duty_eur),
            "financial_transaction_tax_eur": str(self.financial_transaction_tax_eur),
            "dividend_withholding_tax_eur": str(self.dividend_withholding_tax_eur),
            "other_eur": str(self.other_eur),
            "total_eur": str(self.total()),
        }

    def add_amount(self, category: FeeCategory, amount_eur: Decimal) -> None:
        if category == FeeCategory.TRADING_COMMISSION:
            self.trading_commission_eur += amount_eur
        elif category == FeeCategory.FX_CONVERSION:
            self.fx_conversion_eur += amount_eur
        elif category == FeeCategory.STAMP_DUTY:
            self.stamp_duty_eur += amount_eur
        elif category == FeeCategory.FINANCIAL_TRANSACTION_TAX:
            self.financial_transaction_tax_eur += amount_eur
        elif category == FeeCategory.DIVIDEND_WITHHOLDING_TAX:
            self.dividend_withholding_tax_eur += amount_eur
        else:
            self.other_eur += amount_eur


def categorize_fee(event: LedgerEvent) -> FeeCategory:
    """Map a ledger event to its fee category."""
    return EVENT_TYPE_TO_FEE_CATEGORY.get(event.event_type, FeeCategory.OTHER)


def categorize_fee_from_raw(raw_type: str, event_type: EventType) -> FeeCategory:
    """Categorize fee from raw string and event type (for unknown events)."""
    if event_type != EventType.UNKNOWN:
        return categorize_fee_raw(event_type)
    
    # Try to infer from raw string
    normalized = raw_type.lower().strip()
    if any(kw in normalized for kw in ["commission", "trading fee", "broker fee"]):
        return FeeCategory.TRADING_COMMISSION
    if any(kw in normalized for kw in ["fx", "currency conversion", "conversion fee", "foreign exchange"]):
        return FeeCategory.FX_CONVERSION
    if any(kw in normalized for kw in ["stamp duty", "sdrtax", "stamp tax"]):
        return FeeCategory.STAMP_DUTY
    if any(kw in normalized for kw in ["financial transaction tax", "ftt", "tobin tax"]):
        return FeeCategory.FINANCIAL_TRANSACTION_TAX
    if any(kw in normalized for kw in ["withholding tax", "dividend tax", "wht"]):
        return FeeCategory.DIVIDEND_WITHHOLDING_TAX
    return FeeCategory.OTHER


def categorize_fee_raw(event_type: EventType) -> FeeCategory:
    """Categorize fee from event type only."""
    return EVENT_TYPE_TO_FEE_CATEGORY.get(event_type, FeeCategory.OTHER)


def aggregate_fees(events: List[LedgerEvent]) -> FeeBreakdown:
    """Aggregate all fee events into a FeeBreakdown in EUR."""
    breakdown = FeeBreakdown()
    for event in events:
        if event.event_type in (
            EventType.COMMISSION,
            EventType.FX_CONVERSION_FEE,
            EventType.STAMP_DUTY,
            EventType.FINANCIAL_TRANSACTION_TAX,
            EventType.DIVIDEND_WITHHOLDING_TAX,
        ) or (event.event_type == EventType.UNKNOWN and (event.fee_amount > 0 or event.tax_amount > 0 or event.fx_fee_amount > 0)):
            category = categorize_fee(event)
            # Get the amount from the correct field based on event type
            if event.event_type == EventType.FX_CONVERSION_FEE:
                amount = event.fx_fee_amount
                amount_eur = event.fx_fee_amount_eur if event.fx_fee_amount_eur is not None else amount
            elif event.event_type in (EventType.STAMP_DUTY, EventType.FINANCIAL_TRANSACTION_TAX, EventType.DIVIDEND_WITHHOLDING_TAX):
                amount = event.tax_amount
                amount_eur = event.tax_amount_eur if event.tax_amount_eur is not None else amount
            elif event.event_type == EventType.COMMISSION:
                amount = event.fee_amount
                amount_eur = event.fee_amount_eur if event.fee_amount_eur is not None else amount
            else:
                amount = event.fee_amount + event.tax_amount + event.fx_fee_amount
                amount_eur = amount
            breakdown.add_amount(category, amount_eur)
    return breakdown


def aggregate_fees_from_matched_trades(matched_trades) -> FeeBreakdown:
    """Aggregate fees from matched trade results."""
    breakdown = FeeBreakdown()
    for trade in matched_trades:
        if trade.fees_eur > 0:
            # Fees from matched trades are already categorized during matching
            # This is a simplification – in practice fees are allocated to lots
            breakdown.other_eur += trade.fees_eur
    return breakdown