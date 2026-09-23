"""Canonical ledger event models for historical transaction accounting.

This module defines the event types and data structures for importing and
processing Trading 212 transaction/activity CSV exports. All amounts use
Decimal for precision. Events are immutable once created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Known event types from Trading 212 transaction exports."""
    TRADE_BUY = "trade_buy"
    TRADE_SELL = "trade_sell"
    DIVIDEND = "dividend"
    DIVIDEND_WITHHOLDING_TAX = "dividend_withholding_tax"
    FX_CONVERSION_FEE = "fx_conversion_fee"
    STAMP_DUTY = "stamp_duty"
    FINANCIAL_TRANSACTION_TAX = "financial_transaction_tax"
    COMMISSION = "commission"
    CASH_DEPOSIT = "cash_deposit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    CASH_INTEREST = "cash_interest"
    CORPORATE_ACTION = "corporate_action"
    ADJUSTMENT = "adjustment"
    UNKNOWN = "unknown"


class ParseStatus(str, Enum):
    """Parsing status for each event."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LedgerEvent:
    """Canonical ledger event – single source of truth for historical accounting.

    All monetary amounts are in original currency. EUR conversions are stored
    separately and calculated at import time using the event's exchange rate.
    """
    # Identity & source
    event_id: str
    timestamp: datetime
    source_file: str
    source_row: int
    source_type: str  # "t212_activity_csv" | "t212_cash_csv" | "manual_adjustment"

    # Instrument identification
    broker_symbol: str
    normalized_symbol: Optional[str] = None
    instrument_id: Optional[str] = None
    isin: Optional[str] = None

    # Event classification
    event_type: EventType = EventType.UNKNOWN
    raw_event_type: str = ""

    # Currency
    currency: str = "EUR"

    # Amounts in original currency
    gross_amount: Decimal = Decimal("0")
    fee_amount: Decimal = Decimal("0")
    tax_amount: Decimal = Decimal("0")
    fx_fee_amount: Decimal = Decimal("0")
    net_amount: Decimal = Decimal("0")

    # Trade-specific
    quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None

    # FX
    exchange_rate: Optional[Decimal] = None  # original_currency -> EUR

    # EUR-converted amounts (calculated at import)
    gross_amount_eur: Optional[Decimal] = None
    fee_amount_eur: Optional[Decimal] = None
    tax_amount_eur: Optional[Decimal] = None
    fx_fee_amount_eur: Optional[Decimal] = None
    net_amount_eur: Optional[Decimal] = None

    # Parsing metadata
    parse_status: ParseStatus = ParseStatus.OK
    parse_warning: Optional[str] = None

    # Raw row preserved for diagnostics (not serialized to user-facing reports)
    _raw_row: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """Serialize to dict. Raw row excluded by default for privacy."""
        d = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source_file": self.source_file,
            "source_row": self.source_row,
            "source_type": self.source_type,
            "broker_symbol": self.broker_symbol,
            "normalized_symbol": self.normalized_symbol,
            "instrument_id": self.instrument_id,
            "isin": self.isin,
            "event_type": self.event_type.value,
            "raw_event_type": self.raw_event_type,
            "currency": self.currency,
            "gross_amount": str(self.gross_amount),
            "fee_amount": str(self.fee_amount),
            "tax_amount": str(self.tax_amount),
            "fx_fee_amount": str(self.fx_fee_amount),
            "net_amount": str(self.net_amount),
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price is not None else None,
            "exchange_rate": str(self.exchange_rate) if self.exchange_rate is not None else None,
            "gross_amount_eur": str(self.gross_amount_eur) if self.gross_amount_eur is not None else None,
            "fee_amount_eur": str(self.fee_amount_eur) if self.fee_amount_eur is not None else None,
            "tax_amount_eur": str(self.tax_amount_eur) if self.tax_amount_eur is not None else None,
            "fx_fee_amount_eur": str(self.fx_fee_amount_eur) if self.fx_fee_amount_eur is not None else None,
            "net_amount_eur": str(self.net_amount_eur) if self.net_amount_eur is not None else None,
            "parse_status": self.parse_status.value,
            "parse_warning": self.parse_warning,
        }
        if include_raw:
            d["_raw_row"] = self._raw_row
        return d

    @property
    def is_trade(self) -> bool:
        return self.event_type in (EventType.TRADE_BUY, EventType.TRADE_SELL)

    @property
    def is_buy(self) -> bool:
        return self.event_type == EventType.TRADE_BUY

    @property
    def is_sell(self) -> bool:
        return self.event_type == EventType.TRADE_SELL

    @property
    def is_dividend(self) -> bool:
        return self.event_type in (EventType.DIVIDEND, EventType.DIVIDEND_WITHHOLDING_TAX)

    @property
    def is_fee_or_tax(self) -> bool:
        return self.event_type in (
            EventType.FX_CONVERSION_FEE,
            EventType.STAMP_DUTY,
            EventType.FINANCIAL_TRANSACTION_TAX,
            EventType.COMMISSION,
            EventType.DIVIDEND_WITHHOLDING_TAX,
        )

    @property
    def is_cash_flow(self) -> bool:
        return self.event_type in (
            EventType.CASH_DEPOSIT,
            EventType.CASH_WITHDRAWAL,
            EventType.CASH_INTEREST,
        )


@dataclass
class Lot:
    """Tax lot for FIFO matching."""
    symbol: str
    buy_event_id: str
    buy_timestamp: datetime
    quantity: Decimal
    remaining_quantity: Decimal
    cost_basis_per_share: Decimal  # in original currency
    cost_basis_eur_per_share: Optional[Decimal] = None
    currency: str = "EUR"
    fees_allocated: Decimal = Decimal("0")  # fees allocated to this lot
    fees_allocated_eur: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "buy_event_id": self.buy_event_id,
            "buy_timestamp": self.buy_timestamp.isoformat(),
            "quantity": str(self.quantity),
            "remaining_quantity": str(self.remaining_quantity),
            "cost_basis_per_share": str(self.cost_basis_per_share),
            "cost_basis_eur_per_share": str(self.cost_basis_eur_per_share) if self.cost_basis_eur_per_share is not None else None,
            "currency": self.currency,
            "fees_allocated": str(self.fees_allocated),
            "fees_allocated_eur": str(self.fees_allocated_eur) if self.fees_allocated_eur is not None else None,
        }


@dataclass
class MatchedTrade:
    """Result of matching a sell event to buy lots (FIFO)."""
    sell_event_id: str
    sell_timestamp: datetime
    symbol: str
    quantity_sold: Decimal
    sell_price: Decimal
    sell_price_eur: Optional[Decimal]
    proceeds_eur: Decimal
    matched_lots: List[Dict[str, Any]]  # list of {lot_id, quantity, cost_basis_eur, fees_eur}
    realized_pnl_eur: Decimal
    fees_eur: Decimal
    short_quantity: Decimal = Decimal("0")  # >0 if sell exceeds available lots

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sell_event_id": self.sell_event_id,
            "sell_timestamp": self.sell_timestamp.isoformat(),
            "symbol": self.symbol,
            "quantity_sold": str(self.quantity_sold),
            "sell_price": str(self.sell_price),
            "sell_price_eur": str(self.sell_price_eur) if self.sell_price_eur is not None else None,
            "proceeds_eur": str(self.proceeds_eur),
            "matched_lots": self.matched_lots,
            "realized_pnl_eur": str(self.realized_pnl_eur),
            "fees_eur": str(self.fees_eur),
            "short_quantity": str(self.short_quantity),
        }


@dataclass
class LedgerSummary:
    """Aggregated summary of the historical ledger."""
    # Realized P&L
    gross_realized_pnl_eur: Decimal = Decimal("0")
    realized_trading_fees_eur: Decimal = Decimal("0")
    fx_conversion_fees_eur: Decimal = Decimal("0")
    stamp_duty_eur: Decimal = Decimal("0")
    financial_transaction_tax_eur: Decimal = Decimal("0")
    commissions_eur: Decimal = Decimal("0")
    other_known_fees_eur: Decimal = Decimal("0")

    # Derived
    known_explicit_costs_eur: Decimal = Decimal("0")
    net_realized_pnl_after_known_costs_eur: Decimal = Decimal("0")

    # Dividends
    dividends_gross_eur: Decimal = Decimal("0")
    dividend_withholding_tax_eur: Decimal = Decimal("0")
    dividends_net_eur: Decimal = Decimal("0")

    # Cash flows
    cash_interest_eur: Decimal = Decimal("0")
    deposits_eur: Decimal = Decimal("0")
    withdrawals_eur: Decimal = Decimal("0")
    net_deposits_eur: Decimal = Decimal("0")

    # Open positions
    open_lot_cost_basis_eur: Decimal = Decimal("0")

    # Data quality
    unmatched_sell_count: int = 0
    unknown_event_count: int = 0
    accounting_status: str = "PARTIAL"  # RECONCILED | PARTIAL | UNAVAILABLE

    # Metadata
    first_event_date: Optional[datetime] = None
    last_event_date: Optional[datetime] = None
    total_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gross_realized_pnl_eur": str(self.gross_realized_pnl_eur),
            "realized_trading_fees_eur": str(self.realized_trading_fees_eur),
            "fx_conversion_fees_eur": str(self.fx_conversion_fees_eur),
            "stamp_duty_eur": str(self.stamp_duty_eur),
            "financial_transaction_tax_eur": str(self.financial_transaction_tax_eur),
            "commissions_eur": str(self.commissions_eur),
            "other_known_fees_eur": str(self.other_known_fees_eur),
            "known_explicit_costs_eur": str(self.known_explicit_costs_eur),
            "net_realized_pnl_after_known_costs_eur": str(self.net_realized_pnl_after_known_costs_eur),
            "dividends_gross_eur": str(self.dividends_gross_eur),
            "dividend_withholding_tax_eur": str(self.dividend_withholding_tax_eur),
            "dividends_net_eur": str(self.dividends_net_eur),
            "cash_interest_eur": str(self.cash_interest_eur),
            "deposits_eur": str(self.deposits_eur),
            "withdrawals_eur": str(self.withdrawals_eur),
            "net_deposits_eur": str(self.net_deposits_eur),
            "open_lot_cost_basis_eur": str(self.open_lot_cost_basis_eur),
            "unmatched_sell_count": self.unmatched_sell_count,
            "unknown_event_count": self.unknown_event_count,
            "accounting_status": self.accounting_status,
            "first_event_date": self.first_event_date.isoformat() if self.first_event_date else None,
            "last_event_date": self.last_event_date.isoformat() if self.last_event_date else None,
            "total_events": self.total_events,
        }


# Manual adjustment configuration example structure
@dataclass
class ManualAdjustments:
    """Optional manual adjustments for historical accounting gaps.
    
    This is a configuration template – users should copy to a local ignored
    file and populate with their actual data. Never committed to Git.
    """
    starting_cash_eur: Optional[Decimal] = None
    historical_fee_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    external_deposits: List[Dict[str, Any]] = field(default_factory=list)
    external_withdrawals: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "starting_cash_eur": str(self.starting_cash_eur) if self.starting_cash_eur is not None else None,
            "historical_fee_adjustments": self.historical_fee_adjustments,
            "external_deposits": self.external_deposits,
            "external_withdrawals": self.external_withdrawals,
            "notes": self.notes,
        }