"""FIFO lot matching for historical trade accounting.

Implements First-In-First-Out (FIFO) cost basis tracking for sell events.
Matches each sell to the earliest available buy lots for the same symbol.

Key rules:
- Only matches sell events to earlier buy events
- Preserves partial lot closes
- Flags short/negative inventory as reconciliation failure
- Corporate actions are NOT silently treated as buys/sells
- All amounts in EUR for P&L calculation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from investment_engine.accounting.models import (
    LedgerEvent,
    Lot,
    MatchedTrade,
    EventType,
)


@dataclass
class MatchingResult:
    """Result of FIFO matching for all symbols."""
    matched_trades: List[MatchedTrade] = field(default_factory=list)
    open_lots: Dict[str, List[Lot]] = field(default_factory=dict)  # symbol -> lots
    unmatched_sells: List[LedgerEvent] = field(default_factory=list)
    short_sells: List[MatchedTrade] = field(default_factory=list)  # sells exceeding available lots
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_trades": [t.to_dict() for t in self.matched_trades],
            "open_lots": {sym: [l.to_dict() for l in lots] for sym, lots in self.open_lots.items()},
            "unmatched_sells": [e.to_dict() for e in self.unmatched_sells],
            "short_sells": [t.to_dict() for t in self.short_sells],
            "warnings": self.warnings,
        }


def _convert_to_eur(amount: Decimal, currency: str, exchange_rate: Optional[Decimal],
                    default_fx_rates: Dict[str, Decimal]) -> Decimal:
    """Convert amount to EUR using event's exchange rate or default rates."""
    if currency == "EUR":
        return amount
    if exchange_rate is not None and exchange_rate != 0:
        return amount * exchange_rate
    # Fallback to default rates
    rate = default_fx_rates.get(currency)
    if rate is not None:
        return amount * rate
    # No rate available – return original (caller should warn)
    return amount


def match_sell_fifo(
    events: List[LedgerEvent],
    default_fx_rates: Optional[Dict[str, Decimal]] = None,
) -> MatchingResult:
    """Match all sell events to buy lots using FIFO.
    
    Args:
        events: All ledger events sorted by timestamp
        default_fx_rates: Fallback FX rates {currency: rate_to_EUR}
    
    Returns:
        MatchingResult with matched trades, open lots, and any issues.
    """
    if default_fx_rates is None:
        default_fx_rates = {
            Decimal("GBP"): Decimal("1.183"),
            Decimal("USD"): Decimal("0.922"),
            Decimal("EUR"): Decimal("1.0"),
        }
    # Ensure keys are strings for lookup
    default_fx_rates_str = {str(k): v for k, v in default_fx_rates.items()}
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda e: (e.timestamp, e.source_row))
    
    # Track open lots per symbol
    open_lots: Dict[str, List[Lot]] = {}
    matched_trades: List[MatchedTrade] = []
    unmatched_sells: List[LedgerEvent] = []
    short_sells: List[MatchedTrade] = []
    warnings: List[str] = []
    
    for event in sorted_events:
        symbol = event.normalized_symbol or event.broker_symbol
        if not symbol or symbol == "UNKNOWN":
            if event.is_trade:
                warnings.append(f"Event {event.event_id}: Unknown symbol, cannot match")
            continue
        
        if event.is_buy:
            # Create new lot from buy event
            cost_basis_per_share = event.price if event.price and event.price != 0 else Decimal("0")
            if event.quantity > 0 and cost_basis_per_share > 0:
                # Calculate cost basis in EUR per share
                cost_basis_eur_per_share = None
                if event.exchange_rate and event.exchange_rate != 0:
                    cost_basis_eur_per_share = cost_basis_per_share * event.exchange_rate
                elif event.currency in default_fx_rates_str:
                    cost_basis_eur_per_share = cost_basis_per_share * default_fx_rates_str[event.currency]
                
                # Allocate fees to this lot proportionally
                fees_allocated = event.fee_amount + event.tax_amount + event.fx_fee_amount
                fees_allocated_eur = None
                if fees_allocated > 0:
                    if event.exchange_rate and event.exchange_rate != 0:
                        fees_allocated_eur = fees_allocated * event.exchange_rate
                    elif event.currency in default_fx_rates_str:
                        fees_allocated_eur = fees_allocated * default_fx_rates_str[event.currency]
                
                lot = Lot(
                    symbol=symbol,
                    buy_event_id=event.event_id,
                    buy_timestamp=event.timestamp,
                    quantity=event.quantity,
                    remaining_quantity=event.quantity,
                    cost_basis_per_share=cost_basis_per_share,
                    cost_basis_eur_per_share=cost_basis_eur_per_share,
                    currency=event.currency,
                    fees_allocated=fees_allocated,
                    fees_allocated_eur=fees_allocated_eur,
                )
                open_lots.setdefault(symbol, []).append(lot)
        
        elif event.is_sell:
            # Match sell to earliest buy lots (FIFO)
            symbol_lots = open_lots.get(symbol, [])
            if not symbol_lots:
                # No lots available – unmatched sell
                unmatched_sells.append(event)
                warnings.append(f"Sell event {event.event_id} ({symbol}): No buy lots available for matching")
                continue
            
            quantity_to_sell = event.quantity
            if quantity_to_sell <= 0:
                warnings.append(f"Sell event {event.event_id} ({symbol}): Zero or negative quantity")
                continue
            
            sell_price = event.price if event.price and event.price != 0 else Decimal("0")
            sell_price_eur = None
            if sell_price > 0:
                if event.exchange_rate and event.exchange_rate != 0:
                    sell_price_eur = sell_price * event.exchange_rate
                elif event.currency in default_fx_rates_str:
                    sell_price_eur = sell_price * default_fx_rates_str[event.currency]
            
            matched_lots_details = []
            total_cost_basis_eur = Decimal("0")
            total_fees_eur = Decimal("0")
            remaining_to_sell = quantity_to_sell
            
            # Iterate through lots in FIFO order
            lots_to_remove = []
            for i, lot in enumerate(symbol_lots):
                if remaining_to_sell <= 0:
                    break
                
                available = lot.remaining_quantity
                if available <= 0:
                    lots_to_remove.append(i)
                    continue
                
                qty_from_lot = min(available, remaining_to_sell)
                remaining_to_sell -= qty_from_lot
                lot.remaining_quantity -= qty_from_lot
                
                # Cost basis for this portion
                if lot.cost_basis_eur_per_share is not None:
                    cost_basis_eur = lot.cost_basis_eur_per_share * qty_from_lot
                else:
                    # Fallback: use original currency cost basis
                    cost_basis_eur = lot.cost_basis_per_share * qty_from_lot
                    if lot.currency in default_fx_rates_str:
                        cost_basis_eur *= default_fx_rates_str[lot.currency]
                
                # Fees allocated proportionally
                if lot.fees_allocated_eur is not None and lot.quantity > 0:
                    fee_proportion = qty_from_lot / lot.quantity
                    fees_eur = lot.fees_allocated_eur * fee_proportion
                else:
                    fees_eur = Decimal("0")
                
                total_cost_basis_eur += cost_basis_eur
                total_fees_eur += fees_eur
                
                matched_lots_details.append({
                    "lot_id": lot.buy_event_id,
                    "quantity": str(qty_from_lot),
                    "cost_basis_eur": str(cost_basis_eur),
                    "fees_eur": str(fees_eur),
                    "buy_timestamp": lot.buy_timestamp.isoformat(),
                })
                
                if lot.remaining_quantity <= Decimal("1e-12"):  # effectively zero
                    lots_to_remove.append(i)
            
            # Remove exhausted lots (in reverse order to preserve indices)
            for i in reversed(lots_to_remove):
                symbol_lots.pop(i)
            
            # Calculate proceeds in EUR
            if sell_price_eur is not None:
                proceeds_eur = sell_price_eur * quantity_to_sell
            else:
                # Fallback: use net amount from event
                proceeds_eur = _convert_to_eur(event.net_amount, event.currency, event.exchange_rate, default_fx_rates_str)
            
            realized_pnl_eur = proceeds_eur - total_cost_basis_eur - total_fees_eur
            
            matched_trade = MatchedTrade(
                sell_event_id=event.event_id,
                sell_timestamp=event.timestamp,
                symbol=symbol,
                quantity_sold=quantity_to_sell,
                sell_price=sell_price,
                sell_price_eur=sell_price_eur,
                proceeds_eur=proceeds_eur,
                matched_lots=matched_lots_details,
                realized_pnl_eur=realized_pnl_eur,
                fees_eur=total_fees_eur,
                short_quantity=remaining_to_sell if remaining_to_sell > 0 else Decimal("0"),
            )
            
            if remaining_to_sell > 0:
                # Short sell – sell exceeded available lots
                matched_trade.short_quantity = remaining_to_sell
                short_sells.append(matched_trade)
                warnings.append(
                    f"Sell event {event.event_id} ({symbol}): Short sell of {remaining_to_sell} shares "
                    f"(only {quantity_to_sell - remaining_to_sell} matched to lots)"
                )
            else:
                matched_trades.append(matched_trade)
    
    # Filter out empty lots
    for symbol, lots in list(open_lots.items()):
        open_lots[symbol] = [l for l in lots if l.remaining_quantity > Decimal("1e-12")]
        if not open_lots[symbol]:
            del open_lots[symbol]
    
    return MatchingResult(
        matched_trades=matched_trades,
        open_lots=open_lots,
        unmatched_sells=unmatched_sells,
        short_sells=short_sells,
        warnings=warnings,
    )


def calculate_open_lot_cost_basis(open_lots: Dict[str, List[Lot]]) -> Decimal:
    """Calculate total cost basis of all open lots in EUR."""
    total = Decimal("0")
    for lots in open_lots.values():
        for lot in lots:
            if lot.cost_basis_eur_per_share is not None:
                total += lot.cost_basis_eur_per_share * lot.remaining_quantity
            elif lot.currency in {"EUR", "GBP", "USD"}:  # fallback
                # This shouldn't happen if matching worked correctly
                pass
    return total