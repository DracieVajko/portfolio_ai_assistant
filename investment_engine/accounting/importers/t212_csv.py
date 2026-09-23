"""Trading 212 CSV importer – reads transaction/activity and cash transaction exports.

This module parses manually exported CSV files from Trading 212. It does NOT
make any API calls. All parsing is local and deterministic.

Supported CSV formats:
1. Transaction/Activity export (trades, dividends, fees, taxes, corporate actions)
2. Cash Transaction export (deposits, withdrawals, interest, FX conversions)

Column detection is flexible – matches by header name patterns rather than
fixed positions. Unknown rows are preserved as UNKNOWN events with warnings.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from investment_engine.accounting.models import (
    LedgerEvent,
    EventType,
    ParseStatus,
)


# Known T212 activity CSV column patterns (case-insensitive, flexible matching)
ACTIVITY_COLUMN_MAP = {
    "date_time": ["date/time", "datetime", "date time", "time", "date"],
    "action": ["action", "type", "transaction type", "operation"],
    "symbol": ["ticker", "symbol", "instrument", "name"],
    "isin": ["isin"],
    "quantity": ["qty", "quantity", "no. of shares", "shares", "amount"],
    "price": ["price", "price / share", "unit price", "avg price", "average price"],
    "currency": ["currency", "ccy"],
    "amount": ["amount", "total", "value", "result", "net amount"],
    "fee": ["fee", "commission", "fees", "trading fee"],
    "tax": ["tax", "stamp duty", "financial transaction tax", "ftt", "withholding tax"],
    "fx_fee": ["fx fee", "currency conversion fee", "fx conversion fee", "conversion fee"],
    "exchange_rate": ["exchange rate", "fx rate", "rate"],
    "notes": ["notes", "description", "comment", "details"],
}

# Action type mappings to our canonical EventType
ACTION_TYPE_MAP = {
    # Buys
    "buy": EventType.TRADE_BUY,
    "purchase": EventType.TRADE_BUY,
    "market buy": EventType.TRADE_BUY,
    "limit buy": EventType.TRADE_BUY,
    # Sells
    "sell": EventType.TRADE_SELL,
    "sale": EventType.TRADE_SELL,
    "market sell": EventType.TRADE_SELL,
    "limit sell": EventType.TRADE_SELL,
    # Dividends
    "dividend": EventType.DIVIDEND,
    "dividend (cash)": EventType.DIVIDEND,
    "dividend (reinvested)": EventType.DIVIDEND,
    "dividend withholding tax": EventType.DIVIDEND_WITHHOLDING_TAX,
    "withholding tax": EventType.DIVIDEND_WITHHOLDING_TAX,
    "tax - dividend": EventType.DIVIDEND_WITHHOLDING_TAX,
    # Fees & taxes
    "stamp duty": EventType.STAMP_DUTY,
    "stamp duty reserve tax": EventType.STAMP_DUTY,
    "financial transaction tax": EventType.FINANCIAL_TRANSACTION_TAX,
    "ftt": EventType.FINANCIAL_TRANSACTION_TAX,
    "commission": EventType.COMMISSION,
    "trading fee": EventType.COMMISSION,
    "currency conversion fee": EventType.FX_CONVERSION_FEE,
    "fx conversion fee": EventType.FX_CONVERSION_FEE,
    "conversion fee": EventType.FX_CONVERSION_FEE,
    # Cash flows (may appear in activity CSV)
    "cash deposit": EventType.CASH_DEPOSIT,
    "deposit": EventType.CASH_DEPOSIT,
    "cash withdrawal": EventType.CASH_WITHDRAWAL,
    "withdrawal": EventType.CASH_WITHDRAWAL,
    "withdraw": EventType.CASH_WITHDRAWAL,
    "interest": EventType.CASH_INTEREST,
    "cash interest": EventType.CASH_INTEREST,
    # Corporate actions
    "stock split": EventType.CORPORATE_ACTION,
    "reverse split": EventType.CORPORATE_ACTION,
    "merger": EventType.CORPORATE_ACTION,
    "spin-off": EventType.CORPORATE_ACTION,
    "spinoff": EventType.CORPORATE_ACTION,
    "rights issue": EventType.CORPORATE_ACTION,
    # Adjustments
    "adjustment": EventType.ADJUSTMENT,
    "correction": EventType.ADJUSTMENT,
    # Unknown fallback
}

# Cash transaction action mappings
CASH_ACTION_MAP = {
    "deposit": EventType.CASH_DEPOSIT,
    "withdrawal": EventType.CASH_WITHDRAWAL,
    "withdraw": EventType.CASH_WITHDRAWAL,
    "interest": EventType.CASH_INTEREST,
    "cash interest": EventType.CASH_INTEREST,
    "fx conversion": EventType.FX_CONVERSION_FEE,
    "currency conversion": EventType.FX_CONVERSION_FEE,
    "conversion": EventType.FX_CONVERSION_FEE,
}


def _normalize_header(header: str) -> str:
    """Normalize CSV header for matching."""
    return re.sub(r"[^a-z0-9]+", " ", header.strip().lower()).strip()


def _detect_columns(fieldnames: List[str]) -> Dict[str, Optional[str]]:
    """Map canonical column names to actual CSV fieldnames."""
    normalized_to_actual = {_normalize_header(fn): fn for fn in fieldnames}
    detected = {}
    for canonical, candidates in ACTIVITY_COLUMN_MAP.items():
        found = None
        for cand in candidates:
            norm_cand = _normalize_header(cand)
            if norm_cand in normalized_to_actual:
                found = normalized_to_actual[norm_cand]
                break
        detected[canonical] = found
    return detected


def _parse_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely parse a value to Decimal."""
    if value is None:
        return default
    s = str(value).strip()
    if s == "" or s.lower() in ("n/a", "nan", "null", "-"):
        return default
    # Remove currency symbols, commas, spaces
    s = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return default


def _parse_datetime(value: str) -> Optional[datetime]:
    """Parse datetime from various T212 formats."""
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Common formats: "2024-01-15 14:30:00", "2024-01-15T14:30:00Z", "15/01/2024 14:30"
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Try parsing just the date part
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    return None


def _map_action_to_event_type(action: str, is_cash_csv: bool = False) -> EventType:
    """Map raw action string to canonical EventType."""
    if not action:
        return EventType.UNKNOWN
    normalized = _normalize_header(action)
    action_map = CASH_ACTION_MAP if is_cash_csv else ACTION_TYPE_MAP
    for key, etype in action_map.items():
        if _normalize_header(key) == normalized:
            return etype
    # Fuzzy match for partial strings
    for key, etype in action_map.items():
        if _normalize_header(key) in normalized or normalized in _normalize_header(key):
            return etype
    return EventType.UNKNOWN


def _generate_event_id(source_file: str, row_num: int, prefix: str = "evt") -> str:
    """Generate a stable event ID."""
    import hashlib
    content = f"{source_file}:{row_num}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_suffix}"


def detect_csv_type(file_path: str | Path) -> str:
    """Detect whether CSV is activity/transactions or cash transactions."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        # Read first few lines
        sample = f.read(4096)
        f.seek(0)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        headers = next(reader, [])
    
    headers_lower = [_normalize_header(h) for h in headers]
    
    # Cash CSV typically has: Date/Time, Action, Currency, Amount, Balance, Reference
    # Activity CSV typically has: Date/Time, Action, Ticker, ISIN, Qty, Price, Currency, Amount, Fee, Tax
    cash_indicators = {"balance", "reference", "running balance"}
    activity_indicators = {"ticker", "isin", "qty", "quantity", "price", "fee", "tax", "stamp duty", "financial transaction tax"}
    
    cash_score = sum(1 for h in headers_lower if any(ci in h for ci in cash_indicators))
    activity_score = sum(1 for h in headers_lower if any(ai in h for ai in activity_indicators))
    
    if cash_score > activity_score:
        return "cash"
    return "activity"


def import_t212_activity_csv(file_path: str | Path) -> List[LedgerEvent]:
    """Import Trading 212 transaction/activity CSV.
    
    Returns list of LedgerEvent objects. Unknown/unparseable rows become
    EventType.UNKNOWN with parse_status=WARNING.
    """
    path = Path(file_path)
    events: List[LedgerEvent] = []
    
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        # Detect dialect
        sample = f.read(4096)
        f.seek(0)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            dialect = csv.excel
        
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            return events
        
        col_map = _detect_columns(reader.fieldnames)
        
        for row_num, row in enumerate(reader, start=2):  # 1-indexed, header is row 1
            if not any(v and str(v).strip() for v in row.values()):
                continue  # skip empty rows
            
            raw_action = row.get(col_map.get("action", ""), "") if col_map.get("action") else ""
            event_type = _map_action_to_event_type(raw_action, is_cash_csv=False)
            
            # Parse basic fields
            timestamp = _parse_datetime(row.get(col_map.get("date_time", ""), "") if col_map.get("date_time") else "")
            if timestamp is None:
                timestamp = datetime.now()
            
            broker_symbol = str(row.get(col_map.get("symbol", ""), "")).strip().upper() if col_map.get("symbol") else "UNKNOWN"
            isin = str(row.get(col_map.get("isin", ""), "")).strip().upper() if col_map.get("isin") else None
            
            quantity = _parse_decimal(row.get(col_map.get("quantity", ""), "") if col_map.get("quantity") else "")
            price = _parse_decimal(row.get(col_map.get("price", ""), "") if col_map.get("price") else "")
            price = price if price != 0 else None
            
            currency = str(row.get(col_map.get("currency", ""), "EUR")).strip().upper() if col_map.get("currency") else "EUR"
            
            # For FX conversion fee with empty currency, infer from exchange rate or default to USD
            if event_type == EventType.FX_CONVERSION_FEE and (not currency or currency == "EUR"):
                # Try to infer from exchange rate - if rate != 1.0, likely not EUR
                if exchange_rate and exchange_rate != Decimal("1.0"):
                    currency = "USD"  # Most common for FX fees
                else:
                    currency = "USD"
            
            # Amounts - for trades, Amount is gross; for taxes/fees, Tax column holds the amount
            gross_amount = _parse_decimal(row.get(col_map.get("amount", ""), "") if col_map.get("amount") else "")
            fee_amount = _parse_decimal(row.get(col_map.get("fee", ""), "") if col_map.get("fee") else "")
            tax_amount = _parse_decimal(row.get(col_map.get("tax", ""), "") if col_map.get("tax") else "")
            fx_fee_amount = _parse_decimal(row.get(col_map.get("fx_fee", ""), "") if col_map.get("fx_fee") else "")
            
            # For stamp duty, FTT, dividend WHT: the tax amount is in Tax column, not Amount
            # For these event types, treat Tax column as the primary amount
            if event_type in (EventType.STAMP_DUTY, EventType.FINANCIAL_TRANSACTION_TAX, EventType.DIVIDEND_WITHHOLDING_TAX):
                # Tax column contains the fee/tax amount (positive)
                # Amount column is typically 0 for these
                if tax_amount != 0 and gross_amount == 0:
                    gross_amount = tax_amount
                    tax_amount = Decimal("0")  # Will be re-categorized below
            
            # For FX conversion fee: the fee is in FX Fee column
            if event_type == EventType.FX_CONVERSION_FEE:
                if fx_fee_amount != 0 and gross_amount == 0:
                    gross_amount = fx_fee_amount
                    fx_fee_amount = Decimal("0")
            
            # Net amount (if not provided, calculate)
            # For tax/fee events, net = -gross (it's a cost)
            if event_type in (EventType.STAMP_DUTY, EventType.FINANCIAL_TRANSACTION_TAX, 
                             EventType.DIVIDEND_WITHHOLDING_TAX, EventType.FX_CONVERSION_FEE,
                             EventType.COMMISSION):
                net_amount = -gross_amount
            else:
                net_amount = gross_amount - fee_amount - tax_amount - fx_fee_amount
            
            exchange_rate = _parse_decimal(row.get(col_map.get("exchange_rate", ""), "") if col_map.get("exchange_rate") else "")
            exchange_rate = exchange_rate if exchange_rate != 0 else None
            
            # Determine parse status
            parse_status = ParseStatus.OK
            parse_warning = None
            if event_type == EventType.UNKNOWN and raw_action:
                parse_status = ParseStatus.WARNING
                parse_warning = f"Unrecognized action type: {raw_action}"
            if timestamp == datetime.now() and not row.get(col_map.get("date_time", ""), "").strip():
                parse_status = ParseStatus.WARNING
                parse_warning = (parse_warning + "; " if parse_warning else "") + "Missing timestamp"
            
            # For tax/fee events, set the correct amount fields
            final_fee_amount = fee_amount
            final_tax_amount = tax_amount
            final_fx_fee_amount = fx_fee_amount
            
            if event_type == EventType.STAMP_DUTY:
                final_tax_amount = gross_amount
                final_fee_amount = Decimal("0")
                gross_amount = Decimal("0")
            elif event_type == EventType.FINANCIAL_TRANSACTION_TAX:
                final_tax_amount = gross_amount
                final_fee_amount = Decimal("0")
                gross_amount = Decimal("0")
            elif event_type == EventType.DIVIDEND_WITHHOLDING_TAX:
                final_tax_amount = gross_amount
                final_fee_amount = Decimal("0")
                gross_amount = Decimal("0")
            elif event_type == EventType.FX_CONVERSION_FEE:
                final_fx_fee_amount = gross_amount
                final_fee_amount = Decimal("0")
                gross_amount = Decimal("0")
            elif event_type == EventType.COMMISSION:
                final_fee_amount = gross_amount
                gross_amount = Decimal("0")
            
            event = LedgerEvent(
                event_id=_generate_event_id(str(path), row_num, "act"),
                timestamp=timestamp,
                source_file=str(path),
                source_row=row_num,
                source_type="t212_activity_csv",
                broker_symbol=broker_symbol,
                normalized_symbol=None,  # filled later if needed
                instrument_id=None,
                isin=isin if isin and isin != "UNKNOWN" else None,
                event_type=event_type,
                raw_event_type=raw_action,
                currency=currency,
                gross_amount=gross_amount,
                fee_amount=final_fee_amount,
                tax_amount=final_tax_amount,
                fx_fee_amount=final_fx_fee_amount,
                net_amount=net_amount,
                quantity=quantity,
                price=price,
                exchange_rate=exchange_rate,
                parse_status=parse_status,
                parse_warning=parse_warning,
                _raw_row=dict(row),
            )
            events.append(event)
    
    return events


def import_t212_cash_csv(file_path: str | Path) -> List[LedgerEvent]:
    """Import Trading 212 cash transaction CSV.
    
    Returns list of LedgerEvent objects for cash flows (deposits, withdrawals,
    interest, FX conversions).
    """
    path = Path(file_path)
    events: List[LedgerEvent] = []
    
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            dialect = csv.excel
        
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            return events
        
        col_map = _detect_columns(reader.fieldnames)
        
        for row_num, row in enumerate(reader, start=2):
            if not any(v and str(v).strip() for v in row.values()):
                continue
            
            raw_action = row.get(col_map.get("action", ""), "") if col_map.get("action") else ""
            event_type = _map_action_to_event_type(raw_action, is_cash_csv=True)
            
            timestamp = _parse_datetime(row.get(col_map.get("date_time", ""), "") if col_map.get("date_time") else "")
            if timestamp is None:
                timestamp = datetime.now()
            
            # Cash CSV may not have symbol – use action as symbol for cash flows
            broker_symbol = str(row.get(col_map.get("symbol", ""), "")).strip().upper() if col_map.get("symbol") else "CASH"
            if broker_symbol == "CASH" or not broker_symbol:
                # Try to infer from action
                broker_symbol = raw_action.strip().upper() if raw_action else "CASH"
            
            currency = str(row.get(col_map.get("currency", ""), "EUR")).strip().upper() if col_map.get("currency") else "EUR"
            
            # For cash CSV, amount column is the primary amount
            gross_amount = _parse_decimal(row.get(col_map.get("amount", ""), "") if col_map.get("amount") else "")
            fee_amount = _parse_decimal(row.get(col_map.get("fee", ""), "") if col_map.get("fee") else "")
            fx_fee_amount = _parse_decimal(row.get(col_map.get("fx_fee", ""), "") if col_map.get("fx_fee") else "")
            
            # Net amount
            net_amount = gross_amount - fee_amount - fx_fee_amount
            
            exchange_rate = _parse_decimal(row.get(col_map.get("exchange_rate", ""), "") if col_map.get("exchange_rate") else "")
            exchange_rate = exchange_rate if exchange_rate != 0 else None
            
            parse_status = ParseStatus.OK
            parse_warning = None
            if event_type == EventType.UNKNOWN and raw_action:
                parse_status = ParseStatus.WARNING
                parse_warning = f"Unrecognized cash action type: {raw_action}"
            
            event = LedgerEvent(
                event_id=_generate_event_id(str(path), row_num, "csh"),
                timestamp=timestamp,
                source_file=str(path),
                source_row=row_num,
                source_type="t212_cash_csv",
                broker_symbol=broker_symbol,
                normalized_symbol=None,
                instrument_id=None,
                isin=None,
                event_type=event_type,
                raw_event_type=raw_action,
                currency=currency,
                gross_amount=gross_amount,
                fee_amount=fee_amount,
                tax_amount=Decimal("0"),
                fx_fee_amount=fx_fee_amount,
                net_amount=net_amount,
                quantity=Decimal("0"),
                price=None,
                exchange_rate=exchange_rate,
                parse_status=parse_status,
                parse_warning=parse_warning,
                _raw_row=dict(row),
            )
            events.append(event)
    
    return events