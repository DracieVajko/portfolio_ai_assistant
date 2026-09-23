# Historical Transaction Ledger – Read-Only Accounting Layer

## Overview

This module imports manually exported Trading 212 transaction/activity CSV files and calculates historical performance **without corrupting or replacing live account reconciliation**.

**Key principles:**
- **Read-only**: Never calls Trading 212 API or any external service
- **Source priority**: Actual CSV → Manual adjustments → Broker API (if verified)
- **Decimal precision**: All monetary calculations use `Decimal`
- **Timezone-aware**: All timestamps are timezone-aware
- **Unknown events**: Force `PARTIAL` status unless explicitly ignored
- **Privacy**: Raw CSV rows preserved locally only, never in user-facing reports

This module is **independent** from the live portfolio reconciliation in `trading212_portfolio.py` and `investment_engine/portfolio/`.

## Data Sources (Priority Order)

1. **Actual imported Trading 212 transaction/activity CSV** – Primary source
2. **Actual imported Trading 212 cash transaction CSV** – Supplements deposits/withdrawals/interest
3. **Manual structured adjustments file** – For known gaps (starting cash, external transfers)
4. **Broker API history** – Only if read-only endpoint exists and verified with redacted fixture
5. **Current online fee schedule** – Only for forward-looking simulations, **never** historic accounting

## Event Types

Canonical ledger events:
- `TradeBuy` / `TradeSell` – Executed trades
- `Dividend` / `DividendWithholdingTax` – Dividends and tax
- `FxConversionFee` – Currency conversion fees
- `StampDuty` – UK stamp duty
- `FinancialTransactionTax` – French FTT, etc.
- `Commission` – Trading commissions
- `CashDeposit` / `CashWithdrawal` / `CashInterest` – Cash flows
- `CorporateAction` – Splits, mergers, spin-offs
- `Adjustment` – Corrections
- `UnknownEvent` – Unrecognized rows (forces PARTIAL status)

## Lot Matching (FIFO)

- **Default**: First-In-First-Out (FIFO) – only supported method in v1
- Sell events matched only to **earlier** buy lots for same symbol
- Partial lot closes preserved
- Short/negative inventory → reconciliation failure, not normal result
- Corporate actions **never** silently treated as buys/sells

## Calculated Metrics

| Metric | Description |
|--------|-------------|
| `gross_realized_pnl_eur` | Realized P&L before explicit costs |
| `realized_trading_fees_eur` | Commissions |
| `fx_conversion_fees_eur` | Currency conversion fees |
| `stamp_duty_eur` | Stamp duty (UK) |
| `financial_transaction_tax_eur` | FTT (France, etc.) |
| `commissions_eur` | Trading commissions |
| `other_known_fees_eur` | Uncategorized fees |
| `known_explicit_costs_eur` | Sum of all above |
| `net_realized_pnl_after_known_costs_eur` | Gross P&L - known costs |
| `dividends_gross_eur` | Gross dividends received |
| `dividend_withholding_tax_eur` | WHT on dividends |
| `dividends_net_eur` | Net dividends |
| `cash_interest_eur` | Interest on cash |
| `deposits_eur` / `withdrawals_eur` / `net_deposits_eur` | Cash flows |
| `open_lot_cost_basis_eur` | Cost basis of remaining positions |
| `unmatched_sell_count` | Sells without matching buys |
| `unknown_event_count` | Unclassified events |
| `accounting_status` | `RECONCILED` \| `PARTIAL` \| `UNAVAILABLE` |

## Required Report Wording

**Do not call** `net_realized_pnl_after_known_costs_eur` "account total return".

- If all necessary events present:
  > "Net realized P/L after known explicit costs."
- If unknown/missing events exist:
  > "Partial historical ledger result — excludes unavailable or unclassified events."
- Account-wide return remains **broker-authoritative** and is **not overwritten**.

## Historical Reconciliation

- Compares ledger net deposits with manual broker net-deposit summary (if provided)
- Compares ledger open lots with live position quantities (diagnostic only)
- Compares ledger performance with broker account return (diagnostic only)
- **Never** forces matching by inserting unexplained adjustments
- Returns explicit differences and explains unavailable dimensions

## Usage

```python
from investment_engine.accounting import build_ledger, ManualAdjustments
from decimal import Decimal

# Build ledger from exported CSVs
ledger = build_ledger(
    activity_csv_path="exports/t212_activity_2024.csv",
    cash_csv_path="exports/t212_cash_2024.csv",
    default_fx_rates={"USD": Decimal("0.922"), "GBP": Decimal("1.183")},
)

# Get summary for reporting
summary = ledger.get_summary_report()
print(summary)
```

## Manual Adjustments

Create a local file (e.g., `accounting/manual_adjustments.json`) from the example template. **Never commit to Git.**

```json
{
  "starting_cash_eur": "10000.00",
  "historical_fee_adjustments": [],
  "external_deposits": [{"date": "2023-01-15", "amount_eur": "5000", "source": "bank_transfer"}],
  "external_withdrawals": [],
  "notes": "Starting cash from bank statement Jan 2023"
}
```

## Safety

- No network calls
- No API credentials required
- No broker write operations
- No LLM involvement in calculations
- Output paths controlled
- Secrets masked in any serialized output

## Tests

Run with:
```bash
pytest tests/test_transaction_ledger.py tests/test_transaction_import.py tests/test_fee_accounting.py -v
```

Fixtures in `tests/fixtures/` use **sanitized synthetic data only** – no real portfolio numbers.