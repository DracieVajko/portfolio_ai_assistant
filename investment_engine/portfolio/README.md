# Pie-Aware Portfolio Exposure – Advisory Only

## CSV means pie composition metadata, not broker data

`PIEs/*.csv` files are **universe / metadata only**:

- `Slice` and `Name` columns define which instruments conceptually belong to a pie.
- `Invested value`, `Value`, `Result`, `Owned quantity`, `Dividends` columns are **ignored** – even when they contain zeros like `0` or `0.03`. They are placeholders from Trading 212 exports and never overwrite Trading 212 API values.
- `Owned quantity = 0` for every holding in `DailyDivPieLongShort.csv` does not mean the account holds zero; it means the CSV was exported before funding.
- Target weights are extracted only if a valid weight column (`Weight`, `Target weight`, `Allocation %`, etc.) exists and contains non-zero values that sum to 100% ±0.5%. If no weight column is present – as in the current `TechPieShort.csv` and others – weights are kept as `unknown` (`null`) and **never invented**. No fake rebalance recommendation is generated from unknown weights.

Source of truth for account calculations remains the Trading 212 API via `trading212_portfolio.parse_position` and `PortfolioMonitor.get_portfolio_summary`:
- `quantity`, `pieQuantity`, `averagePrice`, `currentPrice`, `ppl`, `fxPpl`, `value_eur`, `total_equity`, `cash`, `reconciliation_status`.

## Difference between exposure aggregation and execution constraints

### Exposure aggregation (risk view)

For every ticker the system builds a **canonical exposure**:

```
symbol: NVDA_US_EQ
pie_exposures: [{pie_id: TechPieShort, quantity: 0.60, value_eur: 99.58, source: api_pie_quantity_split}]
standalone_exposure: {quantity: 0.15, value_eur: 24.89}
total_quantity: 0.75  (= 0.60 + 0.15)
total_value_eur: 124.47
total_weight_of_account_pct: 1.24%  (= 124.47 / total_equity)
```

- Pie quantity comes from API `pieQuantity`; standalone quantity is `quantity - pieQuantity`. If a ticker belongs to multiple pies (e.g., `AAPL` in both `TechPieShort` and `DailyDivPieLongShort`), the pie quantity is split equally among those pies – no value is double-counted.
- Pie CSV only supplies *membership* (which pies contain the ticker) and optional `weight_in_pie`. It never supplies quantity/value.
- All tickers are the union of API-held symbols and pie-universe tickers. A ticker in the universe but not held appears with `total_quantity: 0` and a note `In pie universe but not currently held`.

This aggregated view is used for concentration checks: `total_weight_of_account_pct` is compared against the most restrictive `max_total_ticker_pct_of_account` among pies containing the ticker (e.g., `7.0%` for `TechPieShort`, `6.0%` for `DailyDiv`). `WARNING` is emitted when exceeded.

### Execution constraints (actionability view)

Aggregation and actionability are deliberately separate:

| Part | Allowed actions | Meaning |
|------|-----------------|---------|
| **Pie-held** | `HOLD_PIE`, `ADD_VIA_PIE`, `REDUCE_VIA_PIE`, `REVIEW_PIE` | The entire pie is adjusted – you cannot sell a single constituent inside a pie without changing the pie. |
| **Standalone** | `HOLD`, `BUY_STANDALONE`, `SELL_STANDALONE`, `WATCH` | A separately bought outside-the-pie position. |

- A pie constituent is **never** labelled simply `SELL`. If single-ticker risk inside a pie is detected (e.g., `NVDA` exceeds cap), the system outputs `REVIEW_PIE` with explanation: *“single-ticker risk inside pie cannot be sold individually – review pie as a whole.”*
- `can_add_via_pie` / `can_reduce_via_pie` are true when the ticker belongs to at least one pie. `can_buy_standalone` / `can_sell_standalone` are derived from the sidecar `allow_standalone_adds` / `allow_standalone_sells` of the pies containing the ticker (OR-logic across pies; standalone-only tickers default to true).
- `default_action_scope` (`pie`|`standalone`|`mixed`) comes from the sidecar; `TechPieShort` is `mixed`, `DailyDivPieLongShort` is `pie`.

## Standalone purchases create intentional drift

Buying 0.15 shares of `NVDA` outside `TechPieShort` while holding 0.60 inside the pie creates an **intentional target-weight drift**:

- The pie's internal target weights (if known) are untouched.
- Total exposure becomes `0.75` shares; standalone portion is noted as drift: *“Aggregated pie + standalone exposure; standalone creates intentional drift from pie target weights.”*
- A standalone action never implies `REBALANCE_PIE` or an automatic weight update. The drift is explicit and must be reviewed separately.

## All proposed actions are advisory only

- No function in `investment_engine/portfolio/*` imports `Trade212Client`, `requests.post` for orders, or any broker write API. Verified by tests.
- `evaluate_opportunistic_standalone_add` returns `OPPORTUNISTIC_STANDALONE_ADD` only when:
  1. Reconciliation is `PASS`,
  2. Signal is validated,
  3. `allow_standalone_adds` is true,
  4. `proposed_value / total_equity < max_standalone_add_pct_of_account`,
  5. `(total_value + proposed_value) / total_equity < max_total_ticker_pct_of_account`.
  Otherwise it returns `BLOCKED` with a reason. No order is placed; the result must be manually executed by the user in the Trading 212 app as a separate standalone buy.
- For `DailyDivPieLongShort` (`long_term`, `pie_only`), daily ticker-level technical signals are shown as informational only; `allow_standalone_adds` is `false`, so any daily `BUY` signal is downgraded to `REVIEW_PIE` or `WATCH`.
- Reports include the disclaimer: *“Advisory only. Pie-held actions are pie-level; standalone actions do not modify pie weights. No automatic orders.”*
