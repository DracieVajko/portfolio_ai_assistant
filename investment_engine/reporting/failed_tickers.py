"""Failed-ticker ledger: unresolvable Yahoo symbols with copy-paste config.

Deterministic and network-free. Consumed by run_engine() (collection) and
portfolio_ai_assistant.py wrapper (publish). Mirrors the V4 failed_tickers.md
report so coverage gaps become visible instead of silent debug skips.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def collect_failed_tickers(
    yahoo_by_display: dict[str, str | None],
    standalone: list[dict] | None = None,
    fetch_failed: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the failed-ticker ledger for one run.

    Classification rules:
    - HELD_UNRESOLVED: nonzero broker qty/value AND has T212 instrument ID.
      Truly unresolvable holding requiring broker data fallback.
    - WATCHLIST: no broker holding (qty=0, value=0), only configured/research.
      Not described as an unresolved holding.
    - STALE_ALIAS: neither held nor actively watched; stale config entry.
    Zero-quantity, zero-value watchlist/config items are NEVER described
    as unresolved holdings.
    """
    try:
        from investment_engine.portfolio.symbols import support_state
    except Exception:
        return []
    try:
        by_display: dict[str, dict] = {}
        for pos in standalone or []:
            if isinstance(pos, dict) and pos.get("display"):
                by_display[str(pos["display"]).strip().upper()] = pos
        failed_set = set(fetch_failed or set())
        out: list[dict[str, Any]] = []
        for display, yahoo in (yahoo_by_display or {}).items():
            disp = str(display).strip().upper()
            try:
                state = support_state(yahoo, "market_data")
            except Exception:
                state = "UNRESOLVED"
            if state != "UNRESOLVED" and (yahoo or "") not in failed_set:
                continue
            pos = by_display.get(disp, {})
            qty = pos.get("qty", 0) or 0
            value_eur = pos.get("value_eur", 0) or 0
            t212_id = pos.get("t212", "") or ""
            has_broker_holding = qty != 0 and value_eur != 0 and bool(t212_id)
            # Classify the item type.
            if has_broker_holding:
                category = "HELD_UNRESOLVED"
            elif qty == 0 and value_eur == 0:
                # Watchlist-only or stale config alias: not a holding.
                category = "WATCHLIST" if t212_id == "" else "STALE_ALIAS"
            else:
                category = "WATCHLIST"
            # Skip watchlist-only and stale alias items from the failed ledger
            # (they are not unresolved holdings).
            if category != "HELD_UNRESOLVED":
                continue
            out.append({
                "display": disp,
                "t212_id": t212_id,
                "yahoo_attempted": yahoo or "",
                "support_state": state,
                "fetch_failed": (yahoo or "") in failed_set,
                "qty": qty,
                "value_eur": value_eur,
                "alias_hint": {disp: "<YahooSymbol>"},
                "category": category,
            })
        out.sort(key=lambda e: (-(e.get("value_eur") or 0), e.get("display") or ""))
        return out
    except Exception as exc:  # never break the run over the ledger itself
        logger.debug("collect_failed_tickers failed: %s", type(exc).__name__)
        return []


def render_failed_tickers_md(failed: list[dict[str, Any]], run_id: str = "") -> str:
    """Render failed_tickers.md with ready-to-copy symbol_aliases JSON."""
    if not failed:
        return ""
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    held_failed = [e for e in failed if e.get("category") == "HELD_UNRESOLVED"]
    lines = [
        "# Failed Ticker Resolution",
        f"Run: `{run_id}` | {ts}",
        "",
        f"Total unresolvable holdings: **{len(held_failed)}**",
        "",
        "These HELD positions have no usable Yahoo Finance symbol: mapping is UNRESOLVED",
        "or the market-data fetch already failed this run. They are valued from",
        "broker data only — no technicals, no news, no AI signal.",
        "",
        "## Unresolvable holdings",
        "",
        "| Display | T212 ID | Attempted Yahoo | State | Fetch failed | Qty | Value € | Category |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in held_failed:
        lines.append(
            f"| {e.get('display', '?')} | {e.get('t212_id', '') or '—'} | "
            f"{e.get('yahoo_attempted', '') or '—'} | {e.get('support_state', '?')} | "
            f"{'yes' if e.get('fetch_failed') else 'no'} | {e.get('qty', 0)} | "
            f"{e.get('value_eur', 0):,.2f} | {e.get('category', '?')} |"
        )
    lines += [
        "",
        "## Fix: add to `symbol_aliases` in portfolio_config.json",
        "",
        "Find the correct Yahoo symbol (finance.yahoo.com → quote URL), then add:",
        "",
        "```json",
        '"symbol_aliases": {',
    ]
    import json as _json
    hints: dict[str, str] = {}
    for e in failed:
        hints[str(e.get("display", "?"))] = "<YAHOO_SYMBOL>"
    lines.append(_json.dumps({"symbol_aliases": hints}, indent=2, ensure_ascii=False))
    lines += [
        "```",
        "",
        "_Takes effect on the next run. Entries here beat the built-in table._",
        "",
    ]
    return "\n".join(lines)
