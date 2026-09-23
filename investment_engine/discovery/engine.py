from __future__ import annotations

from typing import Any, Dict, List

try:
    from investment_engine.research.news_engine import validate_idea_with_news
except Exception:
    validate_idea_with_news = None  # type: ignore[assignment]

__all__ = ["DiscoveryEngine", "validate_idea_with_news"]


def _norm_ticker(value):
    return str(value or "").strip().upper()


def _suggestions_from_assets(assets):
    symbols = {_norm_ticker(item.get("broker_symbol") or item.get("name") or "") for item in (assets or [])}
    symbols.discard("")
    mapping = {
        "NVDA": ["AVGO", "MRVL", "MU", "SKHYNIX", "TSM", "ASML", "VRT", "DELL", "AMD", "PLTR"],
        "AAPL": ["AVGO", "QCOM", "AMD", "MSFT", "NFLX", "INTC"],
        "MSFT": ["NVDA", "AMD", "AVGO", "CRM", "ORCL"],
        "TSLA": ["F", "RIVN", "LCID", "NIO"],
    }
    suggestions = []
    for symbol in sorted(symbols):
        for suggestion in mapping.get(symbol, []):
            if suggestion not in symbols:
                suggestions.append({"broker_symbol": suggestion, "reason": f"ecosystem for {symbol}", "discovery_score": 0.8})
    return suggestions


def _watchlist_tickers(watchlist):
    tickers = []
    for entry in watchlist or []:
        if isinstance(entry, dict):
            t = _norm_ticker(entry.get("broker_symbol") or entry.get("ticker") or entry.get("symbol") or entry.get("name"))
        else:
            t = _norm_ticker(entry)
        if t and t != "UNKNOWN" and t not in tickers:
            tickers.append(t)
    return tickers


class DiscoveryEngine:
    """Generate discovery ideas from portfolio context rather than random assets."""

    def __init__(self, portfolio_context=None):
        self.portfolio_context = portfolio_context or {}

    def discover(self, assets, watchlist=None):
        """Static ecosystem suggestions (backward compatible).

        When watchlist tickers are supplied they are appended as ideas
        (deduplicated) so callers can validate them with news separately.
        """
        suggestions = _suggestions_from_assets(assets)
        if watchlist:
            known = {_norm_ticker(s.get("broker_symbol", "")) for s in suggestions}
            known |= {_norm_ticker(item.get("broker_symbol") or item.get("name") or "") for item in (assets or [])}
            for t in _watchlist_tickers(watchlist):
                if t not in known:
                    suggestions.append({"broker_symbol": t, "reason": "watchlist idea", "discovery_score": 0.7})
                    known.add(t)
        return suggestions
    def discover_validated(
        self,
        assets=None,
        watchlist=None,
        suggestions=None,
        *,
        fetcher=None,
        limit_per_idea=3,
        min_relevance=30,
        max_age_hours=72,
        max_total=20,
    ):
        """Validate suggestions + watchlist ideas against recent news.

        For each idea calls fetch_for_symbol (via validate_idea_with_news,
        limit 2-3), keeps relevance >= threshold and age <= 72h, caps output
        at ~20 total items. Evidence matching uses _record_links style
        (symbol-key + company-name), not a literal TICKER-substring scan.
        Returns validated idea dicts with evidence + url + confidence.
        """
        if validate_idea_with_news is None:
            return []
        base = list(suggestions) if suggestions else _suggestions_from_assets(assets or [])
        seen = set()
        candidates = []
        for s in base:
            t = _norm_ticker((s or {}).get("broker_symbol") or (s or {}).get("ticker"))
            if t and t not in seen:
                seen.add(t)
                candidates.append((t, str((s or {}).get("reason", "ecosystem idea"))))
        for t in _watchlist_tickers(watchlist):
            if t not in seen:
                seen.add(t)
                candidates.append((t, "watchlist idea"))
        try:
            limit_per_idea = max(2, min(int(limit_per_idea or 3), 3))
        except (TypeError, ValueError):
            limit_per_idea = 3
        validated = []
        total_items = 0
        for ticker, reason in candidates:
            if len(validated) >= max_total or total_items >= max_total:
                break
            try:
                res = validate_idea_with_news(
                    ticker, ticker,
                    fetcher=fetcher, limit=limit_per_idea,
                    min_relevance=min_relevance, max_age_hours=max_age_hours,
                )
            except Exception:
                continue
            if not isinstance(res, dict) or not res.get("supported"):
                continue
            items = res.get("items", []) or []
            if total_items + len(items) > max_total and validated:
                room = max_total - total_items
                if room <= 0:
                    break
                items = items[:room]
                res = dict(res)
                res["items"] = items
                res["evidence"] = (res.get("evidence", []) or [])[:room]
                res["urls"] = (res.get("urls", []) or [])[:room]
            total_items += len(items)
            entry = dict(res)
            entry["reason"] = reason
            validated.append(entry)
        return validated