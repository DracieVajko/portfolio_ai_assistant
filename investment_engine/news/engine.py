from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence


def _normalize_date(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    if "/" in text:
        try:
            return datetime.strptime(text, "%Y/%m/%d").date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text


def normalize_news(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and de-duplicate news items before they reach the LLM."""

    seen: set[tuple[str, str]] = set()
    normalized: List[Dict[str, Any]] = []

    for item in items:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        if any(keyword in title.lower() for keyword in ("sponsored", "advertisement", "promo")):
            continue

        key = (title.lower(), str(item.get("url") or "").lower())
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            {
                "title": title,
                "published": _normalize_date(item.get("published")),
                "url": str(item.get("url") or ""),
                "source": str(item.get("source") or "unknown"),
                "relevance_score": 0.0,
            }
        )

    # filter recent (<=2 days)
    now = datetime.utcnow()
    filtered: List[Dict[str, Any]] = []
    for n in normalized:
        if not n.get("published"):
            continue
        try:
            pub_date = datetime.fromisoformat(n["published"])
        except ValueError:
            continue
        if (now - pub_date).days <= 2:
            filtered.append(n)
    # group by ticker and keep most recent per stock
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for n in filtered:
        ticker = n.get("ticker", "UNKNOWN")
        grouped.setdefault(ticker, []).append(n)
    result: List[Dict[str, Any]] = []
    max_per_stock = 5
    for lst in grouped.values():
        sorted_lst = sorted(lst, key=lambda x: x["published"], reverse=True)
        result.extend(sorted_lst[:max_per_stock])
    return result
