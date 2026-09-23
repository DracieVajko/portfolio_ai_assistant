from __future__ import annotations

from typing import Mapping, Any


def score_asset(asset: Mapping[str, Any], portfolio_context: Mapping[str, Any] | None = None) -> float:
    """Deterministically score an asset for prioritization.

    The LLM should never be given every asset. Python ranks them first so only the
    highest-priority items reach the reasoning stages.
    """

    quantity = float(asset.get("quantity", 0) or 0)
    value = float(asset.get("value", 0) or 0)
    buy_probability = float(asset.get("buy_probability", 0) or 0)
    sell_probability = float(asset.get("sell_probability", 0) or 0)
    manual_priority = float(asset.get("manual_priority", 0) or 0)
    news_signal = float(asset.get("news_signal", 0) or 0)
    discovery_score = float(asset.get("discovery_score", 0) or 0)

    size_component = min(value / 10000.0, 25.0)
    position_component = min(quantity * 3.0, 20.0)
    buy_component = max(buy_probability - 50.0, 0.0) * 0.6
    sell_component = max(50.0 - sell_probability, 0.0) * 0.5
    manual_component = manual_priority * 3.0
    news_component = max(news_signal, 0.0) * 10.0
    discovery_component = discovery_score * 15.0

    score = size_component + position_component + buy_component + sell_component + manual_component + news_component + discovery_component

    if portfolio_context:
        holdings = portfolio_context.get("holdings") or []
        largest_positions = portfolio_context.get("largest_positions") or []
        symbols = {str(item.get("symbol") or item.get("broker_symbol") or "").upper() for item in holdings}
        largest_symbols = {str(item.get("symbol") or item.get("broker_symbol") or "").upper() for item in largest_positions}

        asset_symbol = str(asset.get("broker_symbol") or asset.get("name") or "").upper()
        if asset_symbol in symbols:
            score += 8.0
        if asset_symbol in largest_symbols:
            score += 5.0

    return round(score, 2)
