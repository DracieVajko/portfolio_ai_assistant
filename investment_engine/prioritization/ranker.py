from __future__ import annotations

from typing import Any, Dict, List

from investment_engine.scoring.priority import score_asset


def rank_assets(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return assets ordered by deterministic priority score."""
    scored = []
    for asset in assets:
        scored.append({**asset, "priority_score": score_asset(asset)})
    return sorted(scored, key=lambda item: item.get("priority_score", 0), reverse=True)
