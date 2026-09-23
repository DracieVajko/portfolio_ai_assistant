"""
Trading 212 API helper functions.

This module provides lightweight wrappers around the Trading 212 REST API.
The real endpoints and authentication details are not exposed here –
the user must supply a valid bearer token via `trading212_auth.py`.

Functions:
- get_price_stats(ticker, token)
- get_candles(ticker, interval_minutes, token)
- average_purchase_price(ticker, portfolio_data)
"""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Base URL placeholder – replace with the real Trading 212 API endpoint.
API_BASE = "https://api.trading212.com"


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def get_price_stats(ticker: str, token: str) -> Dict[str, Any]:
    """Return current price and min/max for 24h, 7d, 1y.

    Expected response structure (simplified):
        {
            "current_price": float,
            "history": [ {"timestamp": str, "price": float}, ... ]
        }
    """
    url = f"{API_BASE}/price/{ticker}"
    resp = requests.get(url, headers=_auth_headers(token))
    data = resp.json()

    now = datetime.utcnow()
    hist = [h for h in data.get("history", []) if datetime.fromisoformat(h["timestamp"]) >= now - timedelta(days=365)]
    prices = [h["price"] for h in hist]

    return {
        "current": data.get("current_price"),
        "24h_min": min(prices[-24 * 60:]) if len(prices) >= 24 * 60 else None,
        "24h_max": max(prices[-24 * 60:]) if len(prices) >= 24 * 60 else None,
        "7d_min": min(prices[-7 * 24 * 60:]) if len(prices) >= 7 * 24 * 60 else None,
        "7d_max": max(prices[-7 * 24 * 60:]) if len(prices) >= 7 * 24 * 60 else None,
        "1y_min": min(prices) if prices else None,
        "1y_max": max(prices) if prices else None,
    }


def get_candles(ticker: str, interval_minutes: int, token: str) -> List[Dict[str, Any]]:
    """Return candle data for the last 24h or week.

    Parameters:
        ticker: Stock symbol.
        interval_minutes: Candle interval in minutes (e.g., 15, 30, 60).
        token: Bearer token.
    """
    url = f"{API_BASE}/candles/{ticker}"
    params = {"interval": f"{interval_minutes}m", "limit": 1000}
    resp = requests.get(url, headers=_auth_headers(token), params=params)
    return resp.json()


def average_purchase_price(ticker: str, portfolio_data: Dict[str, Any]) -> float | None:
    """Compute the average purchase price from holdings.

    Expected portfolio structure (simplified):
        {
            "holdings": {
                "AAPL": {"quantity": 10, "total_cost": 1500.0},
                ...
            }
        }
    """
    holding = portfolio_data.get("holdings", {}).get(ticker)
    if not holding:
        return None
    qty = holding.get("quantity", 0)
    total_cost = holding.get("total_cost", 0.0)
    return (total_cost / qty) if qty else None
