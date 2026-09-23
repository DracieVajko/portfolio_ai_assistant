from __future__ import annotations

from typing import Dict, List


_SECTOR_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "AI": {
        "queries": ["earnings guidance", "Blackwell", "CUDA", "hyperscaler demand", "analyst ratings"],
    },
    "Semiconductors": {
        "queries": ["earnings guidance", "capacity expansion", "customer demand", "supply chain", "analyst ratings"],
    },
    "GPU": {
        "queries": ["GPU demand", "product roadmap", "supply constraints", "partner wins", "analyst ratings"],
    },
    "Cloud": {
        "queries": ["cloud growth", "data center demand", "margin outlook", "customer spending", "analyst ratings"],
    },
    "Data Centers": {
        "queries": ["data center buildout", "power constraints", "customer backlog", "capital expenditure", "analyst ratings"],
    },
    "Battery": {
        "queries": ["battery demand", "production ramp", "raw material risk", "analyst ratings", "supply chain"],
    },
    "Lithium": {
        "queries": ["lithium pricing", "production guidance", "inventory levels", "analyst ratings", "supply constraints"],
    },
    "Renewables": {
        "queries": ["renewable demand", "grid demand", "policy support", "analyst ratings", "margin outlook"],
    },
    "Utilities": {
        "queries": ["rate case", "regulatory outlook", "earnings guidance", "analyst ratings", "capital plan"],
    },
    "Grid": {
        "queries": ["grid investment", "transmission backlog", "policy support", "analyst ratings", "earnings guidance"],
    },
    "Defense": {
        "queries": ["defense budget", "contract wins", "margin outlook", "analyst ratings", "earnings guidance"],
    },
    "Nuclear": {
        "queries": ["nuclear demand", "construction backlog", "regulatory progress", "analyst ratings", "earnings guidance"],
    },
    "Hydrogen": {
        "queries": ["hydrogen demand", "project pipeline", "policy support", "analyst ratings", "earnings guidance"],
    },
    "Dividend": {
        "queries": ["dividend sustainability", "cash flow", "balance sheet", "analyst ratings", "earnings guidance"],
    },
    "ETF": {
        "queries": ["ETF flows", "fundamental allocation", "market positioning", "analyst ratings", "yield outlook"],
    },
    "Crypto": {
        "queries": ["regulatory outlook", "ETF flows", "network adoption", "analyst ratings", "earnings guidance"],
    },
}


def build_dynamic_search_queries(symbol: str, sector: str) -> List[str]:
    """Create company-specific research queries based on sector templates."""
    sector_key = (sector or "AI").strip().title()
    base_queries = _SECTOR_TEMPLATES.get(sector_key, _SECTOR_TEMPLATES["AI"])["queries"]
    return [f"{symbol} {query}" for query in base_queries]
