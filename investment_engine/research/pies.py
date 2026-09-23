from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


class PieLoader:
    """Load Trading212 PIE composition CSV files and expose them as structured portfolio context."""

    def __init__(self, pies_dir: str | None = None) -> None:
        self.pies_dir = Path(pies_dir or "PIEs")

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.pies_dir.exists():
            return []

        pies: List[Dict[str, Any]] = []
        for csv_path in sorted(self.pies_dir.glob("*.csv")):
            pie = self._load_file(csv_path)
            if pie:
                pies.append(pie)
        return pies

    def _load_file(self, csv_path: Path) -> Dict[str, Any] | None:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            return None

        holdings: List[Dict[str, Any]] = []
        for row in rows:
            slice_code = (row.get("Slice") or "").strip()
            name = (row.get("Name") or "").strip()
            if not slice_code or slice_code.lower() == "total":
                continue
            # CSV values are deliberately ignored: this tool tracks only the
            # investable tickers in each Pie, never account size or P/L.
            holdings.append({"slice": slice_code, "name": name})

        if not holdings:
            return None

        return {
            "name": csv_path.stem,
            "path": str(csv_path),
            "holdings": holdings,
        }

    def as_portfolio_context(self) -> Dict[str, Any]:
        pies = self.load_all()
        return {
            "pie_count": len(pies),
            "pies": pies,
            "tickers": sorted({holding["slice"] for pie in pies for holding in pie["holdings"]}),
            "summary": [
                {
                    "name": pie["name"],
                    "holding_count": len(pie["holdings"]),
                }
                for pie in pies
            ],
        }
