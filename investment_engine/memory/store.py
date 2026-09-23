from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MemoryStore:
    """Lightweight persistence for research notes and investment decisions."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or "investment_engine/memory/memory.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._data = {"research": [], "decisions": []}
            self._write()
        else:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def record_research(self, symbol: str, payload: Dict[str, Any]) -> None:
        self._data.setdefault("research", []).append({"symbol": symbol, **payload})
        self._write()

    def record_decision(self, symbol: str, payload: Dict[str, Any]) -> None:
        self._data.setdefault("decisions", []).append({"symbol": symbol, **payload})
        self._write()

    def snapshot(self) -> Dict[str, Any]:
        symbols = {entry.get("symbol") for entry in self._data.get("research", []) + self._data.get("decisions", []) if entry.get("symbol")}
        return {
            "research_count": len(self._data.get("research", [])),
            "decision_count": len(self._data.get("decisions", [])),
            "symbols": sorted(symbols),
        }

    def _write(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
