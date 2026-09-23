"""Pie metadata layer - CSV as universe/metadata only.

Never uses current value / invested value / owned quantity / result / dividends for
account calculations. Zero placeholders are ignored. Never overwrites API positions.

Hardened matching: prefers stable broker identifier (ISIN/instrument ID) when available,
then exact normalized base-symbol via the single canonical layer
``investment_engine.portfolio.symbols.to_display_symbol``. Never uses loose
substring/prefix matching.
"""
from __future__ import annotations
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from investment_engine.portfolio.symbols import to_display_symbol as _canonical_display
logger = logging.getLogger(__name__)
_WEIGHT_CANDIDATES = {
    "weight",
    "target weight",
    "target_weight",
    "allocation",
    "target allocation",
    "allocation %",
    "weight %",
    "target weight %",
    "pct",
    "percent",
    "target_pct",
}
_ISIN_CANDIDATES = {
    "isin",
    "instrument_isin",
    "instrument id",
    "instrument_id",
    "figi",
}
_IGNORED_VALUE_COLUMNS = {
    "invested value",
    "value",
    "result",
    "owned quantity",
    "dividends gained",
    "dividends cash",
    "dividends reinvested",
    "invested_value",
    "owned_quantity",
}
def _normalize_base_symbol(ticker: str, known_clean=None) -> str:
    """Thin provenance wrapper around the single mapping layer.
    Delegates to ``symbols.to_display_symbol`` (canonical). The previous local
    copy diverged (``VWSBd_EQ`` -> ``VWSBD``); the canonical layer gives
    ``VWSB``. Kept under the historic name for backward-compatible imports
    (``main.py`` / tests); new code should call ``to_display_symbol`` directly.
    """
    try:
        disp = _canonical_display(ticker, known_clean)
        logger.debug("pie_metadata normalize %r -> %r via canonical display", ticker, disp)
        return disp
    except Exception:
        s = (ticker or "").strip().upper()
        if "_" in s:
            s = s.split("_")[0]
        return s
@dataclass
class PieConstituent:
    """One row in a pie CSV - metadata only."""
    ticker: str
    name: str
    target_weight: Optional[float] = None
    isin: Optional[str] = None
    instrument_id: Optional[str] = None
@dataclass
class PieMetadata:
    """Validated metadata for a single pie."""
    pie_id: str
    display_name: str
    path: str
    constituents: List[PieConstituent] = field(default_factory=list)
    has_target_weights: bool = False
    weight_sum: Optional[float] = None
    weights_valid: Optional[bool] = None
    weight_validation_error: Optional[str] = None
    source: str = "csv_universe_only"
    @property
    def tickers(self) -> List[str]:
        return [c.ticker for c in self.constituents]
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pie_id": self.pie_id,
            "display_name": self.display_name,
            "path": self.path,
            "tickers": self.tickers,
            "has_target_weights": self.has_target_weights,
            "weight_sum": self.weight_sum,
            "weights_valid": self.weights_valid,
            "weight_validation_error": self.weight_validation_error,
            "constituents": [
                {"ticker": c.ticker, "name": c.name, "target_weight": c.target_weight, "isin": c.isin}
                for c in self.constituents
            ],
            "source": self.source,
        }
@dataclass
class PieUniverse:
    """Collection of all pies."""
    pies: List[PieMetadata] = field(default_factory=list)
    @property
    def all_tickers(self) -> List[str]:
        s = set()
        for p in self.pies:
            s.update(p.tickers)
        return sorted(s)
    def pies_for_ticker(self, ticker: str, isin: Optional[str] = None, instrument_id: Optional[str] = None) -> List[PieMetadata]:
        """Backward-compatible: returns pies for ticker using hardened matching.
        Prefers ISIN/instrument_id when available, then exact normalized base-symbol.
        Never uses loose substring.
        """
        matched = self.pies_for_ticker_with_provenance(ticker, isin, instrument_id)
        return [pie for pie, _, _ in matched]
    def pies_for_ticker_with_provenance(self, ticker: str, isin: Optional[str] = None, instrument_id: Optional[str] = None) -> List[Tuple[PieMetadata, str, str]]:
        """
        Returns list of (pie, match_method, match_confidence) with hardened matching.
        match_method: instrument_id | exact_normalized_symbol | safe_base_symbol | unmatched
        match_confidence: high | medium | low
        """
        _known = {str(c.ticker or "").strip().upper() for pie in self.pies for c in pie.constituents if c.ticker}
        ticker_norm = _normalize_base_symbol(ticker, _known)
        isin_norm = isin.strip().upper() if isin else None
        inst_norm = str(instrument_id).strip().upper() if instrument_id else None
        results: List[Tuple[PieMetadata, str, str]] = []
        for pie in self.pies:
            for constituent in pie.constituents:
                if isin_norm and constituent.isin and isin_norm == constituent.isin.strip().upper():
                    results.append((pie, "instrument_id", "high"))
                    break
                if inst_norm and constituent.instrument_id and inst_norm == constituent.instrument_id.strip().upper():
                    results.append((pie, "instrument_id", "high"))
                    break
                csv_base = _normalize_base_symbol(constituent.ticker, _known)
                if ticker_norm and csv_base and ticker_norm == csv_base:
                    results.append((pie, "exact_normalized_symbol", "high"))
                    break
        return results
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pie_count": len(self.pies),
            "pies": [p.to_dict() for p in self.pies],
            "all_tickers": self.all_tickers,
        }
def _detect_weight_column(fieldnames: List[str]) -> Optional[str]:
    """Return actual fieldname that matches weight candidates, or None."""
    if not fieldnames:
        return None
    lowered_map = {fn.strip().lower(): fn for fn in fieldnames}
    for cand in _WEIGHT_CANDIDATES:
        if cand in lowered_map:
            return lowered_map[cand]
    for fn in fieldnames:
        key = fn.strip().lower().replace("%", "").strip()
        if key in _WEIGHT_CANDIDATES:
            return fn
    return None
def _detect_isin_column(fieldnames: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    lowered_map = {fn.strip().lower(): fn for fn in fieldnames}
    for cand in _ISIN_CANDIDATES:
        if cand in lowered_map:
            return lowered_map[cand]
    return None
def _parse_weight(value: str) -> Optional[float]:
    """Parse weight cell; return None if empty/zero-placeholder/invalid."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "n/a" or s.lower() == "nan":
        return None
    s = s.replace("%", "").strip()
    try:
        f = float(s)
    except (ValueError, TypeError):
        return None
    return f
def load_pie_universe(pies_dir: str | Path = "PIEs") -> PieUniverse:
    """Load all PIEs/*.csv as metadata only. Ignores PIEs/config/*.json."""
    pies_dir = Path(pies_dir)
    if not pies_dir.exists():
        return PieUniverse(pies=[])
    pies: List[PieMetadata] = []
    for csv_path in sorted(pies_dir.glob("*.csv")):
        meta = _load_single_pie(csv_path)
        if meta is not None:
            pies.append(meta)
    return PieUniverse(pies=pies)
def _load_single_pie(csv_path: Path) -> Optional[PieMetadata]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            weight_col = _detect_weight_column(fieldnames)
            isin_col = _detect_isin_column(fieldnames)
            rows = list(reader)
    except Exception:
        return None
    constituents: List[PieConstituent] = []
    weight_values: List[float] = []
    for row in rows:
        slice_code = (row.get("Slice") or row.get("slice") or row.get("Ticker") or row.get("ticker") or "").strip()
        name = (row.get("Name") or row.get("name") or "").strip()
        if not slice_code or slice_code.lower() == "total":
            continue
        target_weight: Optional[float] = None
        if weight_col is not None:
            raw_w = row.get(weight_col, "")
            parsed = _parse_weight(raw_w)
            if parsed is not None:
                target_weight = parsed
                weight_values.append(parsed)
        isin_val = None
        instrument_id = None
        if isin_col is not None:
            raw_isin = row.get(isin_col, "")
            if raw_isin and str(raw_isin).strip():
                isin_val = str(raw_isin).strip().upper()
        for id_col in ["instrument_id", "instrument id", "figi"]:
            if id_col in [fn.strip().lower() for fn in fieldnames]:
                for fn in fieldnames:
                    if fn.strip().lower() == id_col:
                        val = row.get(fn, "")
                        if val and str(val).strip():
                            instrument_id = str(val).strip()
                        break
        constituents.append(
            PieConstituent(
                ticker=slice_code.strip().upper(),
                name=name,
                target_weight=target_weight,
                isin=isin_val,
                instrument_id=instrument_id,
            )
        )
    if not constituents:
        return None
    pie_id = csv_path.stem
    has_weights = False
    weight_sum: Optional[float] = None
    weights_valid: Optional[bool] = None
    weight_error: Optional[str] = None
    if weight_col is not None and weight_values:
        non_zero = [w for w in weight_values if abs(w) > 1e-9]
        if not non_zero:
            has_weights = False
        else:
            if len(weight_values) != len(constituents):
                has_weights = True
                weight_sum = sum(weight_values)
                weights_valid = False
                weight_error = f"Incomplete target weights: {len(weight_values)}/{len(constituents)} supplied"
            else:
                has_weights = True
                weight_sum = sum(weight_values)
                if abs(weight_sum - 100.0) <= 0.5:
                    weights_valid = True
                else:
                    weights_valid = False
                    weight_error = f"Target weights sum to {weight_sum:.2f}%, expected 100% +/-0.5%"
    return PieMetadata(
        pie_id=pie_id,
        display_name=pie_id,
        path=str(csv_path),
        constituents=constituents,
        has_target_weights=has_weights,
        weight_sum=weight_sum,
        weights_valid=weights_valid,
        weight_validation_error=weight_error,
        source="csv_universe_only",
    )
