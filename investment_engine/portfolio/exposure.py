"""Canonical exposure aggregation and actionability views.

Uses Trading 212 API data for actual holdings (quantity/value) and pie CSV/config
only for classification and constraints. Never overwrites API values with CSV values.

No live order behavior, advisory only.
Hardened: per-pie allocation provenance prevents estimated splits from becoming facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from investment_engine.portfolio.pie_metadata import PieUniverse
from investment_engine.portfolio.sidecar import PieSidecarRegistry, PieSidecarConfig


@dataclass
class PieExposure:
    pie_id: str
    quantity: float
    value_eur: float
    weight_in_pie: Optional[float]  # None if unknown, never invented
    source: str  # api_pie_quantity_split or csv_universe
    display_name: Optional[str] = None
    # Provenance – prevents estimated allocations from becoming fake facts
    allocation_source: str = "unknown"  # api_exact | api_aggregate_single_pie | api_aggregate_estimated_split | csv_target_weight_estimate | unknown
    allocation_confidence: str = "low"  # high | medium | low
    allocation_is_estimated: bool = False
    allocation_note: str = "No reliable classification"
    per_pie_breakdown_reliability: str = "unavailable"  # exact | inferred | estimated | unavailable
    match_method: str = "unmatched"  # instrument_id | exact_normalized_symbol | safe_base_symbol | unmatched
    match_confidence: str = "low"  # high | medium | low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pie_id": self.pie_id,
            "display_name": self.display_name or self.pie_id,
            "quantity": round(self.quantity, 8),
            "value_eur": round(self.value_eur, 2),
            "weight_in_pie": round(self.weight_in_pie, 4) if self.weight_in_pie is not None else None,
            "source": self.source,
            "allocation_source": self.allocation_source,
            "allocation_confidence": self.allocation_confidence,
            "allocation_is_estimated": self.allocation_is_estimated,
            "allocation_note": self.allocation_note,
            "per_pie_breakdown_reliability": self.per_pie_breakdown_reliability,
            "match_method": self.match_method,
            "match_confidence": self.match_confidence,
        }


@dataclass
class StandaloneExposure:
    quantity: float
    value_eur: float
    source: str  # api_standalone or none

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantity": round(self.quantity, 8),
            "value_eur": round(self.value_eur, 2),
            "source": self.source,
        }


@dataclass
class ExecutionCapabilities:
    can_add_via_pie: bool
    can_reduce_via_pie: bool
    can_buy_standalone: bool
    can_sell_standalone: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_add_via_pie": self.can_add_via_pie,
            "can_reduce_via_pie": self.can_reduce_via_pie,
            "can_buy_standalone": self.can_buy_standalone,
            "can_sell_standalone": self.can_sell_standalone,
        }


@dataclass
class CanonicalExposure:
    symbol: str
    pie_exposures: List[PieExposure] = field(default_factory=list)
    standalone_exposure: StandaloneExposure = field(default_factory=lambda: StandaloneExposure(0.0, 0.0, "none"))
    total_quantity: float = 0.0
    total_value_eur: float = 0.0
    total_weight_of_account_pct: float = 0.0
    pie_count: int = 0
    execution_capabilities: ExecutionCapabilities = field(default_factory=lambda: ExecutionCapabilities(False, False, False, False))
    default_action_scope: str = "mixed"  # pie | standalone | mixed
    concentration_status: str = "OK"  # OK | WARNING | UNKNOWN
    concentration_cap_pct: Optional[float] = None
    data_validation_status: str = "PASS"
    data_validation_reason: str = ""
    pies_involved: List[str] = field(default_factory=list)
    horizon: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    per_pie_breakdown_reliability: str = "unavailable"  # exact|inferred|estimated|unavailable for whole symbol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "pie_exposures": [p.to_dict() for p in self.pie_exposures],
            "standalone_exposure": self.standalone_exposure.to_dict(),
            "total_quantity": round(self.total_quantity, 8),
            "total_value_eur": round(self.total_value_eur, 2),
            "total_weight_of_account_pct": round(self.total_weight_of_account_pct, 4),
            "pie_count": self.pie_count,
            "execution_capabilities": self.execution_capabilities.to_dict(),
            "default_action_scope": self.default_action_scope,
            "concentration_status": self.concentration_status,
            "concentration_cap_pct": self.concentration_cap_pct,
            "data_validation_status": self.data_validation_status,
            "data_validation_reason": self.data_validation_reason,
            "pies_involved": self.pies_involved,
            "horizon": self.horizon,
            "notes": self.notes,
            "per_pie_breakdown_reliability": self.per_pie_breakdown_reliability,
        }


def _resolve_price_eur(api_position: Optional[Dict[str, Any]]) -> float:
    """Get current price in EUR from API position, 0 if not held."""
    if not api_position:
        return 0.0
    return float(api_position.get("current_price_eur") or api_position.get("currentPrice") or 0.0)


def _get_sidecar_for_pie(pie_id: str, registry: PieSidecarRegistry) -> PieSidecarConfig:
    return registry.get_for_pie(pie_id)


def _has_per_pie_exact_data(api_pos: Dict[str, Any], pie_id: str) -> bool:
    """Check if raw T212 data provides per-pie quantity/value (api_exact)."""
    # Future-proof: check for explicit per-pie breakdown dict
    # e.g., api_pos["pie_allocations"] = {pie_id: {"quantity":..., "value_eur":...}}
    # or "per_pie_quantity", "pie_details"
    for key in ("pie_allocations", "per_pie_allocations", "pie_details", "per_pie_quantity"):
        val = api_pos.get(key)
        if isinstance(val, dict) and pie_id in val:
            entry = val[pie_id]
            if isinstance(entry, dict) and ("quantity" in entry or "value_eur" in entry):
                return True
        if isinstance(val, list):
            for entry in val:
                if isinstance(entry, dict) and entry.get("pie_id") == pie_id:
                    return True
    # Also check if pie_exposures already split
    return False


def _get_per_pie_exact_values(api_pos: Dict[str, Any], pie_id: str) -> tuple[Optional[float], Optional[float]]:
    for key in ("pie_allocations", "per_pie_allocations", "pie_details", "per_pie_quantity"):
        val = api_pos.get(key)
        if isinstance(val, dict) and pie_id in val:
            entry = val[pie_id]
            if isinstance(entry, dict):
                return entry.get("quantity"), entry.get("value_eur")
        if isinstance(val, list):
            for entry in val:
                if isinstance(entry, dict) and entry.get("pie_id") == pie_id:
                    return entry.get("quantity"), entry.get("value_eur")
    return None, None


def build_canonical_exposures(
    api_positions: List[Dict[str, Any]],
    pie_universe: PieUniverse,
    sidecar_registry: PieSidecarRegistry,
    total_equity_eur: float,
) -> List[CanonicalExposure]:
    """
    Build canonical exposures for every instrument.
    Preserves exact total exposure; per-pie allocation is marked with provenance.
    """
    api_by_symbol: Dict[str, Dict[str, Any]] = {}
    for pos in api_positions:
        sym = str(pos.get("symbol") or pos.get("ticker") or "").strip().upper()
        if not sym or sym == "UNKNOWN":
            continue
        if sym not in api_by_symbol:
            api_by_symbol[sym] = pos

    pie_tickers = set(t.upper() for t in pie_universe.all_tickers)
    api_tickers = set(api_by_symbol.keys())
    all_symbols = sorted(pie_tickers.union(api_tickers))

    exposures: List[CanonicalExposure] = []

    for symbol in all_symbols:
        api_pos = api_by_symbol.get(symbol)
        total_quantity = float(api_pos.get("quantity", 0.0)) if api_pos else 0.0
        pie_quantity_total = float(api_pos.get("pie_quantity", 0.0)) if api_pos else 0.0
        standalone_quantity = max(0.0, total_quantity - pie_quantity_total)
        total_value_eur = float(api_pos.get("value_eur", 0.0)) if api_pos else 0.0
        current_price_eur = _resolve_price_eur(api_pos)
        pie_value_total = pie_quantity_total * current_price_eur if api_pos else 0.0
        standalone_value_eur = max(0.0, total_value_eur - pie_value_total) if api_pos else 0.0
        if standalone_value_eur < 0 and standalone_value_eur > -0.01:
            standalone_value_eur = 0.0

        # Hardened matching with provenance
        # Get isin/instrument_id from api_pos if available
        isin = api_pos.get("isin") if api_pos else None
        instrument_id = api_pos.get("instrumentId") or api_pos.get("instrument_id") or api_pos.get("ticker") if api_pos else None
        # For pie matching, we need to use the original ticker string for base matching; instrument_id as secondary
        matched_with_provenance = pie_universe.pies_for_ticker_with_provenance(symbol, isin=isin, instrument_id=instrument_id)
        # If no match via provenance method, also try without isin (fallback exact base)
        # pies_for_ticker_with_provenance already handles exact base
        pies_for_symbol = [pie for pie, _, _ in matched_with_provenance]
        # Build map pie_id -> (match_method, match_confidence)
        match_map = {pie.pie_id: (method, conf) for pie, method, conf in matched_with_provenance}

        # If symbol is in all_symbols via pie_tickers union but not via pie match (e.g., pie_tickers contains raw CSV ticker "AAPL" but symbol is "AAPL_US_EQ" should have matched above),
        # the above should have matched; if still empty but symbol originated from pie_tickers, ensure we include pies containing that raw ticker
        # This handles case where api_tickers set contains "AAPL_US_EQ" but pie_tickers contains "AAPL" – the matching above will find it, so fine.

        pie_exposures: List[PieExposure] = []
        pies_involved = [p.pie_id for p in pies_for_symbol]

        # Determine allocation provenance for this symbol
        # Priority: api_exact > single_pie inferred > multi-pie estimated > unknown
        has_exact = False
        if api_pos and pies_for_symbol:
            # Check if any pie has exact per-pie data
            has_exact = any(_has_per_pie_exact_data(api_pos, pie.pie_id) for pie in pies_for_symbol)

        # Determine per-symbol reliability
        if has_exact:
            per_symbol_reliability = "exact"
        elif pies_for_symbol and len(pies_for_symbol) == 1 and pie_quantity_total > 1e-9:
            per_symbol_reliability = "inferred"
        elif pies_for_symbol and len(pies_for_symbol) > 1 and pie_quantity_total > 1e-9:
            per_symbol_reliability = "estimated"
        elif pies_for_symbol and pie_quantity_total == 0 and not api_pos:
            per_symbol_reliability = "unavailable"
        else:
            per_symbol_reliability = "unavailable"

        if pies_for_symbol:
            n_pies = len(pies_for_symbol)
            for pie_meta in pies_for_symbol:
                constituent = next((c for c in pie_meta.constituents if c.ticker.upper() == symbol or _base_match(c.ticker, symbol)), None)
                # Fallback base match for weight lookup
                if constituent is None:
                    # Try base match
                    for c in pie_meta.constituents:
                        if _base_match(c.ticker, symbol):
                            constituent = c
                            break
                weight_in_pie = constituent.target_weight if constituent and constituent.target_weight is not None else None

                # Determine per-pie allocation provenance
                match_method, match_conf = match_map.get(pie_meta.pie_id, ("unmatched", "low"))
                if has_exact:
                    qty_per_pie, val_per_pie = _get_per_pie_exact_values(api_pos, pie_meta.pie_id)
                    # Fallback to actual values if None
                    if qty_per_pie is None:
                        qty_per_pie = pie_quantity_total / n_pies if n_pies else 0.0
                    if val_per_pie is None:
                        val_per_pie = pie_value_total / n_pies if n_pies else 0.0
                    allocation_source = "api_exact"
                    allocation_confidence = "high"
                    allocation_is_estimated = False
                    allocation_note = "Per-pie allocation from broker"
                    reliability = "exact"
                    source = "api_exact"
                elif n_pies == 1 and pie_quantity_total > 1e-9:
                    qty_per_pie = pie_quantity_total
                    val_per_pie = pie_value_total
                    allocation_source = "api_aggregate_single_pie"
                    allocation_confidence = "medium"
                    allocation_is_estimated = False
                    allocation_note = "Pie allocation: inferred from aggregate pie quantity"
                    reliability = "inferred"
                    source = "api_aggregate_single_pie"
                elif n_pies > 1 and pie_quantity_total > 1e-9:
                    qty_per_pie = pie_quantity_total / n_pies if n_pies else 0.0
                    val_per_pie = pie_value_total / n_pies if n_pies else 0.0
                    allocation_source = "api_aggregate_estimated_split"
                    allocation_confidence = "low"
                    allocation_is_estimated = True
                    allocation_note = "Ticker belongs to multiple pies and the API exposes only aggregate pie quantity. Per-pie allocation is estimated."
                    reliability = "estimated"
                    source = "api_aggregate_estimated_split"
                elif not api_pos:
                    qty_per_pie = 0.0
                    val_per_pie = 0.0
                    allocation_source = "unknown"
                    allocation_confidence = "low"
                    allocation_is_estimated = False
                    allocation_note = "Not held – no allocation"
                    reliability = "unavailable"
                    source = "csv_universe"
                else:
                    # Held but no pie quantity? Should be standalone only but still in pie universe
                    qty_per_pie = 0.0
                    val_per_pie = 0.0
                    allocation_source = "unknown"
                    allocation_confidence = "low"
                    allocation_is_estimated = False
                    allocation_note = "No reliable classification"
                    reliability = "unavailable"
                    source = "unknown"

                sidecar = _get_sidecar_for_pie(pie_meta.pie_id, sidecar_registry)
                pie_exposures.append(
                    PieExposure(
                        pie_id=pie_meta.pie_id,
                        quantity=qty_per_pie,
                        value_eur=val_per_pie,
                        weight_in_pie=weight_in_pie,
                        source=source,
                        display_name=sidecar.display_name,
                        allocation_source=allocation_source,
                        allocation_confidence=allocation_confidence,
                        allocation_is_estimated=allocation_is_estimated,
                        allocation_note=allocation_note,
                        per_pie_breakdown_reliability=reliability,
                        match_method=match_method,
                        match_confidence=match_conf,
                    )
                )

        standalone = StandaloneExposure(
            quantity=standalone_quantity,
            value_eur=standalone_value_eur,
            source="api_standalone" if standalone_quantity > 0 else ("api_none" if not api_pos else "api_pie_only"),
        )

        total_weight_pct = (total_value_eur / total_equity_eur * 100) if total_equity_eur and total_equity_eur > 0 else 0.0

        can_add_via_pie = len(pies_for_symbol) > 0
        can_reduce_via_pie = len(pies_for_symbol) > 0 and pie_quantity_total > 1e-9
        if not pies_for_symbol:
            can_buy_standalone = True
            can_sell_standalone = standalone_quantity > 1e-9
            default_scope = "standalone"
            horizon = None
        else:
            allows_add = any(_get_sidecar_for_pie(p.pie_id, sidecar_registry).allow_standalone_adds for p in pies_for_symbol)
            allows_sell = any(_get_sidecar_for_pie(p.pie_id, sidecar_registry).allow_standalone_sells for p in pies_for_symbol)
            can_buy_standalone = bool(allows_add)
            can_sell_standalone = bool(allows_sell and standalone_quantity > 1e-9)
            scopes = [_get_sidecar_for_pie(p.pie_id, sidecar_registry).default_action_scope for p in pies_for_symbol]
            if "mixed" in scopes:
                default_scope = "mixed"
            elif "standalone" in scopes and "pie" in scopes:
                default_scope = "mixed"
            elif "standalone" in scopes:
                default_scope = "standalone"
            else:
                default_scope = "pie"
            horizons = [_get_sidecar_for_pie(p.pie_id, sidecar_registry).horizon for p in pies_for_symbol]
            horizon = "short_term" if "short_term" in horizons else "long_term"

        exec_caps = ExecutionCapabilities(
            can_add_via_pie=can_add_via_pie,
            can_reduce_via_pie=can_reduce_via_pie,
            can_buy_standalone=can_buy_standalone,
            can_sell_standalone=can_sell_standalone,
        )

        cap = None
        if pies_for_symbol:
            caps = [_get_sidecar_for_pie(p.pie_id, sidecar_registry).max_total_ticker_pct_of_account for p in pies_for_symbol]
            cap = min(caps) if caps else 10.0
        else:
            cap = 10.0

        concentration_status = "OK"
        if cap is not None and total_weight_pct > cap + 1e-9:
            concentration_status = "WARNING"
        elif total_weight_pct == 0 and not pies_for_symbol:
            concentration_status = "OK"

        if api_pos:
            data_validation_status = str(api_pos.get("validation_status", "PASS"))
            data_validation_reason = str(api_pos.get("validation_reason", ""))
        else:
            data_validation_status = "PASS"
            data_validation_reason = "Not held – universe only"
            if not pies_for_symbol:
                data_validation_reason = "Not held and not in universe"

        exposure = CanonicalExposure(
            symbol=symbol,
            pie_exposures=pie_exposures,
            standalone_exposure=standalone,
            total_quantity=total_quantity,
            total_value_eur=total_value_eur,
            total_weight_of_account_pct=total_weight_pct,
            pie_count=len(pie_exposures),
            execution_capabilities=exec_caps,
            default_action_scope=default_scope,
            concentration_status=concentration_status,
            concentration_cap_pct=cap,
            data_validation_status=data_validation_status,
            data_validation_reason=data_validation_reason,
            pies_involved=pies_involved,
            horizon=horizon,
            notes=[],
            per_pie_breakdown_reliability=per_symbol_reliability,
        )

        if pie_quantity_total > 0 and standalone_quantity > 0:
            exposure.notes.append("Aggregated pie + standalone exposure; standalone creates intentional drift from pie target weights.")
        if pies_for_symbol and not api_pos:
            exposure.notes.append("In pie universe but not currently held.")
        if per_symbol_reliability == "estimated":
            exposure.notes.append("Ticker belongs to multiple pies and the API exposes only aggregate pie quantity. Per-pie allocation is estimated.")

        exposures.append(exposure)

    exposures.sort(key=lambda e: e.total_value_eur, reverse=True)
    return exposures


def _base_match(csv_ticker: str, api_symbol: str) -> bool:
    """Helper for fallback weight lookup: exact normalized base match."""
    def _base(t: str) -> str:
        s = t.strip().upper()
        if "_" in s:
            s = s.split("_")[0]
        import re
        s = re.sub(r"[^A-Z0-9.]", "", s)
        return s
    return _base(csv_ticker) == _base(api_symbol)


def generate_exposure_view(
    canonical_exposures: List[CanonicalExposure],
    total_equity_eur: float,
    group_by_pie: bool = True,
) -> Dict[str, Any]:
    """Exposure view: aggregated pie+standalone by ticker with concentration warnings."""
    rows = []
    for exp in canonical_exposures:
        pie_value = sum(p.value_eur for p in exp.pie_exposures)
        standalone_value = exp.standalone_exposure.value_eur
        # Determine allocation reliability display
        # Only display when not exact
        reliability_note = None
        if exp.per_pie_breakdown_reliability != "exact" and exp.pie_count > 0:
            if exp.per_pie_breakdown_reliability == "inferred":
                reliability_note = "Pie allocation: inferred from aggregate pie quantity"
            elif exp.per_pie_breakdown_reliability == "estimated":
                reliability_note = "Pie allocation: estimated across multiple pies"
            elif exp.per_pie_breakdown_reliability == "unavailable":
                reliability_note = "Pie allocation: unavailable"
        rows.append(
            {
                "symbol": exp.symbol,
                "total_quantity": round(exp.total_quantity, 8),
                "total_value_eur": round(exp.total_value_eur, 2),
                "total_weight_pct": round(exp.total_weight_of_account_pct, 4),
                "pie_value_eur": round(pie_value, 2),
                "pie_quantity": round(sum(p.quantity for p in exp.pie_exposures), 8),
                "standalone_value_eur": round(standalone_value, 2),
                "standalone_quantity": round(exp.standalone_exposure.quantity, 8),
                "pie_count": exp.pie_count,
                "pies": exp.pies_involved,
                "concentration_status": exp.concentration_status,
                "cap_pct": exp.concentration_cap_pct,
                "concentration_warning": exp.concentration_status == "WARNING",
                "horizon": exp.horizon,
                "per_pie_breakdown_reliability": exp.per_pie_breakdown_reliability,
                "allocation_note": reliability_note,
            }
        )

    grouping: Dict[str, List[Dict[str, Any]]] = {}
    if group_by_pie:
        for row in rows:
            key = ",".join(row["pies"]) if row["pies"] else "STANDALONE_ONLY"
            grouping.setdefault(key, []).append(row)

    return {
        "total_equity_eur": round(total_equity_eur, 2),
        "exposures": rows,
        "grouped_by_pie": grouping,
        "disclaimer": "Exposure aggregates pie-held and standalone for risk analysis. CSV zero placeholders were ignored and never overwrote API values.",
    }


def _map_pie_action(exp: CanonicalExposure) -> str:
    """Map pie-held part to allowed actions. Hardened: estimated allocation -> REVIEW_PIE_ALLOCATION."""
    if exp.pie_count == 0:
        return "N/A"
    pie_qty = sum(p.quantity for p in exp.pie_exposures)
    if pie_qty < 1e-9:
        return "HOLD_PIE"
    # If any pie exposure is estimated, restrict
    if any(p.allocation_is_estimated for p in exp.pie_exposures):
        return "REVIEW_PIE_ALLOCATION"
    if exp.concentration_status == "WARNING":
        return "REVIEW_PIE"
    return "HOLD_PIE"


def _map_standalone_action(exp: CanonicalExposure) -> str:
    """Map standalone part to allowed actions."""
    if exp.standalone_exposure.quantity < 1e-9:
        if exp.execution_capabilities.can_buy_standalone:
            return "WATCH"
        else:
            return "WATCH"
    if exp.execution_capabilities.can_sell_standalone:
        return "HOLD"
    else:
        return "HOLD"


def generate_actionability_view(
    canonical_exposures: List[CanonicalExposure],
) -> Dict[str, Any]:
    """Actionability view: what is operationally possible.

    Never labels a pie constituent simply SELL.
    Never renders low-confidence allocation as direct executable pie action.
    """
    rows = []
    for exp in canonical_exposures:
        pie_action = _map_pie_action(exp)
        standalone_action = _map_standalone_action(exp)
        assert pie_action != "SELL", f"Pie action must never be SELL for {exp.symbol}"
        allowed_pie = {"HOLD_PIE", "ADD_VIA_PIE", "REDUCE_VIA_PIE", "REVIEW_PIE", "REVIEW_PIE_ALLOCATION", "N/A"}
        assert pie_action in allowed_pie, f"Invalid pie action {pie_action}"

        explanation = ""
        if pie_action == "REVIEW_PIE_ALLOCATION":
            explanation = "Ticker belongs to multiple pies and the API exposes only aggregate pie quantity. Per-pie allocation is estimated."
        elif pie_action == "REVIEW_PIE":
            explanation = (
                f"{exp.symbol} total exposure {exp.total_weight_of_account_pct:.2f}% exceeds cap {exp.concentration_cap_pct}% – "
                f"review pie as a whole; single-ticker risk inside pie cannot be sold individually."
            )
        elif exp.pie_count > 0 and exp.standalone_exposure.quantity > 1e-9:
            explanation = "Standalone portion creates drift from pie target weights; pie action does not modify standalone and vice versa."

        # Also handle estimated allocation note for actionability
        if any(p.allocation_is_estimated for p in exp.pie_exposures) and pie_action == "REVIEW_PIE_ALLOCATION":
            # Ensure explanation contains required sentence
            if "aggregate pie quantity. Per-pie allocation is estimated" not in explanation:
                explanation = "Ticker belongs to multiple pies and the API exposes only aggregate pie quantity. Per-pie allocation is estimated."

        rows.append(
            {
                "symbol": exp.symbol,
                "pie_action": pie_action,
                "standalone_action": standalone_action,
                "explanation": explanation,
                "can_add_via_pie": exp.execution_capabilities.can_add_via_pie and not any(p.allocation_is_estimated for p in exp.pie_exposures),
                "can_reduce_via_pie": exp.execution_capabilities.can_reduce_via_pie and not any(p.allocation_is_estimated for p in exp.pie_exposures),
                "can_buy_standalone": exp.execution_capabilities.can_buy_standalone,
                "can_sell_standalone": exp.execution_capabilities.can_sell_standalone,
                "default_action_scope": exp.default_action_scope,
                "pies_involved": exp.pies_involved,
                "per_pie_breakdown_reliability": exp.per_pie_breakdown_reliability,
            }
        )

    return {
        "actions": rows,
        "disclaimer": "Advisory only. Pie-held actions are pie-level; standalone actions do not modify pie weights. No automatic orders.",
        "rules": {
            "pie_allowed": ["HOLD_PIE", "ADD_VIA_PIE", "REDUCE_VIA_PIE", "REVIEW_PIE", "REVIEW_PIE_ALLOCATION"],
            "standalone_allowed": ["HOLD", "BUY_STANDALONE", "SELL_STANDALONE", "WATCH"],
            "never": "Never label a pie constituent simply SELL",
        },
    }


def evaluate_opportunistic_standalone_add(
    symbol: str,
    proposed_quantity: float,
    proposed_price_eur: float,
    canonical: CanonicalExposure,
    sidecar: PieSidecarConfig,
    total_equity_eur: float,
    reconciliation_status: str,
    signal_validated: bool,
) -> Dict[str, Any]:
    """Evaluate an opportunistic standalone add proposal.

    Requirements (all must be true):
    - reconciliation PASS
    - signal_validated True
    - sidecar.allow_standalone_adds True
    - total exposure after purchase < max_total_ticker_pct_of_account
    - standalone amount < max_standalone_add_pct_of_account
    No recommendation implies automatic order.
    """
    symbol_u = symbol.strip().upper()
    if canonical.symbol != symbol_u:
        return {"allowed": False, "reason": "Symbol mismatch", "action": "BLOCKED"}

    if reconciliation_status != "PASS":
        return {"allowed": False, "reason": "Reconciliation not PASS – advisory suppressed", "action": "BLOCKED"}

    if not signal_validated:
        return {"allowed": False, "reason": "Signal not validated", "action": "BLOCKED"}

    if not sidecar.allow_standalone_adds:
        return {"allowed": False, "reason": f"Standalone adds disabled for pie {sidecar.pie_id} (horizon {sidecar.horizon})", "action": "BLOCKED"}

    proposed_value = proposed_quantity * proposed_price_eur
    proposed_standalone_pct = (proposed_value / total_equity_eur * 100) if total_equity_eur else 0
    total_after = canonical.total_value_eur + proposed_value
    total_after_pct = (total_after / total_equity_eur * 100) if total_equity_eur else 0

    if proposed_standalone_pct > sidecar.max_standalone_add_pct_of_account + 1e-9:
        return {
            "allowed": False,
            "reason": f"Standalone add {proposed_standalone_pct:.2f}% exceeds cap {sidecar.max_standalone_add_pct_of_account}% for {sidecar.pie_id}",
            "action": "BLOCKED",
            "proposed_value_eur": round(proposed_value, 2),
            "proposed_pct": round(proposed_standalone_pct, 4),
        }

    if total_after_pct > sidecar.max_total_ticker_pct_of_account + 1e-9:
        return {
            "allowed": False,
            "reason": f"Total exposure after purchase {total_after_pct:.2f}% exceeds cap {sidecar.max_total_ticker_pct_of_account}% for {sidecar.pie_id}",
            "action": "BLOCKED",
            "total_after_pct": round(total_after_pct, 4),
            "total_after_value": round(total_after, 2),
        }

    return {
        "allowed": True,
        "reason": "Validated signal, reconciliation PASS, caps respected – advisory OPPORTUNISTIC_STANDALONE_ADD only. Does not modify pie weights. No automatic order.",
        "action": "OPPORTUNISTIC_STANDALONE_ADD",
        "proposed_value_eur": round(proposed_value, 2),
        "proposed_pct": round(proposed_standalone_pct, 4),
        "total_after_pct": round(total_after_pct, 4),
        "advisory": True,
        "execution": "BUY_STANDALONE (advisory, requires manual execution)",
    }
