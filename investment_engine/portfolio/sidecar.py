"""Pie sidecar configuration – advisory execution constraints only.

No account calculations are altered. Target weights are never invented.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_VALID_HORIZONS = {"short_term", "long_term"}
_VALID_EXECUTION_MODES = {"pie_only", "mixed"}
_VALID_ACTION_SCOPES = {"pie", "standalone", "mixed"}
_VALID_REBALANCE_MODES = {"no_rebalance", "drift_allowed", "periodic", "unknown"}


@dataclass
class PieSidecarConfig:
    """Validated sidecar for a single pie."""
    pie_id: str
    display_name: str
    horizon: str  # short_term | long_term
    purpose: str
    default_execution_mode: str  # pie_only | mixed
    allow_standalone_adds: bool
    allow_standalone_sells: bool
    max_standalone_add_pct_of_account: float
    max_total_ticker_pct_of_account: float
    rebalance_mode: str
    analysis_mode: str  # aggregate_pie_and_standalone
    default_action_scope: str  # pie | standalone | mixed
    target_weights: Optional[Dict[str, float]] = None  # None = unknown, never invented
    raw_path: Optional[str] = None
    source: str = "sidecar"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_path: Optional[str] = None) -> "PieSidecarConfig":
        # Required fields validation
        pie_id = str(data.get("pie_id") or data.get("pieId") or "").strip()
        if not pie_id:
            raise ValueError("sidecar missing required field pie_id")
        display_name = str(data.get("display_name") or data.get("displayName") or pie_id).strip()
        horizon = str(data.get("horizon", "long_term")).strip()
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"invalid horizon {horizon} for {pie_id}")
        default_execution_mode = str(data.get("default_execution_mode", "pie_only")).strip()
        if default_execution_mode not in _VALID_EXECUTION_MODES:
            raise ValueError(f"invalid default_execution_mode {default_execution_mode}")
        default_action_scope = str(data.get("default_action_scope", "pie")).strip()
        if default_action_scope not in _VALID_ACTION_SCOPES:
            raise ValueError(f"invalid default_action_scope {default_action_scope}")
        rebalance_mode = str(data.get("rebalance_mode", "unknown")).strip()
        if rebalance_mode not in _VALID_REBALANCE_MODES:
            # Allow unknown but warn
            rebalance_mode = "unknown"
        analysis_mode = str(data.get("analysis_mode", "aggregate_pie_and_standalone")).strip()
        # Booleans
        allow_adds = bool(data.get("allow_standalone_adds", False))
        allow_sells = bool(data.get("allow_standalone_sells", False))
        # Numeric caps
        try:
            max_standalone = float(data.get("max_standalone_add_pct_of_account", 0.0))
            max_total = float(data.get("max_total_ticker_pct_of_account", 10.0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid pct for {pie_id}: {e}")

        # target_weights: explicit None = unknown, never invented
        raw_tw = data.get("target_weights")
        target_weights: Optional[Dict[str, float]] = None
        if raw_tw is not None:
            if not isinstance(raw_tw, dict):
                raise ValueError(f"target_weights must be dict or null for {pie_id}")
            # Normalize tickers to upper
            target_weights = {str(k).strip().upper(): float(v) for k, v in raw_tw.items()}
            # Validate sum only if supplied - caller may check
        else:
            target_weights = None

        return cls(
            pie_id=pie_id,
            display_name=display_name,
            horizon=horizon,
            purpose=str(data.get("purpose", "")),
            default_execution_mode=default_execution_mode,
            allow_standalone_adds=allow_adds,
            allow_standalone_sells=allow_sells,
            max_standalone_add_pct_of_account=max_standalone,
            max_total_ticker_pct_of_account=max_total,
            rebalance_mode=rebalance_mode,
            analysis_mode=analysis_mode,
            default_action_scope=default_action_scope,
            target_weights=target_weights,
            raw_path=raw_path,
            source="sidecar",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pie_id": self.pie_id,
            "display_name": self.display_name,
            "horizon": self.horizon,
            "purpose": self.purpose,
            "default_execution_mode": self.default_execution_mode,
            "allow_standalone_adds": self.allow_standalone_adds,
            "allow_standalone_sells": self.allow_standalone_sells,
            "max_standalone_add_pct_of_account": self.max_standalone_add_pct_of_account,
            "max_total_ticker_pct_of_account": self.max_total_ticker_pct_of_account,
            "rebalance_mode": self.rebalance_mode,
            "analysis_mode": self.analysis_mode,
            "default_action_scope": self.default_action_scope,
            "target_weights": self.target_weights,
            "raw_path": self.raw_path,
            "source": self.source,
        }


@dataclass
class PieSidecarRegistry:
    """Registry of all sidecar configs keyed by pie_id."""
    configs: Dict[str, PieSidecarConfig] = field(default_factory=dict)

    def get(self, pie_id: str) -> Optional[PieSidecarConfig]:
        return self.configs.get(pie_id) or self.configs.get(pie_id.lower()) or self.configs.get(pie_id.upper())

    def get_for_pie(self, pie_id: str) -> PieSidecarConfig:
        """Return sidecar if exists, else conservative default (pie_only, no standalone)."""
        cfg = self.get(pie_id)
        if cfg is not None:
            return cfg
        # Conservative default for pies without sidecar
        return PieSidecarConfig(
            pie_id=pie_id,
            display_name=pie_id,
            horizon="long_term",
            purpose="Default conservative – no sidecar",
            default_execution_mode="pie_only",
            allow_standalone_adds=False,
            allow_standalone_sells=False,
            max_standalone_add_pct_of_account=0.0,
            max_total_ticker_pct_of_account=10.0,
            rebalance_mode="unknown",
            analysis_mode="aggregate_pie_and_standalone",
            default_action_scope="pie",
            target_weights=None,
            raw_path=None,
            source="default",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.configs.items()}


def load_sidecar_registry(config_dir: str | Path = "PIEs/config") -> PieSidecarRegistry:
    """Load all JSON sidecars under config_dir. Invalid files are skipped with warning."""
    config_dir = Path(config_dir)
    registry = PieSidecarRegistry(configs={})
    if not config_dir.exists():
        return registry
    for json_path in sorted(config_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            cfg = PieSidecarConfig.from_dict(data, raw_path=str(json_path))
            # Use pie_id as key; also store variant for case-insensitive lookup
            registry.configs[cfg.pie_id] = cfg
        except Exception:
            # Skip invalid sidecar, do not crash
            continue
    return registry
