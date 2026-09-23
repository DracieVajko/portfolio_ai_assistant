"""Pie-aware portfolio exposure and execution modelling.

Public API for aggregation and actionability views.
Re-exports for convenience.
"""
from investment_engine.portfolio.pie_metadata import PieMetadata, PieUniverse, load_pie_universe
from investment_engine.portfolio.sidecar import PieSidecarConfig, PieSidecarRegistry, load_sidecar_registry
from investment_engine.portfolio.exposure import (
    CanonicalExposure,
    PieExposure,
    StandaloneExposure,
    ExecutionCapabilities,
    build_canonical_exposures,
    evaluate_opportunistic_standalone_add,
    generate_actionability_view,
    generate_exposure_view,
)

__all__ = [
    "PieMetadata",
    "PieUniverse",
    "load_pie_universe",
    "PieSidecarConfig",
    "PieSidecarRegistry",
    "load_sidecar_registry",
    "CanonicalExposure",
    "PieExposure",
    "StandaloneExposure",
    "ExecutionCapabilities",
    "build_canonical_exposures",
    "evaluate_opportunistic_standalone_add",
    "generate_actionability_view",
    "generate_exposure_view",
]
