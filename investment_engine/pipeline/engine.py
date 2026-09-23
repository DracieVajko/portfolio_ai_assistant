from __future__ import annotations

from typing import Any, List, Dict

from investment_engine.config.settings import EngineSettings
from investment_engine.prompts.loader import build_prompt_bundle
from investment_engine.providers.base import BaseLLMProvider
from investment_engine.providers.factory import ProviderFactory
from investment_engine.scoring.priority import score_asset


class InvestmentPipeline:
    """Coordination layer for the staged investment reasoning pipeline."""

    def __init__(self, provider: BaseLLMProvider | None = None, prompt_dir: str | None = None):
        self.settings = EngineSettings.from_defaults()
        self.provider = provider or ProviderFactory.create(self.settings)
        self.prompt_dir = prompt_dir or self.settings.prompt_dir

    def run(self, assets: List[Dict[str, Any]], portfolio_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Run the staged pipeline and return structure-ready output."""
        ranked_assets = sorted(
            assets,
            key=lambda asset: score_asset(asset),
            reverse=True,
        )[:5]

        stage_outputs: Dict[str, Dict[str, str]] = {}
        stage_backends: Dict[str, Dict[str, str]] = {}
        for asset in ranked_assets:
            symbol = str(asset.get("broker_symbol") or asset.get("name") or "UNKNOWN")
            prompt_bundle = build_prompt_bundle(symbol)
            stage_map: Dict[str, str] = {}
            backend_map: Dict[str, str] = {}
            for stage in ("research", "think", "adrian_alpha", "summary"):
                stage_map[stage] = self.provider.generate(
                    prompt_bundle[stage],
                    stage=stage,
                    context={"asset": asset, "portfolio_context": portfolio_context or {}},
                )
                backend_map[stage] = self.provider.name
            stage_outputs[symbol] = stage_map
            stage_backends[symbol] = backend_map

        markdown = [
            "# Investment Intelligence Engine",
            "",
            "## Executive Summary",
            "The sections below contain the returned model output for every analysed asset.",
            "",
            "## Highest Priority Actions",
            *[f"- {asset.get('broker_symbol') or asset.get('name') or 'UNKNOWN'} (priority score: {score_asset(asset):.1f})" for asset in ranked_assets],
            "",
        ]
        for symbol, stages in stage_outputs.items():
            markdown.extend(["", f"## {symbol}"])
            for stage in ("research", "think", "adrian_alpha", "summary"):
                markdown.extend(["", f"### {stage.replace('_', ' ').title()}", stages.get(stage, "No output returned.")])
        markdown.extend([
            "",
            "## Overall Recommendation",
            "Review the per-asset Summary sections above; each contains the model's current recommendation and supporting reasoning.",
        ])

        return {
            "ranked_assets": ranked_assets,
            "stage_outputs": stage_outputs,
            "stage_backends": stage_backends,
            "markdown": "\n".join(markdown),
        }
