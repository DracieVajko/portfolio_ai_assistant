"""
Report structure layer: defines the three-layer output structure and generators.

Layer 1: Debug layer (reports/debug/<run_id>/)
  - raw_endpoint_dump.json
  - reconciliation_diagnostic.md
  - reconciliation_diagnostic.csv
  - run.log (copy)

Layer 2: AI/machine context layer (reports/ai_context/)
  - ai_context_<run_id>.md
  - portfolio_analysis_<run_id>.json

Layer 3: Human summary layer (reports/summary/)
  - portfolio_brief.md (overwritten each run)
  - archive/portfolio_brief_<timestamp>.md (optional archive)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from investment_engine.reporting.regime_report import normalize_signal

logger = logging.getLogger(__name__)

# Retention settings
DEBUG_RETENTION_DAYS = 30
DEBUG_MAX_RUNS = 14


def setup_debug_layer(run_id: str, base_dir: Path = Path("reports")) -> Dict[str, Path]:
    """
    Create debug layer folder structure for a run.
    Returns dict with paths for each debug output.
    """
    debug_dir = Path("reports") / "debug" / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Prune old debug folders
    _prune_debug_layer(base_dir=Path("reports") / "debug")

    paths = {
        "debug_dir": debug_dir,
        "raw_endpoint_dump": debug_dir / "raw_endpoint_dump.json",
        "reconciliation_diagnostic_md": debug_dir / "reconciliation_diagnostic.md",
        "reconciliation_diagnostic_csv": debug_dir / "reconciliation_diagnostic.csv",
        "run_log": None,  # Will be set after log is known
    }
    return paths


def _prune_debug_layer(base_dir: Path) -> None:
    """Prune debug folders older than retention policy."""
    if not base_dir.exists():
        return

    cutoff_date = datetime.now() - timedelta(days=DEBUG_RETENTION_DAYS)
    run_dirs: List[tuple[datetime, Path]] = []

    for item in base_dir.iterdir():
        if item.is_dir():
            try:
                # Expect format: run_id (8 chars) or timestamp
                dir_name = item.name
                # Try to parse as date from name or use mtime
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                run_dirs.append((mtime, item))
            except Exception:
                pass

    run_dirs.sort(key=lambda x: x[0], reverse=True)

    # Keep by count
    for _, dir_path in run_dirs[DEBUG_MAX_RUNS:]:
        logger.info("Pruning debug folder (max runs exceeded): %s", dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)

    # Keep by date
    for mtime, dir_path in run_dirs:
        if mtime < cutoff_date:
            logger.info("Pruning debug folder (older than %d days): %s", DEBUG_RETENTION_DAYS, dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)


def move_debug_outputs(run_id: str, run_log_path: Path, debug_paths: Dict[str, Path]) -> None:
    """Move debug outputs to the debug layer folder."""
    debug_dir = debug_paths["debug_dir"]

    # Move reconciliation diagnostic files
    for src_name, dst_path in [
        ("reconciliation_diagnostic.csv", debug_paths["reconciliation_diagnostic_csv"]),
        ("reconciliation_diagnostic.md", debug_paths["reconciliation_diagnostic_md"]),
    ]:
        src = Path("reports") / src_name
        if src.exists():
            try:
                shutil.move(str(src), str(dst_path))
                logger.debug("Moved %s to %s", src, dst_path)
            except Exception as e:
                logger.warning("Failed to move %s: %s", src, e)

    # Move raw endpoint dump (pattern: raw_endpoint_dump_<run_id>.json or raw_endpoint_dump.json)
    for pattern in [f"raw_endpoint_dump_{run_id}.json", "raw_endpoint_dump.json"]:
        src = Path("reports") / pattern
        if src.exists():
            try:
                shutil.move(str(src), str(debug_paths["raw_endpoint_dump"]))
                logger.debug("Moved %s to %s", src, debug_paths["raw_endpoint_dump"])
                break
            except Exception as e:
                logger.warning("Failed to move %s: %s", src, e)

    # Copy run log
    if run_log_path and run_log_path.exists():
        dst_log = debug_dir / "run.log"
        try:
            shutil.copy2(run_log_path, dst_log)
            debug_paths["run_log"] = dst_log
        except Exception as e:
            logger.warning("Failed to copy run log: %s", e)


def save_ai_context_layer(
    run_id: str,
    context_content: str,
    portfolio_analysis: dict,
    base_dir: Path = Path("reports")
) -> tuple[Path, Path]:
    """
    Save AI context layer files.
    Returns (context_file_path, portfolio_analysis_path).
    """
    ai_context_dir = Path("reports") / "ai_context"
    ai_context_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_file = ai_context_dir / f"ai_context_{run_id}.md"
    analysis_file = ai_context_dir / f"portfolio_analysis_{run_id}.json"

    context_file.write_text(context_content, encoding="utf-8")
    analysis_file.write_text(json.dumps(portfolio_analysis, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    logger.info("AI context saved: %s", context_file)
    logger.info("Portfolio analysis saved: %s", analysis_file)

    return context_file, analysis_file


def generate_human_brief(
    result: dict,
    portfolio_rows: list[dict],
    monitoring_items: list[dict],
    regime_result: Optional[Any],
    t212_data: Optional[dict],
    decision_news: list[dict],
    earnings_7d: dict,
    ideas: list[dict],
    portfolio_names: dict,
    reconciliation: Optional[Any],
) -> str:
    """
    Generate a concise human-readable portfolio brief.
    This is a presentation-only filter over already-computed data.
    """
    from investment_engine.research.market_data import report_now_bta

    report_dt = report_now_bta()
    recon_status = str((result.get("reconciliation") or {}).get("status", "UNKNOWN")).upper()
    regime = getattr(regime_result, "regime", "UNKNOWN") if regime_result else "UNKNOWN"

    try:
        rsi = regime_result.timeframes.get("daily", {}).get("indicators", {}).get("RSI_14", 0)
        regime_line = f"{regime} (RSI {float(rsi):.0f})"
    except Exception:
        regime_line = str(regime)

    # Portfolio status (.get default never fires on explicit nulls -> coerce)
    def _num(value):
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    broker_equity = _num((result.get("t212_data", {}).get("account_summary", {}) or {}).get("total_equity", 0))
    free_cash = _num(((result.get("t212_data", {}).get("cash", {}) or {}).get("free", 0)))
    blocked_cash = _num(((result.get("t212_data", {}).get("cash", {}) or {}).get("blocked", 0)))
    pie_cash = _num(((result.get("t212_data", {}).get("cash", {}) or {}).get("pie_cash", 0)))
    free_available = free_cash  # explicitly free cash, not including blocked
    total_reported_cash = free_cash + pie_cash + blocked_cash

    net_pnl = _num((result.get("account_performance", {}) or {}).get("net_pnl_after_costs_eur", 0))
    return_pct = _num((result.get("account_performance", {}) or {}).get("return_pct", 0))
    stance = _get_portfolio_stance(portfolio_rows, recon_status)

    lines = [
        "# Portfolio Brief",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## Portfolio Status",
        f"- Broker equity: €{broker_equity:,.2f}",
        f"- Net P&L: €{net_pnl:+,.2f} ({return_pct:+.2f}%)",
        f"- Reconciliation: {recon_status}",
        f"- Free cash (deployable): €{free_available:,.2f}",
        f"- Blocked cash (pending orders): €{blocked_cash:,.2f}",
        f"- Pie cash: €{pie_cash:,.2f}",
        f"- Total reported cash: €{total_reported_cash:,.2f}",
        f"- Stance: **{stance}** — {_get_stance_reason(portfolio_rows, recon_status)}",
        "",
        "## Recommendations",
    ]

    # Collect flagged positions
    flagged = _collect_flagged_positions(portfolio_rows, monitoring_items, result)

    if flagged:
        for item in flagged:
            lines.append(f"- **{item['symbol']}** ({item['company']}): {item['reason']}")
    else:
        lines.append("No positions require attention this run.")

    lines.extend(["", "## News & Earnings"])

    # Decision-relevant news (last 48h)
    decision_news_items = result.get("decision_news", [])
    if decision_news_items:
        for news in decision_news_items[:5]:
            lines.append(f"- {news.get('title', '')[:100]} ({news.get('source', '')})")
    else:
        lines.append("No decision-relevant news in the last 48 hours.")

    # Earnings within 7 days
    earnings_7d_window = result.get("earnings_7d", {}).get("in_window", [])
    if earnings_7d_window:
        for ev in earnings_7d_window[:10]:
            tag = "(confirmed)" if ev.get("status") == "confirmed" else "(estimated)"
            lines.append(f"- {ev.get('display', '')}: {ev.get('company', '')} — earnings {tag}")
    else:
        lines.append("No upcoming earnings within 7 days.")

    if not decision_news_items and not earnings_7d_window:
        lines[-1] = "No relevant news or earnings this window."

    lines.extend(["", "## Watchlist"])

    ideas = result.get("ideas", [])
    if ideas:
        for idea in ideas[:5]:
            lines.append(f"- **{idea.get('ticker', '')}** — {idea.get('reason', '')[:120]}")
    else:
        lines.append("No new watchlist ideas met the validation threshold.")

    # Data quality flags
    data_flags = _collect_data_quality_flags(result, portfolio_rows)
    if data_flags:
        lines.extend(["", "## Data Quality Flags"])
        for flag in data_flags:
            lines.append(f"- {flag}")

    # Add AI Agent Attribution section
    lines.extend(_build_ai_attribution_section(result))
    
    return "\n".join(lines)


def _get_portfolio_stance(portfolio_rows: list[dict], recon_status: str) -> str:
    """Determine portfolio stance from canonical signals."""
    if recon_status == "FAIL":
        return "DEGRADED"
    buys = sum(1 for r in portfolio_rows if normalize_signal(r.get("signal", "HOLD")) == "BUY")
    sells = sum(1 for r in portfolio_rows if normalize_signal(r.get("signal", "HOLD")) == "SELL")
    if sells > 0 and sells >= buys:
        return "SELL"
    if buys > sells:
        return "BUY"
    return "HOLD"


def _get_stance_reason(portfolio_rows: list[dict], recon_status: str) -> str:
    if recon_status == "FAIL":
        return "Reconciliation FAIL — review data quality before new deployments."
    buys = [r for r in portfolio_rows if normalize_signal(r.get("signal", "HOLD")) == "BUY"]
    sells = [r for r in portfolio_rows if normalize_signal(r.get("signal", "HOLD")) == "SELL"]
    if sells and len(sells) >= len(buys):
        return f"{len(sells)} position(s) flagged for reduction."
    if buys:
        return f"{len(buys)} position(s) flagged for accumulation."
    return "No strong directional signals; maintain current allocation."


def _collect_flagged_positions(
    portfolio_rows: list[dict],
    monitoring_items: list[dict],
    result: dict,
) -> list[dict]:
    """Collect positions that need attention, with reasons."""
    flagged = []
    monitoring_by_display = {m["display"]: m for m in monitoring_items}

    for r in portfolio_rows:
        if not isinstance(r, dict):
            continue
        disp = str(r.get("display_symbol") or r.get("ticker") or "").strip().upper()
        if not disp:
            continue

        reasons = []
        weight = float(r.get("weight", 0) or 0)
        pnl_pct = float(r.get("pnl_pct", 0) or 0)
        signal = normalize_signal(r.get("signal", "HOLD"))

        if weight >= 5.0:
            reasons.append(f"concentration {weight:.1f}%")
        if pnl_pct <= -8.0 or pnl_pct >= 15.0:
            reasons.append(f"P&L {pnl_pct:+.1f}% outside band")
        if signal == "SELL":
            reasons.append("canonical SELL")
        elif signal == "BUY":
            reasons.append("canonical BUY")

        # RSI check from monitoring items
        mon = monitoring_by_display.get(disp)
        if mon and "RSI" in mon.get("triggers", ""):
            reasons.append("RSI extreme")

        # Reconciliation FAIL with held position
        if result.get("reconciliation", {}).get("status") == "FAIL" and signal == "HOLD":
            reasons.append("reconciliation FAIL")

        if reasons:
            company = r.get("company") or r.get("company_name") or disp
            flagged.append({
                "symbol": disp,
                "company": company,
                "reason": "; ".join(reasons),
            })

    return flagged


def _collect_data_quality_flags(result: dict, portfolio_rows: list[dict]) -> list[str]:
    """Collect data quality flags for the brief."""
    flags = []

    # Reconciliation FAIL
    recon_status = str((result.get("reconciliation") or {}).get("status", "")).upper()
    if recon_status == "FAIL":
        delta = (result.get("reconciliation") or {}).get("reconciliation_delta_eur", 0)
        flags.append(f"Reconciliation FAIL: delta €{delta:+.2f} exceeds tolerance")

    # Missing blocked cash field
    t212_data = result.get("t212_data", {})
    cash = (t212_data.get("cash", {}) or {})
    if "blocked" not in cash and cash.get("blocked") is None:
        flags.append("Blocked cash field missing from T212 API response")

    # Mapping suspects
    portfolio_rows = portfolio_rows or []
    for r in portfolio_rows:
        mapping_status = r.get("external_mapping_status", "")
        if mapping_status == "MAPPING_SUSPECT":
            disp = r.get("display_symbol") or r.get("ticker") or "UNKNOWN"
            flags.append(f"Mapping suspect for {disp}: {r.get('mapping_reason', 'unverified cross-listing')}")

    # Quantity/value jumps (would need previous run comparison)
    # This is a placeholder - would need historical comparison
    # if _detect_unexplained_jumps(portfolio_rows):
    #     flags.append("Unexplained quantity/value jump detected")

    return flags


def _build_ai_attribution_section(result: dict) -> list[str]:
    """Build the AI Agent Attribution section showing which model did what."""
    lines = ["", "## AI Agent Attribution & Sources"]
    
    # Get model info from result
    ai_recs = result.get("ai_recommendations", {}) or {}
    model_info = ai_recs.get("model_info", {}) or {}
    
    # Determine which model was used for each stage
    decision_model = model_info.get("decision_model", "deterministic_fallback")
    summary_model = model_info.get("summary_model", "deterministic_fallback")
    discovery_model = model_info.get("discovery_model", "deterministic_fallback")
    decision_detail_model = model_info.get("decision_detail_model", "deterministic_fallback")
    
    # Python-sourced calculations
    lines = [
        "",
        "### Decision & Reasoning",
        f"- **Decision/Recommendation**: {_format_model_name(decision_model)}",
        f"- **Reasoning/Thinking**: {_format_model_name(decision_detail_model)} (finr1 for math, gpt-oss/qwen for reasoning)",
        f"- **Summary Generation**: {_format_model_name(summary_model)} (gemma with fallback)",
        f"- **Discovery/New Ideas**: {_format_model_name(discovery_model)}",
        "",
        "### Analyst Recommendations & External Sources",
    ]
    
    # Add analyst consensus if available
    analyst_consensus = result.get("analyst_consensus", {})
    if analyst_consensus:
        lines.append(f"- **Analyst Consensus**: {analyst_consensus.get('summary', 'Available')}")
    
    # Python-sourced calculations
    lines.extend([
        "",
        "### Python-Sourced Calculations (Deterministic)",
        "- **Reconciliation & Valuation**: Python (broker-first, no LLM)",
        "- **Technical Indicators (RSI, MACD, SMA, ATR, Support/Resistance)**: Python via yfinance/Playwright",
        "- **Position Valuation & P&L**: Python (broker-first EUR valuation)",
        "- **Reconciliation Delta & Thresholds**: Python (strict tolerance)",
        "- **Position Weights & Concentration**: Python",
        "- **Earnings Dates**: Python (yfinance + calendar classification)",
        "",
        "### Model Fallback Chain",
        "- **Primary**: LM Studio (local, gpt-oss-20b / qwen3.8-9b / gemma4-12b)",
        "- **Fallback 1**: llama.cpp (Raspberry Pi, qwen3.8-9b)",
        "- **Fallback 2**: Ollama (Raspberry Pi, qwen3.8-9b-pi / qwen3-4b-pi / noema-2b)",
        "- **Fallback 3**: Gemini (free tier, gemini-2.5-flash)",
        "- **Fallback 4**: Mistral (free tier, open-mistral-nemo)",
        "- **Fallback 5**: Deterministic Python (always available)",
    ])
    
    return lines


def _format_model_name(model: str) -> str:
    """Format model name for display."""
    if not model or model == "deterministic_fallback":
        return "Python (deterministic)"
    model_lower = model.lower()
    if "gpt-oss" in model_lower:
        return f"gpt-oss (OpenAI OSS)"
    elif "qwen" in model_lower:
        return f"qwen (Alibaba)"
    elif "gemma" in model_lower:
        return f"gemma (Google)"
    elif "mistral" in model_lower:
        return f"mistral (Mistral AI)"
    elif "gemini" in model_lower:
        return f"gemini (Google)"
    elif "finr1" in model_lower:
        return "finr1 (math reasoning)"
    elif "qwen" in model_lower and ("3.8" in model_lower or "9b" in model_lower):
        return f"qwen3.8-9b (Alibaba)"
    else:
        return model

def _collect_data_quality_flags(result: dict, portfolio_rows: list[dict]) -> list[str]:
    """Collect data quality flags for the brief."""
    flags = []

    # Reconciliation FAIL
    recon_status = str((result.get("reconciliation") or {}).get("status", "")).upper()
    if recon_status == "FAIL":
        delta = (result.get("reconciliation") or {}).get("reconciliation_delta_eur", 0)
        flags.append(f"Reconciliation FAIL: delta €{delta:+.2f} exceeds tolerance")

    # Missing blocked cash field
    t212_data = result.get("t212_data", {})
    cash = (t212_data.get("cash", {}) or {})
    if "blocked" not in cash and cash.get("blocked") is None:
        flags.append("Blocked cash field missing from T212 API response")

    # Mapping suspects
    portfolio_rows = portfolio_rows or []
    for r in portfolio_rows:
        mapping_status = r.get("external_mapping_status", "")
        if mapping_status == "MAPPING_SUSPECT":
            disp = r.get("display_symbol") or r.get("ticker") or "UNKNOWN"
            flags.append(f"Mapping suspect for {disp}: {r.get('mapping_reason', 'unverified cross-listing')}")

    # Quantity/value jumps (would need previous run comparison)
    # This is a placeholder - would need historical comparison
    # if _detect_unexplained_jumps(portfolio_rows):
    #     flags.append("Unexplained quantity/value jump detected")

    return flags


def write_human_brief(brief_content: str, output_dir: Path = Path("reports/summary")) -> Path:
    """Write human brief to summary folder, overwrite each run, archive copy."""
    output_dir = Path("reports") / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    brief_path = output_dir / "portfolio_brief.md"
    brief_path.write_text(brief_content, encoding="utf-8")

    # Archive copy
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = archive_dir / f"portfolio_brief_{timestamp}.md"
    shutil.copy2(brief_path, archive_path)

    logger.info("Human brief written: %s", brief_path)
    logger.info("Archive copy: %s", archive_path)

    return brief_path