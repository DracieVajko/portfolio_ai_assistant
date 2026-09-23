from __future__ import annotations

from pathlib import Path
from typing import Dict


_PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a prompt template from disk without hardcoding prompts in Python."""
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt_bundle(symbol: str) -> Dict[str, str]:
    """Assemble the multi-stage prompt bundle for the pipeline."""
    base_context = {
        "symbol": symbol,
    }
    bundle = {}
    for stage in ("research", "think", "adrian_alpha", "summary", "constraints", "style"):
        prompt = load_prompt(stage)
        bundle[stage] = prompt.format(**base_context)

    bundle["research"] = "\n\n".join(
        [bundle["research"], bundle["constraints"], bundle["style"]]
    )
    bundle["think"] = "\n\n".join(
        [bundle["think"], bundle["constraints"], bundle["style"]]
    )
    bundle["adrian_alpha"] = "\n\n".join(
        [bundle["adrian_alpha"], bundle["constraints"], bundle["style"]]
    )
    bundle["summary"] = "\n\n".join(
        [bundle["summary"], bundle["constraints"], bundle["style"]]
    )
    return bundle
