"""Pydantic schemas for AI recommendation validation."""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal


class PortfolioAction(BaseModel):
    """Portfolio-level action recommendation."""
    action: Literal["hold", "reduce", "increase", "rotate"]
    asset: str = Field(..., min_length=1, max_length=20)
    reason: str = Field(..., min_length=10, max_length=500)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"hold", "reduce", "increase", "rotate"}
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v


class TradeRecommendation(BaseModel):
    """Specific trade recommendation."""
    side: Literal["BUY", "SELL"]
    asset: str = Field(..., min_length=1, max_length=20)
    qty: float = Field(ge=0.0001, le=1000000)
    price: float = Field(ge=0.01, le=1000000)
    stop_loss: float = Field(ge=0.01, le=1000000)
    take_profit: float = Field(ge=0.01, le=1000000)
    risk_pct: float = Field(ge=0.1, le=2.0, description="Risk as percentage of portfolio")
    reason: str = Field(..., min_length=10, max_length=500)

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        if v not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        return v

    @model_validator(mode="after")
    def validate_risk_reward(self):
        if self.side == "BUY":
            if self.take_profit <= self.price:
                raise ValueError("take_profit must be > price for BUY")
            if self.stop_loss >= self.price:
                raise ValueError("stop_loss must be < price for BUY")
            risk = self.price - self.stop_loss
            reward = self.take_profit - self.price
            if reward / risk < 2.0:
                raise ValueError("Risk:Reward must be at least 1:2")
        else:  # SELL
            if self.stop_loss <= self.price:
                raise ValueError("stop_loss must be > price for SELL")
            if self.take_profit >= self.price:
                raise ValueError("take_profit must be < price for SELL")
            risk = self.stop_loss - self.price
            reward = self.price - self.take_profit
            if reward / risk < 2.0:
                raise ValueError("Risk:Reward must be at least 1:2")
        return self

    @field_validator("risk_pct")
    @classmethod
    def validate_risk_pct(cls, v: float) -> float:
        if not (0.1 <= v <= 2.0):
            raise ValueError("risk_pct must be between 0.1 and 2.0")
        return v


class PositionUpdate(BaseModel):
    """Position management update (stop loss / take profit adjustment)."""
    asset: str = Field(..., min_length=1, max_length=20)
    stop_loss: float = Field(ge=0.01, le=1000000)
    new_stop: float = Field(ge=0.01, le=1000000)
    take_profit: float = Field(ge=0.01, le=1000000)
    new_tp: float = Field(ge=0.01, le=1000000)
    reason: str = Field(..., min_length=10, max_length=500)


class CashDeploymentTarget(BaseModel):
    """Target for cash deployment."""
    asset: str = Field(..., min_length=1, max_length= 20)
    amount: float = Field(ge=1.0, le=1000000.0)
    price: float = Field(ge=0.01, le=1000000)
    max_risk: float = Field(ge=0.1, le=2.0)


class CashDeployment(BaseModel):
    """Cash deployment plan."""
    free_cash: float = Field(ge=0.0)
    deploy_amount: float = Field(ge=0.0)
    deploy_pct: float = Field(ge=0.0, le=100.0)
    reserve: float = Field(ge=0.0)
    targets: List[Any] = Field(default_factory=list)

    @field_validator("deploy_pct")
    @classmethod
    def validate_deploy_pct(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError("deploy_pct must be between 0 and 100")
        return v


class AIRecommendations(BaseModel):
    """Complete AI recommendations response."""
    portfolio_actions: List[Any] = Field(default_factory=list)
    trades: List[Any] = Field(default_factory=list)
    position_updates: List[Any] = Field(default_factory=list)
    cash_deployment: Any = None  # Will be validated in model_validator

    @model_validator(mode="after")
    def validate_cash_deployment(self):
        if self.cash_deployment is not None:
            if not isinstance(self.cash_deployment, dict):
                raise ValueError("cash_deployment must be a dict")
            # We'll validate the structure in the CashDeployment model
        return self

    class Config:
        extra = "forbid"  # Reject any extra fields


def parse_ai_recommendations(json_str: str) -> dict:
    """
    Parse and validate AI recommendations JSON.
    
    Returns a dict with either:
    - Validated AIRecommendations object (if successful)
    - {"error": "...", "raw_response": "..."} if validation fails
    """
    import json
    
    if not json_str or not json_str.strip():
        return {"error": "Empty response from model", "raw_response": ""}
    
    # Clean up the response
    cleaned = json_str.strip()
    
    # Remove markdown code fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Strip <answer> tags if present
    if cleaned.startswith("<answer>") and cleaned.endswith("</answer>"):
        cleaned = cleaned[8:-9].strip()
    
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    # Strip <answer> tags
    if cleaned.startswith("<answer>") and cleaned.endswith("</answer>"):
        cleaned = cleaned[8:-9].strip()
    
    if not cleaned.strip():
        return {"error": "Empty response after cleaning", "raw_response": ""}
    
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "raw_response": cleaned[:500]}
    
    # Validate against schema
    try:
        from pydantic import ValidationError
        validated = AIRecommendations.model_validate(data)
        # Convert to dict for compatibility
        return validated.model_dump(mode="json")
    except Exception as e:
        return {"error": f"Schema validation failed: {e}", "raw_response": json_str[:500] if 'json_str' in locals() else "no response"}


def validate_ai_recommendations(ai_recs: dict) -> dict:
    """
    Validate and sanitize AI recommendations.
    Returns validated dict or error dict.
    """
    if not ai_recs:
        return {"error": "Empty recommendations", "raw_response": ""}
    
    if isinstance(ai_recs, dict) and ai_recs.get("error"):
        return ai_recs  # Already an error dict
    
    # Check if it's a dict with error key
    if isinstance(ai_recs, dict) and ai_recs.get("error"):
        return ai_recs
    
    # Validate against schema
    try:
        from pydantic import ValidationError
        validated = AIRecommendations.model_validate(ai_recs)
        return validated.model_dump(mode="json")
    except Exception as e:
        return {"error": f"Schema validation failed: {e}", "raw_response": str(ai_recs)[:500]}


def sanitize_ai_output(text: str) -> str:
    """
    Strip chain-of-thought, reasoning tags, markdown, and other non-JSON content.
    Returns clean JSON string or empty string if nothing valid found.
    """
    if not text:
        return ""
    
    content = str(text).strip()
    
    # Strip <answer> tags
    if content.startswith("<answer>") and content.endswith("</answer>"):
        content = content[8:-9].strip()
    
    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    
    # Strip <answer> tags
    if content.startswith("<answer>") and content.endswith("</answer>"):
        content = content[8:-9].strip()
    
    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    
    # Remove common reasoning prefixes that models sometimes include
    prefixes_to_strip = [
        "```json", "```json\n", "```",
        "Here is the JSON:", "Here is the JSON:", "Here's the JSON:",
        "```json\n", "```\n"
    ]
    for prefix in prefixes_to_strip:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
    
    # Strip trailing ```
    if content.endswith("```"):
        content = content[:-3].strip()
    
    # Find first { and last } to extract JSON object
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        content = content[first_brace:last_brace+1]
    
    return content.strip()