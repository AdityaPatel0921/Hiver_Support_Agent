"""
src/schemas.py
Pydantic v2 schemas and Enums for the Apple Support AI triage & response system.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class IntentType(str, Enum):
    """Supported customer intent categories."""
    OS_SOFTWARE_UPDATE = "OS_SOFTWARE_UPDATE"
    BATTERY_POWER = "BATTERY_POWER"
    ACCOUNT_ICLOUD = "ACCOUNT_ICLOUD"
    HARDWARE_DAMAGE = "HARDWARE_DAMAGE"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"


class TriageAction(str, Enum):
    """Decision action for customer triage."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"


class TriageDecision(BaseModel):
    """Structured decision output from the triage reasoning step."""
    intent: IntentType = Field(
        ...,
        description="The primary categorized intent of the customer inquiry."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the triage decision between 0.0 and 1.0."
    )
    action: TriageAction = Field(
        ...,
        description="Whether to auto-handle the response or escalate to human senior support."
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Detailed mandatory reason if action is ESCALATE, otherwise None."
    )
    urgency_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Urgency score from 1 (lowest) to 5 (critical/safety emergency)."
    )

    @model_validator(mode="after")
    def validate_escalation_reason(self) -> "TriageDecision":
        """Enforce non-empty escalation reason when action is ESCALATE."""
        if self.action == TriageAction.ESCALATE:
            if not self.escalation_reason or not self.escalation_reason.strip():
                raise ValueError("`escalation_reason` must not be empty when `action` is ESCALATE.")
        return self


class AgentResponse(BaseModel):
    """Production-grade agent response schema enclosing triage decision and grounded reply."""
    decision: TriageDecision = Field(
        ...,
        description="Triage categorization, action, confidence, and urgency."
    )
    draft_reply: str = Field(
        ...,
        min_length=1,
        description="Grounded, closed-world draft reply adhering to Apple Support voice and guardrails."
    )
    retrieval_citations: List[int] = Field(
        default_factory=list,
        description="Indices of retrieved historical support pairs cited in the draft reply."
    )


if __name__ == "__main__":
    # Self-test schema validation
    sample_decision = TriageDecision(
        intent=IntentType.BATTERY_POWER,
        confidence=0.95,
        action=TriageAction.AUTO_HANDLE,
        urgency_score=2
    )
    sample_response = AgentResponse(
        decision=sample_decision,
        draft_reply="We understand your battery concerns. Check Battery Health in Settings > Battery. [Apple Support Link]",
        retrieval_citations=[0, 1]
    )
    print("Schema validation successful:")
    print(sample_response.model_dump_json(indent=2))
