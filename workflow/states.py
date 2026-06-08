"""Claims workflow state definitions.

9-state lifecycle:
    INTAKE → TRIAGE → COVERAGE_CHECK → [CONFLICT_REVIEW] →
    INVESTIGATION → [FRAUD_REVIEW] → DECISION → APPROVAL → CLOSED

Pattern: adapted from inventra/agents/coordinator.py AgentState TypedDict.
"""
from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class ClaimStatus(str, Enum):
    """9-state claims lifecycle enum."""
    INTAKE = "INTAKE"
    TRIAGE = "TRIAGE"
    COVERAGE_CHECK = "COVERAGE_CHECK"
    CONFLICT_REVIEW = "CONFLICT_REVIEW"   # HITL gate 1
    INVESTIGATION = "INVESTIGATION"
    FRAUD_REVIEW = "FRAUD_REVIEW"         # HITL gate 2
    DECISION = "DECISION"
    APPROVAL = "APPROVAL"                 # HITL gate 3
    CLOSED = "CLOSED"


# Valid transitions: state → set of allowed next states
TRANSITIONS: dict[ClaimStatus, set[ClaimStatus]] = {
    ClaimStatus.INTAKE:          {ClaimStatus.TRIAGE},
    ClaimStatus.TRIAGE:          {ClaimStatus.COVERAGE_CHECK},
    ClaimStatus.COVERAGE_CHECK:  {ClaimStatus.CONFLICT_REVIEW, ClaimStatus.INVESTIGATION},
    ClaimStatus.CONFLICT_REVIEW: {ClaimStatus.INVESTIGATION},
    ClaimStatus.INVESTIGATION:   {ClaimStatus.FRAUD_REVIEW, ClaimStatus.DECISION},
    ClaimStatus.FRAUD_REVIEW:    {ClaimStatus.DECISION},
    ClaimStatus.DECISION:        {ClaimStatus.APPROVAL, ClaimStatus.CLOSED},
    ClaimStatus.APPROVAL:        {ClaimStatus.CLOSED},
    ClaimStatus.CLOSED:          set(),
}

# States that require a human in the loop before proceeding
HITL_STATES: set[ClaimStatus] = {
    ClaimStatus.CONFLICT_REVIEW,
    ClaimStatus.FRAUD_REVIEW,
    ClaimStatus.APPROVAL,
}


class ClaimState(TypedDict):
    """LangGraph-compatible state for the claims workflow.

    Mirrors AgentState in inventra/agents/coordinator.py.
    messages uses operator.add so each node appends rather than replaces.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Claim identifiers
    claim_id: str
    policy_number: str
    claim_type: str              # motor | health | property

    # Workflow state
    claim_status: str            # ClaimStatus value
    previous_status: Optional[str]

    # Extracted claim data
    claimed_amount: float
    incident_date: str
    incident_description: str

    # Agent outputs
    coverage_result: Dict[str, Any]      # from coverage_check tool
    fraud_result: Dict[str, Any]         # from fraud_assessment tool
    investigation_notes: str
    settlement_amount: float

    # Grounding / audit
    policy_citations: list[str]          # clause references used in decisions
    audit_trail: list[Dict[str, Any]]    # immutable log of each state transition

    # HITL
    hitl_required: bool
    hitl_decision: Optional[str]         # approve | reject | more_info
    hitl_reviewer: Optional[str]

    # Final
    final_response: str


def can_transition(current: ClaimStatus, target: ClaimStatus) -> bool:
    """Return True if the state transition is valid."""
    return target in TRANSITIONS.get(current, set())


def requires_hitl(status: ClaimStatus) -> bool:
    """Return True if this state requires human review before continuing."""
    return status in HITL_STATES
