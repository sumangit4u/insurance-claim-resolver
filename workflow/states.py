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


# ---------------------------------------------------------------------------
# Week 0 demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 60
    print("=" * W)
    print(" Insurance Claims — 9-State Workflow (Week 0 Demo)")
    print("=" * W)
    print()
    print("States (in order):")
    print("  INTAKE → TRIAGE → COVERAGE_CHECK → [CONFLICT_REVIEW *]")
    print("  → INVESTIGATION → [FRAUD_REVIEW *] → DECISION → [APPROVAL *] → CLOSED")
    print("  (* = HITL gate: workflow pauses for mandatory human review)")
    print()

    print("Valid transitions:")
    ordered = [
        ClaimStatus.INTAKE, ClaimStatus.TRIAGE, ClaimStatus.COVERAGE_CHECK,
        ClaimStatus.CONFLICT_REVIEW, ClaimStatus.INVESTIGATION, ClaimStatus.FRAUD_REVIEW,
        ClaimStatus.DECISION, ClaimStatus.APPROVAL, ClaimStatus.CLOSED,
    ]
    for state in ordered:
        targets = TRANSITIONS[state]
        if targets:
            target_str = " | ".join(t.value for t in sorted(targets, key=lambda x: x.value))
            hitl_tag = "  [HITL gate]" if requires_hitl(state) else ""
            print(f"  {state.value:<20} → {target_str}{hitl_tag}")
        else:
            print(f"  {state.value:<20} → (terminal)")

    print()
    print("Transition guard tests:")
    tests = [
        (ClaimStatus.INTAKE,       ClaimStatus.TRIAGE,        True,  "valid"),
        (ClaimStatus.INTAKE,       ClaimStatus.CLOSED,         False, "BLOCKED (illegal skip)"),
        (ClaimStatus.INVESTIGATION, ClaimStatus.FRAUD_REVIEW,  True,  "valid"),
        (ClaimStatus.DECISION,     ClaimStatus.INTAKE,         False, "BLOCKED (no going back)"),
    ]
    for current, target, expected, label in tests:
        result = can_transition(current, target)
        icon = "✓" if result else "✗"
        print(f"  {icon} {current.value} → {target.value:<20} {label}")

    print()
    print("ClaimState key fields:")
    fields = [
        ("claim_id", "str"),
        ("claim_status", "ClaimStatus value (9 states)"),
        ("claimed_amount", "float (Rs)"),
        ("fraud_result", "dict — from assess_fraud_risk tool"),
        ("policy_citations", "list[str] — every decision cited"),
        ("audit_trail", "list[dict] — immutable transition log"),
        ("hitl_required", "bool — True when paused for human"),
    ]
    for name, desc in fields:
        print(f"  • {name:<20} {desc}")
