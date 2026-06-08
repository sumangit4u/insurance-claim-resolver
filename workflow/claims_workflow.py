"""9-state Insurance Claims Workflow — LangGraph orchestration.

State machine:
    INTAKE → TRIAGE → COVERAGE_CHECK → [CONFLICT_REVIEW] →
    INVESTIGATION → [FRAUD_REVIEW] → DECISION → APPROVAL → CLOSED

HITL gates (mandatory, regulatory requirement):
    CONFLICT_REVIEW  — contradictory evidence detected
    FRAUD_REVIEW     — fraud_score > 0.7
    APPROVAL         — settlement > Rs 5,00,000

Pattern: adapted from inventra/agents/coordinator.py StateGraph.
Week 3: intake → triage → coverage_check → investigation → decision nodes.
Week 7: full HITL gates with SSE event emission + human_review interrupt nodes.

Usage (Week 3+):
    from workflow.claims_workflow import build_claims_workflow
    graph = build_claims_workflow(llm)
    result = graph.invoke(initial_state)
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from workflow.states import ClaimState, ClaimStatus, can_transition, requires_hitl


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_intake(state: ClaimState) -> ClaimState:
    """INTAKE: validate claim fields and log arrival."""
    state["claim_status"] = ClaimStatus.INTAKE.value
    state["audit_trail"].append({
        "from": "START",
        "to": ClaimStatus.INTAKE.value,
        "reason": "Claim received and logged.",
        "timestamp": _now(),
    })
    state["messages"] = list(state["messages"]) + [
        AIMessage(content=f"Claim {state['claim_id']} intake complete.")
    ]
    return state


def node_triage(state: ClaimState) -> ClaimState:
    """TRIAGE: verify documents received and classify claim type."""
    _advance(state, ClaimStatus.TRIAGE, "Document check passed. Claim classified.")
    return state


def node_coverage_check(state: ClaimState, llm: ChatGoogleGenerativeAI) -> ClaimState:
    """COVERAGE_CHECK: RAG lookup to confirm policy coverage.

    Week 1-2: stubs with placeholder.
    Week 3+:  calls check_policy_coverage tool via ADK agent.
    """
    _advance(state, ClaimStatus.COVERAGE_CHECK, "Coverage check initiated via RAG.")
    # Week 3: integrate rag/retriever.py here
    state["coverage_result"] = {
        "status": "pending",
        "note": "Full RAG integration in Week 3.",
    }
    return state


def node_conflict_review(state: ClaimState) -> ClaimState:
    """CONFLICT_REVIEW (HITL gate 1): pause for human review of conflicting evidence.

    Week 7: emits SSE 'hitl_gate' event; graph suspends until hitl_decision set.
    """
    _advance(state, ClaimStatus.CONFLICT_REVIEW, "Conflicting evidence detected. Awaiting human review.")
    state["hitl_required"] = True
    # Week 7: emit SSE event here
    # await sse_emitter.emit("hitl_gate", {"gate": "CONFLICT_REVIEW", "claim_id": state["claim_id"]})
    return state


def node_investigation(state: ClaimState) -> ClaimState:
    """INVESTIGATION: agent gathers supporting evidence and runs fraud screening."""
    _advance(state, ClaimStatus.INVESTIGATION, "Investigation phase started.")
    # Week 3+: ADK agent runs assess_fraud_risk tool here
    return state


def node_fraud_review(state: ClaimState) -> ClaimState:
    """FRAUD_REVIEW (HITL gate 2): pause for specialist fraud review.

    Week 7: suspends graph until human specialist provides hitl_decision.
    """
    _advance(state, ClaimStatus.FRAUD_REVIEW, f"Fraud score {state.get('fraud_result', {}).get('fraud_score', '?')} exceeds threshold. Escalated.")
    state["hitl_required"] = True
    # Week 7: emit SSE event
    return state


def node_decision(state: ClaimState, llm: ChatGoogleGenerativeAI) -> ClaimState:
    """DECISION: LLM synthesises coverage + fraud results into settlement recommendation."""
    _advance(state, ClaimStatus.DECISION, "Settlement decision computed.")
    # Week 3+: call LLM with coverage_result + fraud_result to compute settlement_amount
    # Week 8: LLM-as-Judge evaluates the decision before advancing
    return state


def node_approval(state: ClaimState) -> ClaimState:
    """APPROVAL (HITL gate 3): mandatory human sign-off for settlements > Rs 5,00,000.

    Week 7: suspends graph; sends approval request to senior adjudicator via SSE.
    """
    _advance(state, ClaimStatus.APPROVAL, f"Settlement Rs {state.get('settlement_amount', 0):,.0f} requires senior approval.")
    state["hitl_required"] = True
    # Week 7: emit SSE event + notify reviewer
    return state


def node_closed(state: ClaimState) -> ClaimState:
    """CLOSED: finalize claim, disburse settlement, update records."""
    _advance(state, ClaimStatus.CLOSED, "Claim closed. Settlement disbursed.")
    state["hitl_required"] = False
    state["final_response"] = (
        f"Claim {state['claim_id']} closed. "
        f"Settlement: Rs {state.get('settlement_amount', 0):,.0f}. "
        f"Citations: {', '.join(state.get('policy_citations', ['none']))}"
    )
    return state


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_coverage_check(state: ClaimState) -> str:
    """After coverage check: conflict detected → CONFLICT_REVIEW, else INVESTIGATION."""
    coverage = state.get("coverage_result", {})
    if coverage.get("conflict_detected"):
        return "conflict_review"
    return "investigation"


def route_after_investigation(state: ClaimState) -> str:
    """After investigation: high fraud → FRAUD_REVIEW, else DECISION."""
    fraud = state.get("fraud_result", {})
    if fraud.get("fraud_score", 0) > 0.7:
        return "fraud_review"
    return "decision"


def route_after_decision(state: ClaimState) -> str:
    """After decision: large settlement → APPROVAL gate, else CLOSED."""
    if state.get("settlement_amount", 0) > 500_000:
        return "approval"
    return "closed"


def route_after_hitl(state: ClaimState) -> str:
    """After any HITL gate: check human decision.

    Week 7: hitl_decision set by the HITL dashboard API endpoint.
    """
    decision = state.get("hitl_decision", "approve")
    if decision == "approve":
        return "continue"
    elif decision == "reject":
        return "closed"
    else:  # more_info
        return "investigation"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_claims_workflow(llm: ChatGoogleGenerativeAI) -> Any:
    """Build and compile the 9-state claims StateGraph.

    Args:
        llm: Gemini LLM instance for decision and coverage nodes

    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(ClaimState)

    # Bind LLM to nodes that need it (partial application — same as coordinator.py)
    coverage_with_llm = partial(node_coverage_check, llm=llm)
    decision_with_llm = partial(node_decision, llm=llm)

    # Register nodes
    workflow.add_node("intake", node_intake)
    workflow.add_node("triage", node_triage)
    workflow.add_node("coverage_check", coverage_with_llm)
    workflow.add_node("conflict_review", node_conflict_review)
    workflow.add_node("investigation", node_investigation)
    workflow.add_node("fraud_review", node_fraud_review)
    workflow.add_node("decision", decision_with_llm)
    workflow.add_node("approval", node_approval)
    workflow.add_node("closed", node_closed)

    # Linear edges
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "triage")
    workflow.add_edge("triage", "coverage_check")
    workflow.add_edge("fraud_review", "decision")    # after human clears fraud review

    # Conditional edges
    workflow.add_conditional_edges(
        "coverage_check",
        route_after_coverage_check,
        {"conflict_review": "conflict_review", "investigation": "investigation"},
    )
    workflow.add_conditional_edges(
        "conflict_review",
        route_after_hitl,
        {"continue": "investigation", "closed": "closed", "investigation": "investigation"},
    )
    workflow.add_conditional_edges(
        "investigation",
        route_after_investigation,
        {"fraud_review": "fraud_review", "decision": "decision"},
    )
    workflow.add_conditional_edges(
        "decision",
        route_after_decision,
        {"approval": "approval", "closed": "closed"},
    )
    workflow.add_conditional_edges(
        "approval",
        route_after_hitl,
        {"continue": "closed", "closed": "closed", "investigation": "investigation"},
    )
    workflow.add_edge("closed", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _advance(state: ClaimState, target: ClaimStatus, reason: str) -> None:
    """Update status and append audit entry in-place."""
    prev = state.get("claim_status", "START")
    state["previous_status"] = prev
    state["claim_status"] = target.value
    state.setdefault("audit_trail", []).append({
        "from": prev,
        "to": target.value,
        "reason": reason,
        "timestamp": _now(),
    })
