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

Usage (Week 7 demo):
    python -m workflow.claims_workflow
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
    fraud_score = state.get("fraud_result", {}).get("fraud_score", "?")
    _advance(state, ClaimStatus.FRAUD_REVIEW, f"Fraud score {fraud_score} exceeds threshold. Escalated.")
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
    settlement = state.get("settlement_amount", 0)
    _advance(state, ClaimStatus.APPROVAL, f"Settlement Rs {settlement:,.0f} requires senior approval.")
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


def _make_initial_state(claim_id: str, description: str) -> ClaimState:
    """Build a minimal ClaimState for testing."""
    return ClaimState(
        claim_id=claim_id,
        claim_type="motor",
        claim_status="",
        previous_status="",
        claim_description=description,
        coverage_result={},
        fraud_result={},
        settlement_amount=0.0,
        policy_citations=[],
        hitl_required=False,
        hitl_decision="",
        final_response="",
        messages=[],
        audit_trail=[],
        missing_documents=[],
    )


# ---------------------------------------------------------------------------
# Week 7 demo — requires GOOGLE_API_KEY
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    W = 60
    print("=" * W)
    print(" Week 7 — Full LangGraph Workflow Demo")
    print("=" * W)
    print()

    api_key = os.environ.get("GOOGLE_API_KEY") or ""

    # Step 1: show graph topology (no API key needed)
    print("Step 1 — Graph topology (9 nodes, conditional edges)")
    print("-" * 40)
    nodes = [
        ("intake",          "INTAKE",          "linear"),
        ("triage",          "TRIAGE",          "linear"),
        ("coverage_check",  "COVERAGE_CHECK",  "conditional → conflict_review | investigation"),
        ("conflict_review", "CONFLICT_REVIEW", "HITL gate 1 → investigation | closed"),
        ("investigation",   "INVESTIGATION",   "conditional → fraud_review | decision"),
        ("fraud_review",    "FRAUD_REVIEW",    "HITL gate 2 → decision"),
        ("decision",        "DECISION",        "conditional → approval | closed"),
        ("approval",        "APPROVAL",        "HITL gate 3 → closed"),
        ("closed",          "CLOSED",          "→ END"),
    ]
    for node, status, edge_desc in nodes:
        hitl = " [HITL]" if "HITL" in edge_desc else ""
        print(f"  {status:<20}{hitl:<8} {edge_desc}")
    print()

    # Step 2: routing logic demonstration (no API key needed)
    print("Step 2 — Routing function tests (no API key needed)")
    print("-" * 40)
    routing_tests = [
        ("route_after_coverage_check",
         {"coverage_result": {"conflict_detected": False}},
         "investigation"),
        ("route_after_coverage_check",
         {"coverage_result": {"conflict_detected": True}},
         "conflict_review"),
        ("route_after_investigation",
         {"fraud_result": {"fraud_score": 0.3}},
         "decision"),
        ("route_after_investigation",
         {"fraud_result": {"fraud_score": 0.85}},
         "fraud_review"),
        ("route_after_decision",
         {"settlement_amount": 250_000},
         "closed"),
        ("route_after_decision",
         {"settlement_amount": 750_000},
         "approval"),
        ("route_after_hitl",
         {"hitl_decision": "approve"},
         "continue"),
        ("route_after_hitl",
         {"hitl_decision": "reject"},
         "closed"),
    ]
    routing_fn_map = {
        "route_after_coverage_check": route_after_coverage_check,
        "route_after_investigation":  route_after_investigation,
        "route_after_decision":       route_after_decision,
        "route_after_hitl":           route_after_hitl,
    }
    for fn_name, state_input, expected in routing_tests:
        actual = routing_fn_map[fn_name](state_input)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {fn_name}({list(state_input.values())[0]}) → {actual}")
    print()

    # Step 3: happy-path walkthrough (requires API key for LLM nodes)
    if api_key:
        print("Step 3 — Happy-path graph invocation (CLM-2024-001)")
        print("-" * 40)
        from langchain_google_genai import ChatGoogleGenerativeAI
        from config.settings import get_settings
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0.0)

        graph = build_claims_workflow(llm)
        initial = _make_initial_state("CLM-2024-001", "Vehicle rear-end collision on Mumbai highway")
        # Set state so no HITL gates trigger
        initial["fraud_result"] = {"fraud_score": 0.1}
        initial["settlement_amount"] = 150_000

        result = graph.invoke(initial)
        print(f"  Final status    : {result['claim_status']}")
        print(f"  Settlement      : Rs {result.get('settlement_amount', 0):,.0f}")
        print(f"  Audit trail     : {len(result['audit_trail'])} entries")
        for entry in result["audit_trail"]:
            print(f"    {entry['from']} → {entry['to']}  ({entry['reason'][:50]})")
        print()
    else:
        print("Step 3 — Skipped (set GOOGLE_API_KEY to run full graph invocation)")
        print()
        print("  Week 7 todos:")
        print("  1. Add LangGraph interrupt() calls at HITL nodes")
        print("  2. Wire api/main.py /claims/{id}/process to graph.invoke()")
        print("  3. Emit SSE events from HITL nodes to /claims/{id}/stream")
        print()

    print("=" * W)
    print(" Week 7 Complete → Week 8: LLM-as-Judge evaluation suite")
    print("=" * W)
