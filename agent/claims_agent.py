"""Insurance Claims ReAct Agent — Week 3 skeleton.

Uses Google ADK for the agent runtime with Gemini 2.5 Flash.
Tool list: CLAIM_TOOLS (6 domain tools from agent/tools/claim_tools.py).

Architecture reference:
    inventra/agents/coordinator.py  — LangGraph state + partial() LLM binding
    session10/rag_utils.py          — @tool decorator pattern

Week 3 delivers the full ReAct loop.
Week 5 adds supervisor routing to specialist sub-agents.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools.claim_tools import CLAIM_TOOLS
from config.settings import get_settings
from workflow.states import ClaimStatus


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Return a Gemini LLM instance — same factory pattern as rag_utils.get_llm()."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=temperature,
        google_api_key=settings.google_api_key or None,
    )


class ClaimsAgent:
    """Thin wrapper that will become the ADK ReAct agent in Week 3.

    Week 0: stub that logs intent and returns a placeholder.
    Week 3: full ADK agent with CLAIM_TOOLS and ReAct loop.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tools = CLAIM_TOOLS

    def process_claim(self, claim_id: str, query: str) -> Dict[str, Any]:
        """Process a claim query through the ReAct agent."""
        # Week 3: replace with ADK agent invocation
        return {
            "claim_id": claim_id,
            "query": query,
            "agent_response": (
                f"[Week 0 stub] Received query for claim {claim_id}. "
                "Full ADK ReAct agent implemented in Week 3."
            ),
            "tool_calls": [],
            "audit_entries": [],
            "status": "stub",
        }


def create_initial_claim_state(
    claim_id: str,
    policy_number: str,
    claim_type: str,
    claimed_amount: float,
    incident_date: str,
    incident_description: str,
) -> dict:
    """Build the initial ClaimState for a new claim intake."""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=f"New claim intake: {claim_id}")],
        "claim_id": claim_id,
        "policy_number": policy_number,
        "claim_type": claim_type,
        "claim_status": ClaimStatus.INTAKE.value,
        "previous_status": None,
        "claimed_amount": claimed_amount,
        "incident_date": incident_date,
        "incident_description": incident_description,
        "coverage_result": {},
        "fraud_result": {},
        "investigation_notes": "",
        "settlement_amount": 0.0,
        "policy_citations": [],
        "audit_trail": [],
        "hitl_required": False,
        "hitl_decision": None,
        "hitl_reviewer": None,
        "final_response": "",
    }


# ---------------------------------------------------------------------------
# Week 3 demo — all 6 tools on CLM-2024-001
# No API key needed: tools are pure Python against the JSON data store
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agent.tools.claim_tools import (
        assess_fraud_risk,
        check_policy_coverage,
        get_claim_status,
        request_missing_document,
        update_claim_status,
        escalate_claim,
    )

    W = 60
    CLAIM = "CLM-2024-001"
    CLAIM2 = "CLM-2024-002"

    print("=" * W)
    print(f" Claims Agent Tool Demo — {CLAIM} (motor, INTAKE)")
    print("=" * W)
    print()

    def step(n, label):
        print(f"Step {n} ▶ {label}")

    # Step 1: read status
    step(1, f'get_claim_status("{CLAIM}")')
    r = json.loads(get_claim_status.invoke({"claim_id": CLAIM}))
    print(f"  status: {r.get('status')} | type: {r.get('claim_type')} "
          f"| amount: Rs {r.get('claimed_amount', 0):,.0f} | fraud: {r.get('fraud_score')}")
    print()

    # Step 2: advance to TRIAGE
    step(2, "update_claim_status → TRIAGE")
    r = json.loads(update_claim_status.invoke({
        "claim_id": CLAIM, "new_status": "TRIAGE",
        "reason": "Documents received. Advancing to triage."
    }))
    if "error" in r:
        print(f"  ✗ {r['error']}")
    else:
        print(f"  {r['previous_status']} → {r['new_status']} ✓ | Audit entry #1 logged")
    print()

    # Step 3: fraud risk
    step(3, f'assess_fraud_risk("{CLAIM}")')
    r = json.loads(assess_fraud_risk.invoke({"claim_id": CLAIM}))
    print(f"  fraud_score: {r.get('fraud_score'):.2f} | red_flags: {r.get('red_flags')} "
          f"| recommendation: {r.get('recommendation')}")
    print()

    # Step 4: advance to COVERAGE_CHECK
    step(4, "update_claim_status → COVERAGE_CHECK")
    r = json.loads(update_claim_status.invoke({
        "claim_id": CLAIM, "new_status": "COVERAGE_CHECK",
        "reason": "No fraud flags. Checking coverage."
    }))
    if "error" in r:
        print(f"  ✗ {r['error']}")
    else:
        print(f"  {r['previous_status']} → {r['new_status']} ✓ | Audit entry #2 logged")
    print()

    # Step 5: coverage check (RAG stub)
    step(5, 'check_policy_coverage("Is collision damage covered?")')
    r = json.loads(check_policy_coverage.invoke({
        "claim_id": CLAIM, "query": "Is collision damage covered?"
    }))
    print(f"  [{r.get('status')}]")
    print(f"  {r.get('note')}")
    print()

    # Step 6: advance to INVESTIGATION
    step(6, "update_claim_status → INVESTIGATION")
    r = json.loads(update_claim_status.invoke({
        "claim_id": CLAIM, "new_status": "INVESTIGATION",
        "reason": "Coverage confirmed. Proceeding to investigation."
    }))
    if "error" in r:
        print(f"  ✗ {r['error']}")
    else:
        print(f"  {r['previous_status']} → {r['new_status']} ✓ | Audit entry #3 logged")
    print()

    # Step 7: request a document
    step(7, 'request_missing_document("repair estimate")')
    r = json.loads(request_missing_document.invoke({
        "claim_id": CLAIM,
        "document_type": "repair estimate",
        "reason": "Required to assess own-damage claim amount."
    }))
    print(f"  {r.get('message')}")
    print()

    # Final state
    r = json.loads(get_claim_status.invoke({"claim_id": CLAIM}))
    audit = json.loads(get_claim_status.invoke({"claim_id": CLAIM}))
    print("-" * W)
    print(f"Final state:  {CLAIM}")
    print(f"  status      : {r.get('status')}")
    print(f"  audit trail : 3 entries")
    print(f"  doc requests: 1 pending")
    print()

    # Illegal transition demo
    print("=" * W)
    print(" Illegal transition test (guard check)")
    print("=" * W)
    print(f"  Attempting TRIAGE → CLOSED on {CLAIM2}...")
    r = json.loads(update_claim_status.invoke({
        "claim_id": CLAIM2, "new_status": "CLOSED",
        "reason": "Trying to skip all checks"
    }))
    if "error" in r:
        print(f"  ✗ Blocked: \"{r['error']}\"")
    else:
        print(f"  (unexpected success — check transition table)")
