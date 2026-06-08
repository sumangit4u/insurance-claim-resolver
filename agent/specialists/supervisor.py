"""Multi-Agent Supervisor — Week 5 deliverable.

Coordinator pattern (adapted from inventra/agents/coordinator.py):
- Supervisor receives claim state from the main workflow
- Routes to the appropriate specialist agent based on current stage
- Aggregates results and updates claim state

Specialists coordinated:
    FraudAgent         — triggered at INVESTIGATION / FRAUD_REVIEW stage
    CoverageAgent      — triggered at COVERAGE_CHECK / CONFLICT_REVIEW stage
    CommunicationAgent — triggered whenever missing_documents is non-empty

Week 5 deliverables built on this skeleton:
    - Full LLM-based routing (replacing rule-based switch)
    - Parallel specialist invocation where independent
    - Supervisor synthesises multi-agent output into a single claim state update

Usage:
    python -m agent.specialists.supervisor
"""
from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.specialists.communication_agent import CommunicationAgent
from agent.specialists.coverage_agent import CoverageAgent
from agent.specialists.fraud_agent import FraudAgent
from config.settings import get_settings


class ClaimsSupervisor:
    """Multi-agent supervisor that routes to domain specialists.

    Pattern: adapted from inventra/agents/coordinator.py supervisor node.
    The supervisor receives the full ClaimState and decides which specialist(s)
    to invoke based on the current claim_status and pending tasks.

    Week 5 full implementation:
    - LLM-based routing with structured tool selection
    - Parallel specialist invocation where independent
    - Result aggregation with citation preservation
    """

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        settings = get_settings()
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.0,
            google_api_key=settings.google_api_key or None,
        )
        self.fraud_agent = FraudAgent(self.llm)
        self.coverage_agent = CoverageAgent(self.llm)
        self.comm_agent = CommunicationAgent(self.llm)

    def route(self, claim_state: dict[str, Any]) -> str:
        """Rule-based routing — Week 5: replace with LLM-based tool selection.

        Returns:
            Specialist name to invoke: 'fraud', 'coverage', 'communication', 'done'
        """
        status = claim_state.get("claim_status", "")
        missing_docs = claim_state.get("missing_documents", [])

        if status in ("INVESTIGATION", "FRAUD_REVIEW"):
            return "fraud"
        elif status in ("COVERAGE_CHECK", "CONFLICT_REVIEW"):
            return "coverage"
        elif missing_docs:
            return "communication"
        return "done"

    def run(self, claim_state: dict[str, Any]) -> dict[str, Any]:
        """Invoke the appropriate specialist and return updated claim state.

        Args:
            claim_state: Current ClaimState dict

        Returns:
            Updated claim state with specialist results merged in
        """
        specialist = self.route(claim_state)
        claim_id = claim_state.get("claim_id", "UNKNOWN")

        if specialist == "fraud":
            result = self.fraud_agent.assess(claim_id)
            claim_state["fraud_result"] = result
            claim_state.setdefault("audit_trail", []).append({
                "agent": "FraudAgent",
                "result": result,
            })

        elif specialist == "coverage":
            query = claim_state.get("claim_description", "policy coverage check")
            result = self.coverage_agent.check_coverage(claim_id, query)
            claim_state["coverage_result"] = result
            claim_state.setdefault("audit_trail", []).append({
                "agent": "CoverageAgent",
                "result": result,
            })

        elif specialist == "communication":
            missing = claim_state.get("missing_documents", [])
            result = self.comm_agent.request_documents(
                claim_id, missing, reason="Required to process your claim"
            )
            claim_state.setdefault("audit_trail", []).append({
                "agent": "CommunicationAgent",
                "result": result,
            })

        else:
            claim_state.setdefault("audit_trail", []).append({
                "agent": "Supervisor",
                "result": "No specialist required at this stage.",
            })

        return claim_state


# ---------------------------------------------------------------------------
# Week 5 demo — no API key required (all stub mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 60
    print("=" * W)
    print(" Week 5 — Multi-Agent Supervisor Demo")
    print("=" * W)
    print()

    scenarios = [
        {
            "name": "Fraud routing (INVESTIGATION stage)",
            "state": {
                "claim_id": "CLM-2024-001",
                "claim_status": "INVESTIGATION",
                "claim_description": "Vehicle collision on highway",
                "audit_trail": [],
                "missing_documents": [],
            },
        },
        {
            "name": "Coverage routing (COVERAGE_CHECK stage)",
            "state": {
                "claim_id": "CLM-2024-002",
                "claim_status": "COVERAGE_CHECK",
                "claim_description": "Hospitalisation for cardiac surgery",
                "audit_trail": [],
                "missing_documents": [],
            },
        },
        {
            "name": "Communication routing (missing documents)",
            "state": {
                "claim_id": "CLM-2024-003",
                "claim_status": "TRIAGE",
                "claim_description": "Property water damage",
                "audit_trail": [],
                "missing_documents": ["FIR Copy", "Repair Estimate", "Photos"],
            },
        },
        {
            "name": "No specialist needed (INTAKE stage, no missing docs)",
            "state": {
                "claim_id": "CLM-2024-004",
                "claim_status": "INTAKE",
                "claim_description": "New claim submitted",
                "audit_trail": [],
                "missing_documents": [],
            },
        },
    ]

    # Use a lightweight stub supervisor (no LLM, no real agent init)
    class _StubSupervisor:
        """Stub that only tests the route() logic — no real tool calls."""
        def route(self, claim_state: dict[str, Any]) -> str:
            status = claim_state.get("claim_status", "")
            missing_docs = claim_state.get("missing_documents", [])
            if status in ("INVESTIGATION", "FRAUD_REVIEW"):
                return "fraud"
            elif status in ("COVERAGE_CHECK", "CONFLICT_REVIEW"):
                return "coverage"
            elif missing_docs:
                return "communication"
            return "done"

    stub = _StubSupervisor()

    for scenario in scenarios:
        state = scenario["state"]
        routed_to = stub.route(state)
        print(f"Scenario : {scenario['name']}")
        print(f"  Claim   : {state['claim_id']}  |  Status: {state['claim_status']}")
        print(f"  Missing : {state['missing_documents'] or 'none'}")
        print(f"  Routed  → {routed_to.upper()} specialist")
        print()

    print("=" * W)
    print(" Routing Rules (Week 5: replace with LLM tool-selection)")
    print("=" * W)
    routing_rules = [
        ("INVESTIGATION / FRAUD_REVIEW",      "FraudAgent"),
        ("COVERAGE_CHECK / CONFLICT_REVIEW",  "CoverageAgent"),
        ("Any stage + missing_documents",     "CommunicationAgent"),
        ("All other stages",                  "done (no specialist)"),
    ]
    for trigger, action in routing_rules:
        print(f"  {trigger:<42} → {action}")

    print()
    print("Week 5 todos:")
    print("  1. Replace rule-based route() with LLM structured-output tool selection")
    print("  2. Add parallel invocation for fraud + coverage when both needed")
    print("  3. Wire ClaimsSupervisor into claims_workflow.py INVESTIGATION node")
    print("=" * W)
