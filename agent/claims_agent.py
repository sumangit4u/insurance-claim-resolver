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

import os
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools.claim_tools import CLAIM_TOOLS
from config.settings import get_settings
from workflow.states import ClaimState, ClaimStatus, create_initial_claim_state


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Return a Gemini LLM instance — same factory pattern as rag_utils.get_llm()."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=temperature,
        google_api_key=settings.google_api_key or None,
    )


# ---------------------------------------------------------------------------
# ADK agent entry point (Week 3: replace stub with real ADK agent)
# ---------------------------------------------------------------------------

class ClaimsAgent:
    """Thin wrapper that will become the ADK ReAct agent in Week 3.

    Week 0: stub that logs intent and returns a placeholder.
    Week 3: full ADK agent with CLAIM_TOOLS and ReAct loop.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tools = CLAIM_TOOLS
        # ADK agent initialisation goes here in Week 3
        # self.agent = adk.Agent(tools=CLAIM_TOOLS, model=..., system_prompt=...)

    def process_claim(self, claim_id: str, query: str) -> Dict[str, Any]:
        """Process a claim query through the ReAct agent.

        Args:
            claim_id: The claim to process
            query: Natural language instruction from adjudicator or automated pipeline

        Returns:
            Dict with agent_response, tool_calls, and audit_entries
        """
        # Week 3: replace with ADK agent invocation
        # result = self.agent.run(f"Claim {claim_id}: {query}")
        return {
            "claim_id": claim_id,
            "query": query,
            "agent_response": (
                f"[Week 0 stub] Received query for claim {claim_id}. "
                "Full ADK ReAct agent will be implemented in Week 3."
            ),
            "tool_calls": [],
            "audit_entries": [],
            "status": "stub",
        }


# ---------------------------------------------------------------------------
# Helper: initial state factory (delegates to workflow.states)
# ---------------------------------------------------------------------------

def create_initial_claim_state(
    claim_id: str,
    policy_number: str,
    claim_type: str,
    claimed_amount: float,
    incident_date: str,
    incident_description: str,
) -> ClaimState:
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
