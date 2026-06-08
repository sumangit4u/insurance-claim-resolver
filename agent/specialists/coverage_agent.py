"""Coverage Specialist Agent — Week 5 deliverable.

Responsibilities:
    - Run RAG lookup against the policy corpus for coverage questions
    - Generate structured PolicyCitation objects for each retrieved clause
    - Determine covered / excluded / partial with confidence score
    - Store citations in ClaimState.policy_citations for audit trail

Tools available: check_policy_coverage (→ rag/retriever.py in Week 1)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools.claim_tools import check_policy_coverage
from config.settings import get_settings
from rag.retriever import PolicyCitation

COVERAGE_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a policy coverage specialist. Given the claim details and retrieved policy "
     "excerpts, determine:\n"
     "1. Whether the claim is COVERED, EXCLUDED, or PARTIAL\n"
     "2. The applicable coverage limit\n"
     "3. Any exclusions that apply\n"
     "4. Verbatim clause reference for each determination\n\n"
     "You MUST cite specific policy sections. Never answer without a citation."),
    ("human",
     "Claim: {claim_description}\nClaim type: {claim_type}\n\nRetrieved policy context:\n{context}"),
])


class CoverageAgent:
    """Specialist agent for policy coverage determination.

    Week 5: implement with full RAG integration and LLM analysis.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        settings = get_settings()
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.0,
            google_api_key=settings.google_api_key or None,
        )

    def check_coverage(
        self,
        claim_id: str,
        claim_description: str,
        claim_type: str,
    ) -> Dict[str, Any]:
        """Determine coverage for a claim.

        Returns dict with: covered, coverage_limit, exclusions, citations, confidence.
        """
        # Step 1: RAG lookup
        rag_result = json.loads(check_policy_coverage.invoke({
            "claim_id": claim_id,
            "query": f"{claim_type} claim: {claim_description}",
        }))

        # Step 2: LLM analysis (Week 5: wire COVERAGE_AGENT_PROMPT | self.llm)
        return {
            "claim_id": claim_id,
            "covered": None,           # True | False | "partial"
            "coverage_limit": None,
            "exclusions": [],
            "citations": [],           # list[PolicyCitation] in Week 5
            "confidence": 0.0,
            "rag_status": rag_result.get("status"),
            "note": "Wire RAG + LLM in Week 5 using COVERAGE_AGENT_PROMPT.",
        }
