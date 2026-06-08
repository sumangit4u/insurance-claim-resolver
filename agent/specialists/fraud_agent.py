"""Fraud Specialist Agent — Week 5 deliverable.

Responsibilities:
    - Run LLM-powered fraud pattern analysis (beyond rule-based Week 3 checks)
    - Interpret fraud_score + red_flags and form a narrative recommendation
    - Call escalate_claim when fraud_score > 0.7
    - Cite specific fraud indicators in the audit trail

Tools available to this agent: assess_fraud_risk, escalate_claim
"""
from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools.claim_tools import assess_fraud_risk, escalate_claim
from config.settings import get_settings

FRAUD_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fraud detection specialist with 15 years of insurance claims experience. "
     "Analyse the fraud assessment result for this claim and provide:\n"
     "1. A clear narrative of what the red flags indicate\n"
     "2. Whether to approve, investigate further, or escalate to FRAUD_REVIEW\n"
     "3. The specific policy or regulatory basis for your recommendation\n\n"
     "If fraud_score > 0.7, you MUST recommend escalation to FRAUD_REVIEW.\n"
     "Cite specific red flags in your reasoning."),
    ("human", "Claim ID: {claim_id}\nFraud assessment: {fraud_result}"),
])

ESCALATION_THRESHOLD = 0.7


class FraudAgent:
    """Specialist agent for fraud detection and escalation.

    Week 5: implement with full ADK agent loop.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        settings = get_settings()
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.0,
            google_api_key=settings.google_api_key or None,
        )

    def analyse(self, claim_id: str) -> Dict[str, Any]:
        """Run fraud analysis on a claim and escalate if needed.

        Returns dict with: fraud_score, recommendation, analysis, escalated.
        """
        # Step 1: get fraud assessment from tool
        fraud_result = json.loads(assess_fraud_risk.invoke({"claim_id": claim_id}))

        if "error" in fraud_result:
            return {"error": fraud_result["error"]}

        fraud_score = fraud_result.get("fraud_score", 0.0)
        escalated = False

        # Step 2: LLM narrative analysis (Week 5: call self.llm here)
        # chain = FRAUD_AGENT_PROMPT | self.llm
        # analysis = chain.invoke({"claim_id": claim_id, "fraud_result": json.dumps(fraud_result)}).content
        analysis = (
            f"[Week 5 stub] fraud_score={fraud_score}. "
            f"Red flags: {fraud_result.get('red_flags', [])}. "
            f"Recommendation: {fraud_result.get('recommendation')}. "
            "Wire LLM analysis in Week 5."
        )

        # Step 3: escalate if needed
        if fraud_score > ESCALATION_THRESHOLD:
            esc_result = json.loads(escalate_claim.invoke({
                "claim_id": claim_id,
                "gate": "FRAUD_REVIEW",
                "reason": f"Fraud score {fraud_score:.2f} exceeds threshold {ESCALATION_THRESHOLD}. "
                          f"Red flags: {', '.join(fraud_result.get('red_flags', []))}",
            }))
            escalated = "error" not in esc_result

        return {
            "claim_id": claim_id,
            "fraud_score": fraud_score,
            "red_flags": fraud_result.get("red_flags", []),
            "recommendation": fraud_result.get("recommendation"),
            "analysis": analysis,
            "escalated": escalated,
        }
