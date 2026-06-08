"""5 Domain LLM-as-Judge Evaluators — Week 8 deliverable.

Judges (calibrated to Cohen's kappa >= 0.60 against human labels):
    1. CoverageAccuracyJudge   — did the agent correctly determine coverage?
    2. CitationQualityJudge    — are policy citations verbatim and relevant?
    3. EscalationCorrectnessJudge — were HITL gates triggered at the right time?
    4. CompletenessJudge       — does the response address all aspects of the claim?
    5. SafetyJudge             — does the response contain PII or harmful content?

Each judge:
    - Uses the same Pydantic schema pattern as rag_utils.RetrievalGrade
    - Returns a structured verdict with score, reason, and confidence
    - Is tested with Cohen's kappa >= 0.60 against 100 human-labelled examples (Week 8)

RAGAS integration:
    - Existing RAGAS metrics (faithfulness, answer_relevancy, context_precision) run
      alongside these domain judges
    - Week 8 adds RAGAS custom metrics wrapping each judge

Week 8 deliverables built on top of this skeleton:
    - calibration_run(): compute Cohen's kappa vs human labels
    - batch_evaluate(): run all 5 judges over a dataset
    - ragas_suite(): combined RAGAS + domain judge evaluation
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config.settings import get_settings


# ---------------------------------------------------------------------------
# Verdict schemas (same pattern as rag_utils.py grader schemas)
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    """Base verdict returned by all judges."""
    score: float = Field(ge=0.0, le=1.0, description="0 = worst, 1 = best")
    passed: bool = Field(description="True if score meets the passing threshold")
    reason: str = Field(description="One-sentence justification for the score")
    confidence: float = Field(ge=0.0, le=1.0, description="Judge confidence in this verdict")


class CoverageAccuracyVerdict(JudgeVerdict):
    correct_outcome: Literal["covered", "excluded", "partial", "unknown"]
    agent_outcome: Literal["covered", "excluded", "partial", "unknown"]


class CitationQualityVerdict(JudgeVerdict):
    citations_found: int
    citations_verbatim: int
    citations_relevant: int


class EscalationVerdict(JudgeVerdict):
    should_have_escalated: bool
    did_escalate: bool
    gate: str | None


class CompletenessVerdict(JudgeVerdict):
    missing_aspects: List[str]


class SafetyVerdict(JudgeVerdict):
    pii_detected: bool
    harmful_content: bool
    pii_fields_found: List[str]


# ---------------------------------------------------------------------------
# Base judge class
# ---------------------------------------------------------------------------

class BaseJudge:
    """Common infrastructure for all LLM-as-Judge evaluators."""

    PASSING_THRESHOLD = 0.7   # Override per judge

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        settings = get_settings()
        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.0,
            google_api_key=settings.google_api_key or None,
        )


# ---------------------------------------------------------------------------
# Judge 1: Coverage Accuracy
# ---------------------------------------------------------------------------

class CoverageAccuracyJudge(BaseJudge):
    """Did the agent correctly determine coverage per the policy?

    Calibration target: kappa >= 0.60 vs 50 human-labelled coverage decisions.
    Week 8: run calibration_run() to measure and record kappa.
    """

    PASSING_THRESHOLD = 0.8

    PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert insurance adjudicator evaluating AI claim decisions.\n"
         "Given the claim details, retrieved policy excerpts, and the agent's coverage decision,\n"
         "judge whether the coverage determination is correct.\n\n"
         "Return JSON: {{\n"
         "  \"correct_outcome\": \"covered|excluded|partial|unknown\",\n"
         "  \"agent_outcome\": \"covered|excluded|partial|unknown\",\n"
         "  \"score\": 0.0-1.0,\n"
         "  \"passed\": true|false,\n"
         "  \"reason\": \"one sentence\",\n"
         "  \"confidence\": 0.0-1.0\n"
         "}}"),
        ("human",
         "Claim: {claim_description}\n\n"
         "Policy excerpts:\n{policy_context}\n\n"
         "Agent decision: {agent_decision}\n\n"
         "Ground truth (if available): {ground_truth}"),
    ])

    def evaluate(
        self,
        claim_description: str,
        policy_context: str,
        agent_decision: str,
        ground_truth: str = "unknown",
    ) -> CoverageAccuracyVerdict:
        """Week 8: implement full LLM call + structured output parsing."""
        # Stub: return placeholder until Week 8
        return CoverageAccuracyVerdict(
            score=0.0,
            passed=False,
            reason="Week 8 stub — not yet implemented",
            confidence=0.0,
            correct_outcome="unknown",
            agent_outcome="unknown",
        )


# ---------------------------------------------------------------------------
# Judge 2: Citation Quality
# ---------------------------------------------------------------------------

class CitationQualityJudge(BaseJudge):
    """Are policy citations verbatim, correctly referenced, and relevant?

    Core requirement: hallucination rate < 2%. Every decision must cite a clause.
    Calibration target: kappa >= 0.60 vs 50 human-reviewed citation assessments.
    """

    PASSING_THRESHOLD = 0.85   # Higher bar: citations are a hard client requirement

    def evaluate(
        self,
        agent_response: str,
        policy_context: str,
        expected_citations: List[str],
    ) -> CitationQualityVerdict:
        """Week 8: parse citations from agent_response, verify against policy_context."""
        return CitationQualityVerdict(
            score=0.0,
            passed=False,
            reason="Week 8 stub — not yet implemented",
            confidence=0.0,
            citations_found=0,
            citations_verbatim=0,
            citations_relevant=0,
        )


# ---------------------------------------------------------------------------
# Judge 3: Escalation Correctness
# ---------------------------------------------------------------------------

class EscalationCorrectnessJudge(BaseJudge):
    """Were HITL gates triggered at the correct times?

    Rules (from regulatory requirements):
    - CONFLICT_REVIEW: triggered iff evidence is contradictory
    - FRAUD_REVIEW: triggered iff fraud_score > 0.7
    - APPROVAL: triggered iff settlement > Rs 5,00,000
    Calibration target: kappa >= 0.60 vs 50 human-reviewed escalation decisions.
    """

    PASSING_THRESHOLD = 0.9   # Near-perfect required; missed escalation is a compliance risk

    def evaluate(
        self,
        claim_state: Dict[str, Any],
        escalation_events: List[Dict[str, Any]],
    ) -> EscalationVerdict:
        """Week 8: rule-based check with LLM for ambiguous conflict cases."""
        fraud_score = claim_state.get("fraud_result", {}).get("fraud_score", 0)
        settlement = claim_state.get("settlement_amount", 0)
        conflict = claim_state.get("coverage_result", {}).get("conflict_detected", False)

        should_escalate_fraud = fraud_score > 0.7
        should_escalate_approval = settlement > 500_000
        did_escalate = len(escalation_events) > 0

        # Simple rule check (Week 8: LLM handles ambiguous conflict detection)
        should_have_escalated = should_escalate_fraud or should_escalate_approval or conflict
        score = 1.0 if (should_have_escalated == did_escalate) else 0.0

        return EscalationVerdict(
            score=score,
            passed=score >= self.PASSING_THRESHOLD,
            reason="Rule-based check only. LLM judgment added in Week 8.",
            confidence=0.9 if score == 1.0 else 0.5,
            should_have_escalated=should_have_escalated,
            did_escalate=did_escalate,
            gate=escalation_events[0].get("gate") if escalation_events else None,
        )


# ---------------------------------------------------------------------------
# Judge 4: Completeness
# ---------------------------------------------------------------------------

class CompletenessJudge(BaseJudge):
    """Does the agent response address all required aspects of the claim?

    Required aspects for any claim decision:
    1. Coverage determination (yes/no/partial)
    2. Policy clause citation
    3. Settlement amount or rejection reason
    4. Next action / claimant communication
    Calibration target: kappa >= 0.60 vs 50 human-reviewed responses.
    """

    PASSING_THRESHOLD = 0.75

    REQUIRED_ASPECTS = [
        "coverage_determination",
        "policy_citation",
        "settlement_or_rejection",
        "next_action",
    ]

    def evaluate(self, agent_response: str, claim_type: str) -> CompletenessVerdict:
        """Week 8: LLM checks for presence of each required aspect."""
        return CompletenessVerdict(
            score=0.0,
            passed=False,
            reason="Week 8 stub — not yet implemented",
            confidence=0.0,
            missing_aspects=self.REQUIRED_ASPECTS,
        )


# ---------------------------------------------------------------------------
# Judge 5: Safety (PII + harmful content)
# ---------------------------------------------------------------------------

class SafetyJudge(BaseJudge):
    """Does the response contain PII or harmful content?

    PII fields to detect (NEVER appear in agent responses):
    - policy_holder_name, aadhaar_number, pan_number, phone, email, address

    This judge runs on EVERY response — it is not optional.
    Failing this judge blocks the response from being returned to the caller.
    """

    PASSING_THRESHOLD = 1.0   # Zero tolerance: any PII = fail

    PII_PATTERNS = [
        r"\b\d{12}\b",                      # Aadhaar (12 digits)
        r"\b[A-Z]{5}\d{4}[A-Z]\b",          # PAN
        r"\b[6-9]\d{9}\b",                  # Indian mobile
        r"\b[\w.+-]+@[\w-]+\.\w+\b",        # Email
    ]

    def evaluate(self, agent_response: str) -> SafetyVerdict:
        """Rule-based PII detection. LLM-assisted semantic check added in Week 8."""
        import re
        pii_found = []
        for pattern in self.PII_PATTERNS:
            if re.search(pattern, agent_response):
                pii_found.append(pattern)

        safe = len(pii_found) == 0
        return SafetyVerdict(
            score=1.0 if safe else 0.0,
            passed=safe,
            reason="No PII detected." if safe else f"PII pattern match: {pii_found}",
            confidence=0.95,
            pii_detected=not safe,
            harmful_content=False,   # LLM check in Week 8
            pii_fields_found=pii_found,
        )


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------

ALL_JUDGES = [
    CoverageAccuracyJudge,
    CitationQualityJudge,
    EscalationCorrectnessJudge,
    CompletenessJudge,
    SafetyJudge,
]


def run_safety_check(agent_response: str) -> SafetyVerdict:
    """Convenience: run the SafetyJudge on any response before returning it."""
    return SafetyJudge().evaluate(agent_response)
