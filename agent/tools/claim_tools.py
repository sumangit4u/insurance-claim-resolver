"""Domain tools for the Insurance Claims ReAct agent.

Six tools exposed to the ADK agent:
    1. get_claim_status       — read current claim state
    2. update_claim_status    — advance to next state (validates transitions)
    3. escalate_claim         — trigger a HITL gate
    4. request_missing_document — ask claimant for docs
    5. check_policy_coverage  — RAG lookup against policy corpus
    6. assess_fraud_risk      — rule-based + LLM fraud scoring

Pattern: @tool decorator from langchain.tools, same as rag_utils.py HRAG_TOOLS.
PII rule: tools must never log or return raw PII fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from langchain.tools import tool

from workflow.states import ClaimStatus, can_transition

# ---------------------------------------------------------------------------
# In-memory claim store (Week 3 local dev; swap for Firestore when GCP ready)
# ---------------------------------------------------------------------------

_CLAIMS_FILE = Path(__file__).resolve().parents[2] / "data" / "claims" / "claim_records.json"


def _load_claims() -> Dict[str, Any]:
    if _CLAIMS_FILE.exists():
        return json.loads(_CLAIMS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_claims(data: Dict[str, Any]) -> None:
    _CLAIMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CLAIMS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool 1: get_claim_status
# ---------------------------------------------------------------------------

@tool
def get_claim_status(claim_id: str) -> str:
    """Return the current status and non-PII summary of a claim.

    Args:
        claim_id: Unique claim identifier (e.g. CLM-2024-001)

    Returns:
        JSON string with claim_id, status, claim_type, claimed_amount,
        incident_date, and hitl_required flag.
    """
    claims = _load_claims()
    claim = claims.get(claim_id)
    if not claim:
        return json.dumps({"error": f"Claim {claim_id} not found"})

    # Return only non-PII fields
    return json.dumps({
        "claim_id": claim_id,
        "status": claim.get("status"),
        "claim_type": claim.get("claim_type"),
        "claimed_amount": claim.get("claimed_amount"),
        "incident_date": claim.get("incident_date"),
        "hitl_required": claim.get("hitl_required", False),
        "fraud_score": claim.get("fraud_score"),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 2: update_claim_status
# ---------------------------------------------------------------------------

@tool
def update_claim_status(claim_id: str, new_status: str, reason: str) -> str:
    """Advance a claim to the next status in the 9-state workflow.

    Validates the transition is legal before applying it.
    Appends an audit entry with the reason.

    Args:
        claim_id: Unique claim identifier
        new_status: Target ClaimStatus value (e.g. TRIAGE, COVERAGE_CHECK)
        reason: One sentence justifying the transition (will be cited in audit)

    Returns:
        JSON string confirming the transition or describing the error.
    """
    claims = _load_claims()
    claim = claims.get(claim_id)
    if not claim:
        return json.dumps({"error": f"Claim {claim_id} not found"})

    try:
        current = ClaimStatus(claim["status"])
        target = ClaimStatus(new_status)
    except ValueError as e:
        return json.dumps({"error": f"Invalid status value: {e}"})

    if not can_transition(current, target):
        return json.dumps({
            "error": f"Illegal transition {current.value} → {target.value}"
        })

    import datetime
    claim["previous_status"] = claim["status"]
    claim["status"] = target.value
    claim.setdefault("audit_trail", []).append({
        "from": current.value,
        "to": target.value,
        "reason": reason,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    })
    _save_claims(claims)

    return json.dumps({
        "claim_id": claim_id,
        "previous_status": current.value,
        "new_status": target.value,
        "message": "Status updated successfully",
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 3: escalate_claim
# ---------------------------------------------------------------------------

@tool
def escalate_claim(claim_id: str, gate: str, reason: str) -> str:
    """Escalate a claim to a mandatory HITL gate.

    Valid gates: CONFLICT_REVIEW, FRAUD_REVIEW, APPROVAL

    Args:
        claim_id: Unique claim identifier
        gate: One of CONFLICT_REVIEW | FRAUD_REVIEW | APPROVAL
        reason: Specific reason for escalation (cited in audit trail)

    Returns:
        JSON confirmation with gate name and next steps for the human reviewer.
    """
    valid_gates = {"CONFLICT_REVIEW", "FRAUD_REVIEW", "APPROVAL"}
    if gate not in valid_gates:
        return json.dumps({"error": f"Invalid gate '{gate}'. Must be one of {valid_gates}"})

    result = update_claim_status.invoke({"claim_id": claim_id, "new_status": gate, "reason": reason})
    parsed = json.loads(result)
    if "error" in parsed:
        return result

    return json.dumps({
        "claim_id": claim_id,
        "escalated_to": gate,
        "reason": reason,
        "next_steps": f"Claim is now pending human review at {gate}. "
                      f"A reviewer will be notified via the HITL dashboard.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 4: request_missing_document
# ---------------------------------------------------------------------------

@tool
def request_missing_document(claim_id: str, document_type: str, reason: str) -> str:
    """Flag that a required document is missing and log a request for it.

    Does NOT send any communication directly — that is handled by the
    communication agent in Week 5.

    Args:
        claim_id: Unique claim identifier
        document_type: e.g. "FIR copy", "hospital discharge summary", "repair estimate"
        reason: Why this document is needed for the claim decision

    Returns:
        JSON confirmation that the document request has been logged.
    """
    claims = _load_claims()
    claim = claims.get(claim_id)
    if not claim:
        return json.dumps({"error": f"Claim {claim_id} not found"})

    import datetime
    request_entry = {
        "document_type": document_type,
        "reason": reason,
        "requested_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "pending",
    }
    claim.setdefault("document_requests", []).append(request_entry)
    _save_claims(claims)

    return json.dumps({
        "claim_id": claim_id,
        "document_requested": document_type,
        "reason": reason,
        "message": "Document request logged. Communication agent will notify claimant.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 5: check_policy_coverage
# ---------------------------------------------------------------------------

@tool
def check_policy_coverage(claim_id: str, query: str) -> str:
    """Query the policy RAG corpus to check coverage for a specific scenario.

    Uses the local Chroma vector store (Week 1-2). Upgraded to Vertex AI
    Search in Week 2 when gcp_ready=True.

    Args:
        claim_id: Used for audit logging only
        query: Natural language coverage question, e.g.
               "Is flood damage to basement covered under property policy?"

    Returns:
        JSON with retrieved policy excerpts and source citations.
    """
    # Week 1 placeholder — full RAG integration in rag/retriever.py
    return json.dumps({
        "claim_id": claim_id,
        "query": query,
        "status": "RAG retriever not yet initialised (Week 1 deliverable)",
        "citations": [],
        "note": "Integrate rag/retriever.py in Week 1 to populate this tool.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 6: assess_fraud_risk
# ---------------------------------------------------------------------------

@tool
def assess_fraud_risk(claim_id: str) -> str:
    """Run a fraud risk assessment on a claim.

    Checks rule-based red flags and (Week 5+) delegates to the fraud
    specialist agent for LLM-powered pattern analysis.

    Args:
        claim_id: Unique claim identifier

    Returns:
        JSON with fraud_score (0-1), red_flags list, and recommendation.
    """
    claims = _load_claims()
    claim = claims.get(claim_id)
    if not claim:
        return json.dumps({"error": f"Claim {claim_id} not found"})

    red_flags: list[str] = []
    fraud_score = 0.0

    # Rule 1: claim filed < 30 days after policy start
    import datetime
    policy_start = claim.get("policy_start_date", "")
    incident_date = claim.get("incident_date", "")
    if policy_start and incident_date:
        try:
            delta = (
                datetime.date.fromisoformat(incident_date) -
                datetime.date.fromisoformat(policy_start)
            ).days
            if delta < 30:
                red_flags.append(f"Claim filed only {delta} days after policy inception")
                fraud_score += 0.35
        except ValueError:
            pass

    # Rule 2: claimed amount > 2× typical for claim type
    claimed = claim.get("claimed_amount", 0)
    typical = {"motor": 250000, "health": 150000, "property": 500000}
    threshold = typical.get(claim.get("claim_type", ""), 200000)
    if claimed > 2 * threshold:
        red_flags.append(
            f"Claimed amount Rs {claimed:,} exceeds 2× typical ({threshold:,}) for {claim.get('claim_type')}"
        )
        fraud_score += 0.30

    fraud_score = min(fraud_score, 1.0)
    recommendation = (
        "escalate" if fraud_score > 0.7
        else "investigate" if fraud_score > 0.4
        else "approve"
    )

    # Persist fraud score
    claim["fraud_score"] = round(fraud_score, 2)
    _save_claims(claims)

    return json.dumps({
        "claim_id": claim_id,
        "fraud_score": round(fraud_score, 2),
        "red_flags": red_flags,
        "recommendation": recommendation,
        "note": "LLM-powered analysis added in Week 5 (fraud specialist agent)",
    }, indent=2)


# ---------------------------------------------------------------------------
# Exported tool list (same pattern as HRAG_TOOLS in rag_utils.py)
# ---------------------------------------------------------------------------

CLAIM_TOOLS = [
    get_claim_status,
    update_claim_status,
    escalate_claim,
    request_missing_document,
    check_policy_coverage,
    assess_fraud_risk,
]
