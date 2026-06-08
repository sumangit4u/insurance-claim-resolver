"""Unit tests for agent/tools/claim_tools.py.

Run with: pytest tests/test_claim_tools.py -v

Tests are intentionally lightweight (Week 0 scaffold).
Full integration tests added from Week 3 onwards.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CLAIMS = {
    "CLM-2024-001": {
        "claim_id": "CLM-2024-001",
        "policy_number": "POL-MOTOR-001",
        "claim_type": "motor",
        "status": "INTAKE",
        "claimed_amount": 180000,
        "incident_date": "2024-03-15",
        "policy_start_date": "2024-01-01",
        "hitl_required": False,
        "fraud_score": None,
    },
    "CLM-2024-002": {
        "claim_id": "CLM-2024-002",
        "policy_number": "POL-HEALTH-002",
        "claim_type": "health",
        "status": "TRIAGE",
        "claimed_amount": 95000,
        "incident_date": "2024-03-20",
        "policy_start_date": "2023-06-01",
        "hitl_required": False,
        "fraud_score": None,
    },
}


@pytest.fixture
def claims_file(tmp_path: Path):
    """Write sample claims to a temp JSON file and patch the module path."""
    claims_path = tmp_path / "claim_records.json"
    claims_path.write_text(json.dumps(SAMPLE_CLAIMS), encoding="utf-8")
    with patch("agent.tools.claim_tools._CLAIMS_FILE", claims_path):
        yield claims_path


# ---------------------------------------------------------------------------
# Tests: get_claim_status
# ---------------------------------------------------------------------------

def test_get_claim_status_found(claims_file):
    from agent.tools.claim_tools import get_claim_status

    result = json.loads(get_claim_status.invoke({"claim_id": "CLM-2024-001"}))
    assert result["claim_id"] == "CLM-2024-001"
    assert result["status"] == "INTAKE"
    assert result["claim_type"] == "motor"
    # PII should NOT be present
    assert "policy_holder_name" not in result
    assert "aadhaar_number" not in result


def test_get_claim_status_not_found(claims_file):
    from agent.tools.claim_tools import get_claim_status

    result = json.loads(get_claim_status.invoke({"claim_id": "CLM-XXXX"}))
    assert "error" in result


# ---------------------------------------------------------------------------
# Tests: update_claim_status
# ---------------------------------------------------------------------------

def test_update_claim_status_valid(claims_file):
    from agent.tools.claim_tools import update_claim_status

    result = json.loads(update_claim_status.invoke({
        "claim_id": "CLM-2024-001",
        "new_status": "TRIAGE",
        "reason": "Initial intake complete, moving to triage.",
    }))
    assert result["new_status"] == "TRIAGE"
    assert result["previous_status"] == "INTAKE"

    # Verify persisted
    saved = json.loads(claims_file.read_text())
    assert saved["CLM-2024-001"]["status"] == "TRIAGE"
    assert len(saved["CLM-2024-001"]["audit_trail"]) == 1


def test_update_claim_status_invalid_transition(claims_file):
    from agent.tools.claim_tools import update_claim_status

    result = json.loads(update_claim_status.invoke({
        "claim_id": "CLM-2024-001",
        "new_status": "CLOSED",          # INTAKE → CLOSED is illegal
        "reason": "Trying illegal skip",
    }))
    assert "error" in result
    assert "Illegal transition" in result["error"]


# ---------------------------------------------------------------------------
# Tests: escalate_claim
# ---------------------------------------------------------------------------

def test_escalate_claim_fraud_review(claims_file):
    from agent.tools.claim_tools import escalate_claim, update_claim_status

    # First advance to INVESTIGATION (valid path: INTAKE→TRIAGE→COVERAGE_CHECK→INVESTIGATION)
    for step in ["TRIAGE", "COVERAGE_CHECK", "INVESTIGATION"]:
        update_claim_status.invoke({
            "claim_id": "CLM-2024-001",
            "new_status": step,
            "reason": f"Advancing to {step}",
        })

    result = json.loads(escalate_claim.invoke({
        "claim_id": "CLM-2024-001",
        "gate": "FRAUD_REVIEW",
        "reason": "Fraud score 0.75 exceeds threshold",
    }))
    assert result["escalated_to"] == "FRAUD_REVIEW"


def test_escalate_claim_invalid_gate(claims_file):
    from agent.tools.claim_tools import escalate_claim

    result = json.loads(escalate_claim.invoke({
        "claim_id": "CLM-2024-001",
        "gate": "INVALID_GATE",
        "reason": "Test",
    }))
    assert "error" in result


# ---------------------------------------------------------------------------
# Tests: assess_fraud_risk
# ---------------------------------------------------------------------------

def test_assess_fraud_risk_low(claims_file):
    """CLM-2024-002: no red flags, should score low."""
    from agent.tools.claim_tools import assess_fraud_risk

    result = json.loads(assess_fraud_risk.invoke({"claim_id": "CLM-2024-002"}))
    assert 0.0 <= result["fraud_score"] <= 1.0
    assert result["recommendation"] in {"approve", "investigate", "escalate"}


def test_assess_fraud_risk_high_amount(claims_file):
    """Inject a high claimed amount and verify fraud flag is raised."""
    import json as _json
    saved = _json.loads(claims_file.read_text())
    saved["CLM-2024-001"]["claimed_amount"] = 1_000_000   # 4× typical for motor
    claims_file.write_text(_json.dumps(saved))

    from agent.tools.claim_tools import assess_fraud_risk
    result = _json.loads(assess_fraud_risk.invoke({"claim_id": "CLM-2024-001"}))
    assert result["fraud_score"] > 0.0
    assert len(result["red_flags"]) > 0


# ---------------------------------------------------------------------------
# Tests: request_missing_document
# ---------------------------------------------------------------------------

def test_request_missing_document(claims_file):
    from agent.tools.claim_tools import request_missing_document

    result = json.loads(request_missing_document.invoke({
        "claim_id": "CLM-2024-001",
        "document_type": "FIR copy",
        "reason": "Motor accident claim requires police report",
    }))
    assert result["document_requested"] == "FIR copy"

    saved = json.loads(claims_file.read_text())
    assert len(saved["CLM-2024-001"]["document_requests"]) == 1
