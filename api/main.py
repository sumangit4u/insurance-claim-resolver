"""FastAPI application — Insurance Claims Resolution API.

Endpoints:
    POST /claims/{claim_id}/process  — Submit a query to the claims agent
    GET  /claims/{claim_id}/status   — Get current claim status
    GET  /claims/{claim_id}/stream   — SSE stream of agent reasoning (Week 7)
    GET  /health                     — Health check

Week 7 delivers full SSE streaming with HITL gate events.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.claims_agent import ClaimsAgent
from agent.tools.claim_tools import get_claim_status
from config.settings import get_settings

settings = get_settings()
app = FastAPI(
    title="Insurance Claims Resolution Agent",
    description="AI-powered claims processing with mandatory HITL gates",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production (Week 10)
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = ClaimsAgent()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ClaimQuery(BaseModel):
    query: str
    session_id: str | None = None


class ClaimResponse(BaseModel):
    claim_id: str
    agent_response: str
    status: str
    tool_calls: list[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check — always returns 200 if the app is running."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "gcp_ready": settings.gcp_ready,
        "version": "0.1.0",
    }


@app.get("/claims/{claim_id}/status")
async def get_status(claim_id: str) -> Dict[str, Any]:
    """Return current claim status (non-PII fields only)."""
    import json
    result = get_claim_status.invoke({"claim_id": claim_id})
    parsed = json.loads(result)
    if "error" in parsed:
        raise HTTPException(status_code=404, detail=parsed["error"])
    return parsed


@app.post("/claims/{claim_id}/process")
async def process_claim(claim_id: str, body: ClaimQuery) -> ClaimResponse:
    """Submit a query to the claims agent for processing."""
    result = _agent.process_claim(claim_id=claim_id, query=body.query)
    return ClaimResponse(
        claim_id=claim_id,
        agent_response=result["agent_response"],
        status=result["status"],
        tool_calls=result["tool_calls"],
    )


@app.get("/claims/{claim_id}/stream")
async def stream_claim(claim_id: str):
    """SSE stream of agent reasoning steps — implemented in Week 7.

    Returns a placeholder until the full HITL + SSE pipeline is built.
    """
    from fastapi.responses import StreamingResponse

    async def event_generator():
        yield f"data: {{\"event\": \"connected\", \"claim_id\": \"{claim_id}\"}}\n\n"
        yield "data: {\"event\": \"placeholder\", \"message\": \"Full SSE streaming in Week 7\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
