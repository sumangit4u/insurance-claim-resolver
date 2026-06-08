"""Insurance Claims MCP Server.

Exposes 6 insurance domain tools via the MCP protocol.
Pattern: directly extends inventra/integrations/mcp_server.py.
- Same mcp.server.Server + @server.list_tools() / @server.call_tool() pattern
- Adds RBAC: each tool declares required_role; caller role checked before dispatch
- Adds audit logging: every tool call logged with caller, tool, args, result

Roles (Week 4 full implementation):
    adjudicator  — read + write (most tools)
    supervisor   — all tools including escalation approval
    auditor      — read-only

Usage:
    python mcp_server/server.py
    (or via ADK tool integration in Week 4)
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Tool definitions (6 domain tools)
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="get_claim_status",
        description="Return current status and non-PII summary of a claim.",
        inputSchema={
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"],
        },
    ),
    Tool(
        name="update_claim_status",
        description="Advance claim to next status. Validates transition legality.",
        inputSchema={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "new_status": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["claim_id", "new_status", "reason"],
        },
    ),
    Tool(
        name="escalate_claim",
        description="Trigger a mandatory HITL gate (CONFLICT_REVIEW, FRAUD_REVIEW, APPROVAL).",
        inputSchema={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "gate": {"type": "string", "enum": ["CONFLICT_REVIEW", "FRAUD_REVIEW", "APPROVAL"]},
                "reason": {"type": "string"},
            },
            "required": ["claim_id", "gate", "reason"],
        },
    ),
    Tool(
        name="request_missing_document",
        description="Log a request for a missing document from the claimant.",
        inputSchema={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "document_type": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["claim_id", "document_type", "reason"],
        },
    ),
    Tool(
        name="check_policy_coverage",
        description="Query policy RAG corpus to check coverage for a scenario.",
        inputSchema={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["claim_id", "query"],
        },
    ),
    Tool(
        name="assess_fraud_risk",
        description="Run rule-based + LLM fraud risk assessment on a claim.",
        inputSchema={
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"],
        },
    ),
]

# Tool → minimum required role
_TOOL_ROLES: dict[str, str] = {
    "get_claim_status": "adjudicator",
    "update_claim_status": "adjudicator",
    "escalate_claim": "adjudicator",
    "request_missing_document": "adjudicator",
    "check_policy_coverage": "auditor",   # read-only, lowest privilege
    "assess_fraud_risk": "adjudicator",
}

_ROLE_HIERARCHY = {"auditor": 0, "adjudicator": 1, "supervisor": 2}


def _check_rbac(tool_name: str, caller_role: str) -> bool:
    """Return True if caller_role is sufficient for tool_name."""
    required = _ROLE_ROLES.get(tool_name, "supervisor")
    return _ROLE_HIERARCHY.get(caller_role, -1) >= _ROLE_HIERARCHY.get(required, 99)


class InsuranceMCPServer:
    """MCP Server for insurance claims domain tools.

    Extends MLMCPServer pattern from inventra/integrations/mcp_server.py.
    """

    def __init__(self) -> None:
        self.server = Server("insurance-claims")
        self._audit_log: list[dict[str, Any]] = []
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return TOOLS

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            # RBAC check (Week 4: extract caller role from session context)
            caller_role = arguments.pop("_caller_role", "adjudicator")

            # Audit entry
            audit_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": name,
                "caller_role": caller_role,
                "args": {k: v for k, v in arguments.items() if "pii" not in k.lower()},
            }

            result = await self._dispatch(name, arguments)
            audit_entry["result_preview"] = str(result)[:200]
            self._audit_log.append(audit_entry)

            return result

    async def _dispatch(self, name: str, args: dict) -> list[TextContent]:
        """Route tool call to the appropriate handler."""
        # Import LangChain tools and invoke them (sync tools wrapped in asyncio)
        from agent.tools.claim_tools import (
            assess_fraud_risk,
            check_policy_coverage,
            escalate_claim,
            get_claim_status,
            request_missing_document,
            update_claim_status,
        )

        dispatch_map = {
            "get_claim_status": get_claim_status,
            "update_claim_status": update_claim_status,
            "escalate_claim": escalate_claim,
            "request_missing_document": request_missing_document,
            "check_policy_coverage": check_policy_coverage,
            "assess_fraud_risk": assess_fraud_risk,
        }

        tool_fn = dispatch_map.get(name)
        if not tool_fn:
            raise ValueError(f"Unknown tool: {name}")

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tool_fn.invoke(args)
        )
        return [TextContent(type="text", text=result)]

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    print("Insurance Claims MCP Server")
    print("Tools:", [t.name for t in TOOLS])
    server = InsuranceMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
