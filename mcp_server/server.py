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
    python mcp_server/server.py          # start MCP stdio server
    python mcp_server/server.py --test   # Week 4 demo: test all 6 tools directly
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Any

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
    "get_claim_status":       "adjudicator",
    "update_claim_status":    "adjudicator",
    "escalate_claim":         "adjudicator",
    "request_missing_document": "adjudicator",
    "check_policy_coverage":  "auditor",    # read-only, lowest privilege
    "assess_fraud_risk":      "adjudicator",
}

_ROLE_HIERARCHY = {"auditor": 0, "adjudicator": 1, "supervisor": 2}


def _check_rbac(tool_name: str, caller_role: str) -> bool:
    """Return True if caller_role is sufficient for tool_name."""
    required = _TOOL_ROLES.get(tool_name, "supervisor")   # fixed: was _ROLE_ROLES
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
            if not _check_rbac(name, caller_role):
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "RBAC_DENIED",
                        "tool": name,
                        "caller_role": caller_role,
                        "required_role": _TOOL_ROLES.get(name, "supervisor"),
                    })
                )]

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
        from agent.tools.claim_tools import (
            assess_fraud_risk,
            check_policy_coverage,
            escalate_claim,
            get_claim_status,
            request_missing_document,
            update_claim_status,
        )

        dispatch_map = {
            "get_claim_status":        get_claim_status,
            "update_claim_status":     update_claim_status,
            "escalate_claim":          escalate_claim,
            "request_missing_document": request_missing_document,
            "check_policy_coverage":   check_policy_coverage,
            "assess_fraud_risk":       assess_fraud_risk,
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


# ---------------------------------------------------------------------------
# Week 4 --test mode demo (no MCP client needed)
# ---------------------------------------------------------------------------

def _run_test_mode() -> None:
    """Call all 6 tools directly via LangChain — no MCP protocol required.

    Run with:  python mcp_server/server.py --test
    """
    from agent.tools.claim_tools import (
        assess_fraud_risk,
        check_policy_coverage,
        escalate_claim,
        get_claim_status,
        request_missing_document,
        update_claim_status,
    )

    W = 60
    print("=" * W)
    print(" Week 4 — MCP Server Test Mode")
    print("=" * W)
    print()

    # RBAC table
    print("RBAC Configuration")
    print("-" * 40)
    print(f"  {'Tool':<30} {'Min Role':<15} {'Hierarchy'}")
    print(f"  {'-'*28} {'-'*13} {'-'*10}")
    for tool_name, min_role in _TOOL_ROLES.items():
        level = _ROLE_HIERARCHY[min_role]
        print(f"  {tool_name:<30} {min_role:<15} level={level}")
    print()

    # RBAC checks
    print("RBAC Checks")
    print("-" * 40)
    checks = [
        ("get_claim_status",    "auditor",     True),
        ("escalate_claim",      "auditor",     False),
        ("assess_fraud_risk",   "adjudicator", True),
        ("update_claim_status", "supervisor",  True),
    ]
    for tool, role, expected in checks:
        result = _check_rbac(tool, role)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        verdict = "ALLOWED" if result else "DENIED"
        print(f"  {status}  {role:<15} → {tool:<28} {verdict}")
    print()

    # Live tool calls
    print("Live Tool Calls (claim CLM-2024-001)")
    print("-" * 40)
    test_calls = [
        ("get_claim_status",        {"claim_id": "CLM-2024-001"}),
        ("request_missing_document", {"claim_id": "CLM-2024-001",
                                      "document_type": "Police FIR",
                                      "reason": "Required for motor accident claims"}),
        ("assess_fraud_risk",       {"claim_id": "CLM-2024-001"}),
        ("escalate_claim",          {"claim_id": "CLM-2024-001",
                                     "gate": "FRAUD_REVIEW",
                                     "reason": "High fraud score detected"}),
        ("check_policy_coverage",   {"claim_id": "CLM-2024-001",
                                     "query": "Is collision damage covered?"}),
        ("update_claim_status",     {"claim_id": "CLM-2024-001",
                                     "new_status": "INVESTIGATION",
                                     "reason": "Triage complete"}),
    ]

    tool_map = {
        "get_claim_status":        get_claim_status,
        "request_missing_document": request_missing_document,
        "assess_fraud_risk":       assess_fraud_risk,
        "escalate_claim":          escalate_claim,
        "check_policy_coverage":   check_policy_coverage,
        "update_claim_status":     update_claim_status,
    }

    for tool_name, args in test_calls:
        print(f"  → {tool_name}({', '.join(f'{k}={v!r}' for k, v in args.items() if k == 'claim_id' or len(str(v)) < 30)})")
        try:
            output = tool_map[tool_name].invoke(args)
            # Print first 120 chars of output
            preview = output[:120] + ("..." if len(output) > 120 else "")
            print(f"     {preview}")
        except Exception as e:
            print(f"     ERROR: {e}")
        print()

    print("=" * W)
    print(" Week 4 Complete → Week 5: Multi-agent supervisor")
    print("=" * W)


async def _run_server() -> None:
    print("Insurance Claims MCP Server — stdio mode")
    print("Tools:", [t.name for t in TOOLS])
    server = InsuranceMCPServer()
    await server.run()


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test_mode()
    else:
        asyncio.run(_run_server())
