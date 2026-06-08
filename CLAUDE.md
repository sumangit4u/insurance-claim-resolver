# CLAUDE.md — AI Coding Assistant Context

This file gives AI assistants (Claude, Copilot, etc.) the context they need to contribute effectively.

## Current State

**Week 0 — Scaffold only.** No agent logic exists yet. Build week by week.

## Tech Stack

- **Agent runtime**: `google-adk` (NOT bare LangChain agents)
- **LLM**: `ChatGoogleGenerativeAI` with model `gemini-2.5-flash` (from `langchain-google-genai`)
- **RAG (local)**: Chroma + `GoogleGenerativeAIEmbeddings` (same as `session10_Agentic_RAG/rag_utils.py`)
- **RAG (prod)**: Vertex AI Search (GCP not yet available)
- **State**: Firestore (use in-memory dict until GCP ready)
- **MCP**: `mcp` library — same pattern as `inventra/integrations/mcp_server.py`
- **Config**: `pydantic-settings` `BaseSettings` in `config/settings.py`
- **API**: FastAPI + SSE (`sse-starlette`)

## Key Patterns from Teaching Code

### Agent State (from `inventra/agents/coordinator.py`)
```python
class ClaimState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    claim_id: str
    claim_status: str  # 9-state enum
    ...
```

### MCP Tool Pattern (from `inventra/integrations/mcp_server.py`)
```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_claim_status":
        return await self._get_claim_status(arguments["claim_id"])
```

### RAG Pattern (from `session10_Agentic_RAG/rag_utils.py`)
- Two-level chunking: `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter`
- Two Chroma collections: detail (`chunk_size=800`) and summary
- Grader schemas: `RetrievalGrade`, `HallucinationGrade`, `AnswerGrade`

## Domain Context

### Claims Lifecycle (9 states)
```
INTAKE → TRIAGE → COVERAGE_CHECK → [CONFLICT_REVIEW] →
INVESTIGATION → [FRAUD_REVIEW] → DECISION → APPROVAL → CLOSED
```

### Claim Types
- Motor (vehicle damage, theft, third-party liability)
- Health (hospitalisation, OPD, critical illness)
- Property (fire, flood, burglary, structural damage)

### HITL Gates (mandatory)
1. `CONFLICT_REVIEW` — contradictory evidence detected
2. `FRAUD_REVIEW` — fraud score > 0.7
3. `APPROVAL` — settlement > Rs 5,00,000

### PII Fields (always redact before external calls)
- `policy_holder_name`, `aadhaar_number`, `pan_number`, `phone`, `email`, `address`

## File Map

```
agent/claims_agent.py          # ADK ReAct agent entry point
agent/tools/claim_tools.py     # 6 domain tools
rag/retriever.py               # Chroma → Vertex AI Search abstraction
workflow/claims_workflow.py    # 9-state LangGraph workflow (reference)
workflow/states.py             # ClaimState TypedDict + transitions
mcp_server/server.py           # MCP server with 6 tools + RBAC
api/main.py                    # FastAPI + SSE streaming
evaluation/judges.py           # 5 LLM-as-Judge evaluators
config/settings.py             # pydantic-settings BaseSettings
config/models.yaml             # Model registry (name → config)
prompts/registry.yaml          # Versioned prompt registry
```

## GCP Status

**GCP account not yet available.** All GCP-dependent code must:
1. Be behind a feature flag (`settings.gcp_ready`)
2. Have a local fallback (Chroma instead of Vertex AI Search, dict instead of Firestore)
3. GCP packages in `requirements.txt` are commented out
