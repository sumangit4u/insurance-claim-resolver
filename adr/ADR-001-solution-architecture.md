# ADR-001: Solution Architecture

**Status**: Accepted  
**Date**: 2026-06-08  
**Deciders**: Project team

## Context

Five non-negotiable client requirements drive every architectural decision:

1. Autonomous resolution of 70%+ of claims
2. Every LLM decision cited to a policy clause (hallucination rate < 2%)
3. Mandatory HITL gates for conflict, fraud, and high-value approvals
4. Full audit trail of agent reasoning (immutable, tamper-evident)
5. PII redaction before any external API call

## Decisions

### 1. Google ADK over bare LangChain/CrewAI

**Decision**: Use `google-adk` as the agent runtime.

**Rationale**: ADK provides built-in support for tool calling, ReAct loops, streaming, and Vertex AI integration. LangGraph is used as a conceptual reference (taught in Week 3) and for the 9-state workflow state machine only. ADK handles production agent execution.

### 2. Vertex AI Search over self-managed Chroma

**Decision**: Chroma locally, Vertex AI Search in production.

**Rationale**: Chroma is zero-infrastructure and familiar from teaching code (`rag_utils.py`). Vertex AI Search handles scale, compliance (SOC 2, ISO 27001), and managed embeddings. The abstraction layer in `rag/retriever.py` swaps backends via `settings.gcp_ready`.

### 3. Firestore over Redis/PostgreSQL for state

**Decision**: In-memory dict locally, Firestore in production.

**Rationale**: Firestore is serverless, offers strong consistency, and integrates with Cloud IAM. Redis adds operational overhead; PostgreSQL requires schema management. The claims workflow state is document-shaped, which maps naturally to Firestore.

### 4. MCP for enterprise tool layer

**Decision**: All agent tools exposed via MCP (`mcp` library), same as teaching code.

**Rationale**: MCP provides a standard protocol, RBAC hooks, and audit logging at the tool boundary. Reuses the pattern from `inventra/integrations/mcp_server.py`.

### 5. Cloud Run over GKE

**Decision**: Deploy on Cloud Run, not GKE.

**Rationale**: Claims workload is request-driven, not persistent. Cloud Run auto-scales to zero, costs ~60% less than a minimum GKE cluster, and requires no cluster management.

## Component Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Firebase Hosting                   │
│              (React dashboard / HITL UI)             │
└──────────────────────────┬───────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼───────────────────────────┐
│              Cloud Run — FastAPI + SSE               │
│         api/main.py  (streaming responses)           │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│           Google ADK ReAct Claims Agent              │
│         agent/claims_agent.py                        │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Specialist │  │  MCP Server  │  │  GraphRAG   │  │
│  │  Agents   │  │  (6 tools)   │  │  (Week 6)   │  │
│  └────────────┘  └──────────────┘  └─────────────┘  │
└────────┬──────────────────┬───────────────┬──────────┘
         │                  │               │
┌────────▼──────┐  ┌────────▼──────┐  ┌────▼──────────┐
│ Vertex AI     │  │   Firestore   │  │  Cloud Trace  │
│ Search (RAG)  │  │ (claim state) │  │ (OTEL traces) │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Consequences

- **Positive**: Zero GCP lock-in for local dev; swap backends via feature flag
- **Positive**: Teaching code patterns (Chroma, MCP, LangGraph) are directly reused
- **Positive**: Cloud Run keeps infra costs near zero for a 10-week build
- **Negative**: ADK API surface is newer and may have breaking changes
- **Negative**: Firestore strong consistency adds ~10ms latency vs Redis
- **Mitigation**: Pin `google-adk` to a minor version; add Firestore timeout handling
