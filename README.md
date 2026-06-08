# Insurance Claims Resolution Agent

An end-to-end, production-grade AI system that autonomously resolves 70%+ of insurance claims while maintaining full regulatory auditability, built over 10 weeks using Google ADK + Gemini.

## 10-Week Build Plan

| Week | Milestone | Key Deliverable |
|------|-----------|----------------|
| 0 | Project scaffold | Repo, CI, ADRs, synthetic data |
| 1 | Document pipeline | PDF parser, chunker, Chroma vector store |
| 2 | Vertex AI RAG | Production RAG, RAGAS baseline |
| 3 | ReAct claims agent | ADK agent, 6 domain tools, state machine |
| 4 | MCP enterprise layer | 6 MCP tools, RBAC, audit log |
| 5 | Multi-agent system | Specialist agents, supervisor routing |
| 6 | GraphRAG | Policy knowledge graph, typed edges |
| 7 | HITL gates | 3 mandatory review gates, SSE streaming |
| 8 | Evaluation | 5 LLM judges, Cohen's kappa ≥ 0.60 |
| 9 | Observability | OpenTelemetry, Cloud Trace, OWASP audit |
| 10 | Production deploy | Cloud Run, Firebase Hosting, Cloud Build |

## Tech Stack

| Layer | Technology |
|-------|----------|
| Agent runtime | Google ADK (`google-adk`) |
| LLM | Gemini 2.5 Flash / Pro |
| RAG (local) | Chroma + `GoogleGenerativeAIEmbeddings` |
| RAG (prod) | Vertex AI Search |
| State persistence | Firestore (local: in-memory) |
| MCP tools | `mcp` library |
| API | FastAPI + SSE |
| Evaluation | RAGAS + 5 domain LLM judges |
| CI/CD | GitHub Actions + Cloud Build |
| Deploy | Cloud Run + Firebase Hosting |

## Setup

```bash
# 1. Clone and create venv
git clone https://github.com/sumangit4u/insurance-claim-resolver
cd insurance-claim-resolver
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Run checks
make check

# 5. Start API server
make run
```

## Project Structure

```
insurance-claim-resolver/
├── agent/                   # ADK ReAct agent
│   ├── claims_agent.py
│   └── tools/               # 6 domain tools
├── rag/                     # RAG pipeline
│   └── retriever.py
├── workflow/                # 9-state claims workflow
│   ├── claims_workflow.py
│   └── states.py
├── mcp_server/              # MCP enterprise layer
│   └── server.py
├── api/                     # FastAPI + SSE
│   └── main.py
├── evaluation/              # RAGAS + 5 LLM judges
├── config/                  # Settings, model registry
├── prompts/                 # Versioned prompt registry
├── data/                    # Synthetic data (replace with real)
│   ├── policies/
│   ├── claims/
│   ├── communications/
│   ├── sops/
│   ├── historical_claims/
│   └── api_spec/
├── adr/                     # Architecture Decision Records
├── tests/
└── .github/workflows/       # CI pipeline
```

## Branching Strategy

- `main` — protected, deployable
- `develop` — integration branch
- `week/N-*` — weekly feature branches

## Non-Negotiable Client Requirements

1. Autonomous resolution of 70%+ of claims
2. Every LLM decision cited to a policy clause
3. Mandatory HITL gates: CONFLICT_REVIEW, FRAUD_REVIEW, APPROVAL
4. Full audit trail of agent reasoning
5. PII redaction before any external calls
