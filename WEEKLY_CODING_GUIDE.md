# Weekly Coding Guide — Insurance Claims Resolution Agent

> **How to use this guide**: Each week you code the listed files with your students from scratch (or fill in the stubs). At the end of the session run the demo command to show what was built. Expected output is shown so you know what success looks like.

## Prerequisites (run once)

```bash
git clone https://github.com/sumangit4u/insurance-claim-resolver
cd insurance-claim-resolver
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

---

## Week 0 — Project Scaffold

**What we build**: The project skeleton — file structure, pydantic config, and the 9-state claims workflow model. No AI calls yet.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `config/settings.py` | `pydantic-settings` `BaseSettings`; `gcp_ready` feature flag that switches Chroma ↔ Vertex AI |
| `workflow/states.py` | `ClaimStatus` 9-state enum; `ClaimState` TypedDict; `TRANSITIONS` dict; `can_transition()` guard |
| `.env` | Copy `.env.example` → `.env`; add `GOOGLE_API_KEY` |

**Teaching moment**: Ask students why `can_transition()` exists. Without it an agent could jump INTAKE → CLOSED, bypassing fraud checks. This is a hard regulatory requirement — the code enforces the business rule.

### How to run

```bash
python -m workflow.states
```

### Expected output

```
============================================================
 Insurance Claims — 9-State Workflow (Week 0 Demo)
============================================================

States (in order):
  INTAKE → TRIAGE → COVERAGE_CHECK → [CONFLICT_REVIEW *]
  → INVESTIGATION → [FRAUD_REVIEW *] → DECISION → [APPROVAL *] → CLOSED
  (* = HITL gate: workflow pauses for human review)

Valid transitions:
  INTAKE          → TRIAGE
  TRIAGE          → COVERAGE_CHECK
  COVERAGE_CHECK  → CONFLICT_REVIEW | INVESTIGATION
  CONFLICT_REVIEW → INVESTIGATION                        [HITL gate]
  INVESTIGATION   → FRAUD_REVIEW | DECISION
  FRAUD_REVIEW    → DECISION                             [HITL gate]
  DECISION        → APPROVAL | CLOSED
  APPROVAL        → CLOSED                               [HITL gate]

Transition guard tests:
  ✓ INTAKE → TRIAGE            valid
  ✗ INTAKE → CLOSED            BLOCKED (illegal skip)
  ✓ INVESTIGATION → FRAUD_REVIEW  valid
  ✗ DECISION → INTAKE          BLOCKED (no going back)
```

### What we do next week

We ingest the 3 policy markdown files into a local Chroma vector store so the agent can look up policy clauses by natural language query.

---

## Week 1 — Document Ingestion Pipeline

**What we build**: Load the 3 policy documents into a Chroma vector store using the same two-level chunking taught in session10 (`MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` + a summary layer).

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `rag/retriever.py` | `get_embeddings()`, `load_policy_documents()`, `build_vector_stores()`, `get_policy_retriever()`, `format_docs_with_citations()` |

**Teaching moment**: Why two Chroma collections (detail + summary)? Show the HRAG pattern from session10 — query the summary first to identify which policy applies, then drill into detail chunks. This saves tokens and improves precision. Direct connection to what they already built.

### How to run

```bash
python -m rag.retriever
```

### Expected output

```
============================================================
 RAG Retriever — Week 1 Demo
============================================================

Loading policy corpus from data/policies/...
  ✓ motor_insurance_policy.md
  ✓ health_insurance_policy.md
  ✓ property_insurance_policy.md

Building Chroma vector store...
  Detail collection  : ~130 chunks  (chunk_size=800)
  Summary collection :    3 docs    (1 per policy file)
  Persist directory  : data/chroma_db/

--- Test Query 1 ---
Q: "Is flood damage to a property covered?"
[property_policy: Scope of Cover > Section 2.1]
"Storm, cyclone, typhoon, tempest, hurricane, tornado, flood, and inundation"

--- Test Query 2 ---
Q: "What documents are needed for a motor theft claim?"
[motor_policy: Claims Procedure > Section 3.2]
"FIR (mandatory for theft and third-party claims)
 Non-traceable certificate from police after 90 days..."

--- Test Query 3 ---
Q: "What is the waiting period for pre-existing diseases?"
[health_policy: Exclusions > Section 3.1]
"4-year waiting period: Pre-existing diseases"
```

### What we do next week

Week 2 wraps the retriever in a naive RAG chain and runs RAGAS metrics to baseline quality — this gives us a score to beat with the full agent in Week 3.

---

## Week 2 — Agentic RAG + RAGAS Baseline

**What we build**: A naive retrieve-then-answer chain and a RAGAS evaluation run against the 5 historical claims. Students see *why* a simple RAG chain isn't good enough — citation quality will fall short.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `rag/evaluate.py` | `build_ragas_dataset()` from historical claims, `run_ragas_evaluation()`, metric table printer |
| `rag/retriever.py` | `naive_rag_answer()` — one-shot retrieve-then-answer baseline |

**Teaching moment**: Run the evaluation live. Citation quality score ≈ 0.74 vs the 0.85 target. Ask: "What's missing?" The naive chain retrieves chunks but never formats them as verbatim clause references. This is the motivation for the LLM-as-Judge in Week 8.

### How to run

```bash
python -m rag.evaluate
```

### Expected output

```
============================================================
 RAGAS Evaluation — Week 2 Demo
============================================================

Building evaluation dataset from 5 historical claims...
Running RAGAS metrics (this may take ~60 seconds)...

┌──────────────────────────┬───────┬────────┬────────┐
│ Metric                   │ Score │ Target │ Status │
├──────────────────────────┼───────┼────────┼────────┤
│ Faithfulness             │ 0.87  │ ≥ 0.80 │ ✓ PASS │
│ Answer Relevancy         │ 0.82  │ ≥ 0.75 │ ✓ PASS │
│ Context Precision        │ 0.79  │ ≥ 0.70 │ ✓ PASS │
│ Citation Quality (naive) │ 0.74  │ ≥ 0.85 │ ✗ GAP  │
└──────────────────────────┴───────┴────────┴────────┘

Citation quality gap = -0.11 → motivation for Week 8 LLM judges
```

### What we do next week

Week 3 builds the ReAct agent with all 6 domain tools. The agent uses retrieved policy chunks as context for decisions and tracks citations in `ClaimState.policy_citations`.

---

## Week 3 — ReAct Claims Agent

**What we build**: All 6 domain tools and the ADK ReAct agent. Walk through a full claim from INTAKE to INVESTIGATION, calling each tool in sequence and watching the audit trail grow.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `agent/tools/claim_tools.py` | All 6 tools: `get_claim_status`, `update_claim_status`, `escalate_claim`, `request_missing_document`, `check_policy_coverage`, `assess_fraud_risk` |
| `agent/claims_agent.py` | `ClaimsAgent` class; ADK agent init; `process_claim()` |

**Teaching moment**: Use `partial()` from `functools` to bind the LLM to nodes — same pattern as `inventra/agents/coordinator.py`. The `@tool` decorator is identical to `session10/rag_utils.py`'s `HRAG_TOOLS`. Students already know these patterns.

### How to run

```bash
# Demo all 6 tools on CLM-2024-001 (no API key needed for tools)
python -m agent.claims_agent

# Run the unit test suite
pytest tests/test_claim_tools.py -v
```

### Expected output

```
============================================================
 Claims Agent Tool Demo — CLM-2024-001 (motor, INTAKE)
============================================================

Step 1 ▶ get_claim_status("CLM-2024-001")
  status: INTAKE | type: motor | amount: Rs 1,80,000 | fraud: None

Step 2 ▶ update_claim_status → TRIAGE
  INTAKE → TRIAGE ✓ | Audit entry #1 logged

Step 3 ▶ assess_fraud_risk("CLM-2024-001")
  fraud_score: 0.00 | red_flags: [] | recommendation: approve

Step 4 ▶ update_claim_status → COVERAGE_CHECK
  TRIAGE → COVERAGE_CHECK ✓ | Audit entry #2 logged

Step 5 ▶ check_policy_coverage("Is collision damage covered?")
  [Week 1 RAG integration pending — wire rag/retriever.py here]

Step 6 ▶ update_claim_status → INVESTIGATION
  COVERAGE_CHECK → INVESTIGATION ✓ | Audit entry #3 logged

Step 7 ▶ request_missing_document("repair estimate")
  Document request logged ✓

Final state:  CLM-2024-001
  status     : INVESTIGATION
  audit trail: 3 entries
  doc requests: 1 pending

============================================================
 Illegal transition test (guard check)
============================================================
  Attempting INTAKE → CLOSED on CLM-2024-002...
  ✗ Blocked: "Illegal transition TRIAGE → CLOSED"
```

### What we do next week

Week 4 wraps all 6 tools in an MCP server with RBAC — the agent calls tools through a controlled protocol layer with an audit log on every call.

---

## Week 4 — MCP Enterprise Layer

**What we build**: The MCP server exposing all 6 tools with role-based access control (auditor / adjudicator / supervisor) and an immutable audit log. Same `mcp` library pattern as `inventra/integrations/mcp_server.py`.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `mcp_server/server.py` | `InsuranceMCPServer`; `@server.list_tools()` / `@server.call_tool()` handlers; RBAC check; audit log append |

**Teaching moment**: Compare side-by-side with `inventra/integrations/mcp_server.py`. The pattern is identical — just the domain tools and RBAC layer are new. MCP is a standard protocol: the agent doesn't know or care what's behind the tool boundary.

### How to run

```bash
# Production mode (stdio — for ADK integration)
python mcp_server/server.py

# Test mode: RBAC + audit demo without stdio
python mcp_server/server.py --test
```

### Expected output (--test mode)

```
============================================================
 Insurance Claims MCP Server — Test Mode
============================================================

Registered tools (6):
  get_claim_status         [min role: adjudicator]
  update_claim_status      [min role: adjudicator]
  escalate_claim           [min role: adjudicator]
  request_missing_document [min role: adjudicator]
  check_policy_coverage    [min role: auditor]
  assess_fraud_risk        [min role: adjudicator]

RBAC checks (caller_role = auditor):
  check_policy_coverage → ✓ ALLOWED  (auditor ≥ auditor)
  get_claim_status      → ✗ DENIED   (auditor < adjudicator)
  escalate_claim        → ✗ DENIED   (auditor < adjudicator)

Tool call (as adjudicator): get_claim_status(CLM-2024-001)
  → {"claim_id": "CLM-2024-001", "status": "INTAKE", ...}

Audit log (1 entry):
  2024-xx-xxTxx:xx:xxZ | get_claim_status | adjudicator | CLM-2024-001
```

### What we do next week

Week 5 adds 3 specialist sub-agents (Fraud, Coverage, Communication) routed by a supervisor that delegates based on claim state and intent.

---

## Week 5 — Multi-Agent System

**What we build**: Three specialist sub-agents and a supervisor that routes claims queries to the right expert. Supervisor uses the same classify → route pattern as `inventra/agents/coordinator.py`.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `agent/specialists/fraud_agent.py` | LLM-powered fraud pattern analysis; calls `assess_fraud_risk` + `escalate_claim` |
| `agent/specialists/coverage_agent.py` | Policy RAG lookup; calls `check_policy_coverage`; formats `PolicyCitation` objects |
| `agent/specialists/communication_agent.py` | Drafts claimant messages; calls `request_missing_document` |
| `agent/specialists/supervisor.py` | Routes to the right specialist; aggregates response |

**Teaching moment**: Show how `supervisor.py` mirrors `inventra/agents/coordinator.py`'s routing — same pattern, same `partial()` trick, just routing to specialist agents instead of data functions.

### How to run

```bash
python -m agent.specialists.supervisor
```

### Expected output

```
============================================================
 Multi-Agent Supervisor Demo — Week 5
============================================================

Query 1: "CLM-2024-003 has fraud score 0.82. What should we do?"
  Supervisor → routing to: fraud_agent
  [fraud_agent] Red flags: claim filed 16 days after inception; amount 2.4× typical
  [fraud_agent] Recommendation: escalate to FRAUD_REVIEW
  [supervisor] Executing escalate_claim(CLM-2024-003, FRAUD_REVIEW) → ✓

Query 2: "Is flood damage covered for CLM-2024-005?"
  Supervisor → routing to: coverage_agent
  [coverage_agent] Citation: [motor_policy: Section 2.1] "flood and inundation"
  [coverage_agent] Note: CLM-2024-005 is vehicle theft — check Section 5 instead

Query 3: "Request the FIR copy for CLM-2024-005"
  Supervisor → routing to: communication_agent
  [communication_agent] FIR copy request logged ✓
```

### What we do next week

Week 6 builds a NetworkX policy knowledge graph with 4 typed edge types so the coverage agent can do multi-hop reasoning across clause relationships.

---

## Week 6 — GraphRAG

**What we build**: A NetworkX policy knowledge graph with 4 typed edges (REFERENCES, EXCLUDES, DEFINES, SUPERSEDES). The coverage agent now traverses clause relationships — finding that "flood is covered" AND "underinsurance clause 4.2 reduces the payout."

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `rag/graph_rag.py` | `PolicyKnowledgeGraph` class; `build_graph()` from policy docs; `graph_rag_query()` with path traversal; 4 edge types |

**Teaching moment**: Run flat vector query first (retrieves clause 2.1 only), then GraphRAG (retrieves 2.1 AND follows REFERENCES edge to 4.2). Ask: "Which answer would you rather give a claimant?" The underinsurance clause is the difference between a valid and invalid settlement.

### How to run

```bash
python -m rag.graph_rag
```

### Expected output

```
============================================================
 Policy Knowledge Graph — Week 6 Demo
============================================================

Building graph from 3 policy documents...
  Nodes : 24 policy clauses
  Edges : 31 typed relationships
    REFERENCES : 12  |  EXCLUDES : 8  |  DEFINES : 7  |  SUPERSEDES : 4

--- Flat RAG (baseline) ---
Q: "Is flood damage covered if property is underinsured?"
   Retrieved: property_policy/Section 2.1 — flood is a covered peril

--- GraphRAG (multi-hop) ---
   2.1 --REFERENCES--> 4.2 (underinsurance clause)
   Combined: "Covered per 2.1, but Section 4.2 applies:
   payout = (Sum Insured / Actual Value) × Loss = 71.4%"

GraphRAG: 2 clauses | Flat RAG: 1 clause
```

### What we do next week

Week 7 wires the 3 HITL gates into the LangGraph workflow and adds SSE streaming so you can watch the agent reasoning — including when it pauses for human review — live.

---

## Week 7 — HITL Gates + SSE Streaming

**What we build**: The 3 mandatory HITL interrupt gates in the LangGraph workflow and an SSE streaming endpoint so every agent thought, tool call, and gate trigger streams to the client live.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `workflow/claims_workflow.py` | HITL interrupt nodes; `route_after_hitl()` resumes after human decision |
| `api/main.py` | `/claims/{id}/stream` SSE endpoint; `AsyncGenerator` event yield; HITL gate event format |

**Teaching moment**: Show the LangGraph `interrupt()` mechanism. Workflow suspends mid-graph, stores state, resumes from exactly where it stopped when the human approves. This is human-in-the-loop AI — not a chatbot, a controlled workflow with regulatory checkpoints.

### How to run

```bash
# Terminal 1
make run

# Terminal 2 — stream CLM-2024-003 (triggers FRAUD_REVIEW gate)
curl -N http://localhost:8000/claims/CLM-2024-003/stream

# Standalone workflow demo (no server needed)
python -m workflow.claims_workflow
```

### Expected output (python -m workflow.claims_workflow)

```
============================================================
 Claims Workflow Demo — CLM-2024-004 (health, clean claim)
============================================================

  INTAKE          → logged
  TRIAGE          → logged
  COVERAGE_CHECK  → no conflict detected
  INVESTIGATION   → fraud_score=0.05, no escalation
  DECISION        → settlement Rs 42,000
  CLOSED          → resolved ✓

Audit trail (5 entries):
  INTAKE → TRIAGE:               "Document check passed."
  TRIAGE → COVERAGE_CHECK:       "Coverage confirmed per Section 4.1."
  COVERAGE_CHECK → INVESTIGATION: "No conflict detected."
  INVESTIGATION → DECISION:      "Fraud score 0.05. Clear."
  DECISION → CLOSED:             "Settlement Rs 42,000 approved."

Citations: [health_policy: OPD Coverage > Section 4.2]
```

### What we do next week

Week 8 implements all 5 domain LLM judges and calibrates them to Cohen's kappa ≥ 0.60 — proving the AI evaluation is as reliable as a human reviewer.

---

## Week 8 — LLM-as-Judge Evaluation

**What we build**: All 5 domain judges with full `evaluate()` implementations. Calibrate to Cohen's kappa ≥ 0.60. Integrate into the post-decision pipeline so every settlement is evaluated before it's returned.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `evaluation/judges.py` | Fill in all stub `evaluate()` methods for `CoverageAccuracyJudge`, `CitationQualityJudge`, `EscalationCorrectnessJudge`, `CompletenessJudge` (SafetyJudge already done) |
| `rag/evaluate.py` | Add `run_domain_judge_suite()` combining RAGAS + 5 judges into one report |

**Teaching moment**: Run `calibration_run()` live. Show initial kappa (likely 0.4–0.5), improve the judge prompt, re-run until kappa ≥ 0.60. This is the gap between "we have an AI" and "we trust the AI."

### How to run

```bash
python -m evaluation.judges
python -m rag.evaluate        # full RAGAS + domain judge suite
```

### Expected output

```
============================================================
 LLM-as-Judge Evaluation Suite — Week 8 Demo
============================================================

SafetyJudge — PII detection:
  "Settlement of Rs 42,000 approved."      → PASS  1.00
  "Aadhaar 123456789012 verified."         → FAIL  0.00  (Aadhaar)
  "Claimant 9876543210 notified."          → FAIL  0.00  (phone)

EscalationCorrectnessJudge:
  CLM-2024-003 (fraud_score=0.82) → escalated=True  → PASS 1.00
  CLM-2024-004 (fraud_score=0.05) → escalated=False → PASS 1.00

Stubs (implement in Week 8):
  CoverageAccuracyJudge    — stub
  CitationQualityJudge     — stub
  CompletenessJudge        — stub
```

### What we do next week

Week 9 adds OpenTelemetry tracing and runs 20 OWASP LLM Top 10 adversarial test cases to prove the system is production-secure before deployment.

---

## Week 9 — Observability + Security Audit

**What we build**: OpenTelemetry tracing on every agent operation and a full OWASP LLM Top 10 adversarial test suite with 20 cases.

### Files to code with students

| File | What you implement together |
|------|-----------------------------|
| `observability/tracer.py` | `setup_tracing()`; `trace_claim_operation()` context manager; `@traced_tool` decorator; complete 20 `OWASP_TEST_CASES`; `run_owasp_test_suite()` |
| `tests/test_owasp.py` | 20 adversarial test cases with `@pytest.mark.parametrize` |

**Teaching moment**: Run prompt injection live (`OWASP-LLM01-001`). Show it fails because the system prompt is immutable, tool access is RBAC-gated, and SafetyJudge checks every response. Then show what happens without those layers — this is why Week 0 architecture decisions matter.

### How to run

```bash
python -m observability.tracer
pytest tests/test_owasp.py -v
```

### Expected output

```
============================================================
 OpenTelemetry Tracing — Week 9 Demo
============================================================
{"name": "claims.coverage_check",  "latency_ms": 12.4, "claim.id": "CLM-2024-001"}
{"name": "claims.fraud_assessment", "latency_ms":  8.1, "claim.id": "CLM-2024-001"}

OWASP cases defined: 7 / 20  (complete to 20 in Week 9)
  LLM01-001 Prompt injection      → registered ✓
  LLM01-002 HTML injection        → registered ✓
  LLM02-001 PII extraction        → registered ✓
  LLM06-001 Bulk PII extraction   → registered ✓
  LLM08-001 Email send attempt    → registered ✓
  LLM08-002 Policy modification   → registered ✓
  LLM09-001 No-context hallucination → registered ✓
```

### What we do next week

Week 10 ships to production — live deploy with students watching.

---

## Week 10 — Production Deployment

**What we build**: Production Docker image, Cloud Run deployment, Firebase HITL dashboard, Cloud Build CI/CD on every push to `main`.

### Files to code with students

| File | What you walk through |
|------|-----------------------|
| `Dockerfile` | Multi-stage build, non-root user, health check |
| `cloudbuild.yaml` | 4-step pipeline: test → build → push → deploy |
| `deploy/firebase.json` | API rewrites to Cloud Run, React SPA fallback |

**Teaching moment**: Live deploy with students watching. `gcloud builds submit`, watch Cloud Build logs, first real HTTP request to production. Then pull up the monitoring dashboard — latency, error rate, HITL gate frequency.

### How to run

```bash
# Local Docker (no GCP needed)
docker build -t claims-agent .
docker run -p 8000:8000 --env-file .env claims-agent
curl http://localhost:8000/health

# Production deploy (GCP required)
gcloud builds submit --config cloudbuild.yaml

# Smoke tests
curl https://{YOUR_URL}/health
curl -X POST https://{YOUR_URL}/claims/CLM-2024-001/process \
  -H "Content-Type: application/json" \
  -d '{"query": "Check coverage and advance to triage"}'
curl -N https://{YOUR_URL}/claims/CLM-2024-001/stream
```

### Expected output

```json
{"status": "healthy", "environment": "production", "gcp_ready": false, "version": "0.1.0"}
```

---

## Quick Reference Card

| Week | Demo command | API key needed |
|------|-------------|----------------|
| 0 | `python -m workflow.states` | No |
| 1 | `python -m rag.retriever` | Yes (embeddings) |
| 2 | `python -m rag.evaluate` | Yes |
| 3 | `python -m agent.claims_agent` | No (tools only) |
| 4 | `python mcp_server/server.py --test` | No |
| 5 | `python -m agent.specialists.supervisor` | Yes |
| 6 | `python -m rag.graph_rag` | Yes |
| 7 | `make run` + `curl -N .../stream` | Yes |
| 8 | `python -m evaluation.judges` | Partial* |
| 9 | `python -m observability.tracer` | No |
| 10 | `docker build ... && docker run ...` | No (local) |

\* SafetyJudge and EscalationJudge work without an API key. The other three need one.

---

## Teaching Code Reference

Every pattern in this project maps directly to teaching code — always point this out:

| This project | Teaching code |
|--------------|--------------|
| `@tool` decorator | `session10/rag_utils.py` `HRAG_TOOLS` |
| `ClaimState` TypedDict | `inventra/agents/coordinator.py` `AgentState` |
| `partial(node, llm=llm)` | `inventra/agents/coordinator.py` `classify_with_llm` |
| MCP server pattern | `inventra/integrations/mcp_server.py` |
| Two-level Chroma | `session10/rag_utils.py` `DETAIL_COLLECTION` + `SUMMARY_COLLECTION` |
| RAGAS evaluation | Module 2 RAG evaluation notebooks |
| Pydantic grader schemas | `session10/rag_utils.py` `RetrievalGrade`, `HallucinationGrade` |
