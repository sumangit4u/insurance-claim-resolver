"""RAGAS evaluation + domain judge suite — Week 2 and Week 8 deliverable.

Week 2: build_ragas_dataset() + run_ragas_evaluation() — baseline metrics
Week 8: run_domain_judge_suite() — combines RAGAS with all 5 LLM judges

Run:
    python -m rag.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
HISTORICAL_CLAIMS = DATA_DIR / "historical_claims" / "resolved_claims.json"

RAGAS_TARGETS = {
    "faithfulness":       0.80,
    "answer_relevancy":   0.75,
    "context_precision":  0.70,
    "citation_quality":   0.85,
}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_ragas_dataset() -> List[Dict[str, Any]]:
    """Build a RAGAS-compatible dataset from historical resolved claims.

    Each entry has: question, answer, contexts, ground_truth.
    Week 2: implement with real RAG answers from naive_rag_answer().
    """
    claims = json.loads(HISTORICAL_CLAIMS.read_text(encoding="utf-8"))
    dataset = []
    for claim in claims:
        dataset.append({
            "question": (
                f"Is this {claim['claim_type']} claim covered? "
                f"Incident: {claim['incident']}"
            ),
            "answer": claim.get("resolution_reason", ""),
            "contexts": claim.get("policy_citations", []),
            "ground_truth": claim.get("resolution", ""),
        })
    return dataset


# ---------------------------------------------------------------------------
# RAGAS runner (stub — implement in Week 2)
# ---------------------------------------------------------------------------

def run_ragas_evaluation(dataset: List[Dict[str, Any]] | None = None) -> Dict[str, float]:
    """Run RAGAS metrics on the evaluation dataset.

    Week 2: uncomment and wire up:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        return result.to_pandas().mean().to_dict()
    """
    # Stub scores (replace with real RAGAS output in Week 2)
    return {
        "faithfulness":      0.87,
        "answer_relevancy":  0.82,
        "context_precision": 0.79,
        "citation_quality":  0.74,   # Below target — motivation for Week 8
    }


# ---------------------------------------------------------------------------
# Domain judge suite (Week 8)
# ---------------------------------------------------------------------------

def run_domain_judge_suite(claim_decisions: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Run all 5 domain judges on a set of claim decisions.

    Week 8: fully implemented. Week 2: returns stubs.
    """
    from evaluation.judges import (
        ALL_JUDGES,
        EscalationCorrectnessJudge,
        SafetyJudge,
    )

    results = {}

    # SafetyJudge — works without API key
    safety = SafetyJudge()
    test_responses = [
        "Settlement of Rs 42,000 approved per Section 4.2.",
        "Aadhaar 123456789012 verified — claim approved.",
        "Claimant 9876543210 has been notified of rejection.",
    ]
    safety_results = [safety.evaluate(r) for r in test_responses]
    results["SafetyJudge"] = {
        "tests": len(test_responses),
        "passed": sum(1 for r in safety_results if r.passed),
        "scores": [round(r.score, 2) for r in safety_results],
    }

    # EscalationJudge — rule-based, no API key needed
    esc = EscalationCorrectnessJudge()
    results["EscalationCorrectnessJudge"] = "rule-based check available — see evaluation/judges.py"

    # Other judges — stubs until Week 8
    for judge_cls in ALL_JUDGES:
        name = judge_cls.__name__
        if name not in results:
            results[name] = "stub — implement in Week 8"

    return results


# ---------------------------------------------------------------------------
# Week 2 demo
# ---------------------------------------------------------------------------

def _print_table(scores: Dict[str, float]) -> None:
    print()
    print(f"  {'Metric':<30} {'Score':>6}  {'Target':>8}  {'Status'}")
    print("  " + "-" * 58)
    for metric, score in scores.items():
        target = RAGAS_TARGETS.get(metric, 0.80)
        status = "✓ PASS" if score >= target else "✗ GAP "
        print(f"  {metric:<30} {score:>6.2f}  {'>= ' + str(target):>8}  {status}")
    print()


if __name__ == "__main__":
    W = 60
    print("=" * W)
    print(" RAGAS Evaluation — Week 2 Demo")
    print("=" * W)
    print()
    print("Building evaluation dataset from historical claims...")
    dataset = build_ragas_dataset()
    print(f"  {len(dataset)} claim decisions loaded")
    print()
    print("Running RAGAS metrics...")
    print("  (stub scores shown — wire real RAGAS in Week 2)")
    scores = run_ragas_evaluation(dataset)
    _print_table(scores)

    gap = RAGAS_TARGETS["citation_quality"] - scores["citation_quality"]
    print(f"  Citation quality gap: -{gap:.2f}")
    print("  → Motivation for Week 8 LLM-as-Judge (CitationQualityJudge)")
    print()

    print("=" * W)
    print(" Domain Judge Suite — Week 8 Preview")
    print("=" * W)
    suite = run_domain_judge_suite()
    for judge, result in suite.items():
        if isinstance(result, dict):
            passed = result.get("passed", 0)
            total = result.get("tests", 0)
            print(f"  {judge:<35} {passed}/{total} passed  {result.get('scores', '')}")
        else:
            print(f"  {judge:<35} {result}")
