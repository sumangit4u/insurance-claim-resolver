"""GraphRAG — Policy Knowledge Graph — Week 6 deliverable.

Builds a typed knowledge graph over insurance policy documents using NetworkX.
Nodes = policy clauses; Edges = semantic relationships between clauses.

Edge types (4):
    REFERENCES  — clause A explicitly references clause B for detail
    EXCLUDES    — clause A excludes or limits the scope of clause B
    DEFINES     — clause A provides a definition used by clause B
    SUPERSEDES  — clause A (endorsement) replaces or overrides clause B

Week 6 deliverables:
    - PolicyKnowledgeGraph: build, query, and traverse the knowledge graph
    - graph_aware_retrieve(): Chroma seeds + 2-hop graph expansion
    - Context window: enriched context including exclusions + definitions

Pattern:
    - Same policy corpus as rag/retriever.py (data/policies/)
    - Chroma dense retrieval → seed clauses → graph expansion
    - Returns PolicyCitation-compatible context for the LLM

Usage:
    python -m rag.graph_rag
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
POLICY_DIR = DATA_DIR / "policies"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EdgeType = Literal["REFERENCES", "EXCLUDES", "DEFINES", "SUPERSEDES"]

EDGE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "REFERENCES": "Clause A points to clause B for further detail",
    "EXCLUDES":   "Clause A excludes a scenario covered by clause B",
    "DEFINES":    "Clause A provides a definition used in clause B",
    "SUPERSEDES": "Clause A (endorsement/amendment) overrides clause B",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PolicyClause:
    """A single clause or section from a policy document."""
    clause_id: str          # e.g. "motor_policy:3.2.1"
    source_doc: str         # e.g. "motor_policy"
    section: str            # e.g. "3.2.1"
    title: str              # e.g. "Own Damage Coverage"
    text: str               # Full verbatim clause text
    tags: list[str] = field(default_factory=list)


@dataclass
class PolicyEdge:
    """A typed directed relationship between two policy clauses."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    reason: str             # Human-readable rationale for this edge


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class PolicyKnowledgeGraph:
    """Typed knowledge graph over insurance policy documents.

    Nodes: PolicyClause instances
    Edges: Typed relationships (REFERENCES, EXCLUDES, DEFINES, SUPERSEDES)

    Week 6: build from policy markdown files + LLM-extracted relationships.
    Seed graph: hand-crafted for 3 policy types (motor, health, property).
    """

    def __init__(self) -> None:
        if not NX_AVAILABLE:
            raise ImportError(
                "networkx is required for GraphRAG. "
                "Install with: pip install networkx"
            )
        self.graph: nx.DiGraph = nx.DiGraph()
        self._clauses: dict[str, PolicyClause] = {}

    def add_clause(self, clause: PolicyClause) -> None:
        """Add a policy clause as a graph node."""
        self._clauses[clause.clause_id] = clause
        self.graph.add_node(
            clause.clause_id,
            source_doc=clause.source_doc,
            section=clause.section,
            title=clause.title,
            tags=clause.tags,
        )

    def add_edge(self, edge: PolicyEdge) -> None:
        """Add a typed directed relationship between two clauses."""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            reason=edge.reason,
        )

    def get_clause(self, clause_id: str) -> PolicyClause | None:
        return self._clauses.get(clause_id)

    def get_related_clauses(
        self,
        clause_id: str,
        edge_types: list[EdgeType] | None = None,
        depth: int = 2,
    ) -> list[PolicyClause]:
        """Return all clauses reachable from clause_id within depth hops.

        Args:
            clause_id: Starting clause
            edge_types: Filter to specific edge types; None = all
            depth: Maximum traversal depth

        Returns:
            List of related PolicyClause instances (start node excluded)
        """
        if clause_id not in self.graph:
            return []

        related: set[str] = set()
        frontier = {clause_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for successor in self.graph.successors(node):
                    data = self.graph.edges[node, successor]
                    if edge_types is None or data.get("edge_type") in edge_types:
                        if successor != clause_id:
                            next_frontier.add(successor)
                            related.add(successor)
            frontier = next_frontier

        return [self._clauses[cid] for cid in related if cid in self._clauses]

    def find_exclusions(self, clause_id: str) -> list[PolicyClause]:
        """Find clauses that EXCLUDE the given clause (1-hop)."""
        return self.get_related_clauses(clause_id, edge_types=["EXCLUDES"], depth=1)

    def find_definitions(self, clause_id: str) -> list[PolicyClause]:
        """Find definition clauses that DEFINE the given clause (1-hop)."""
        return self.get_related_clauses(clause_id, edge_types=["DEFINES"], depth=1)

    def find_superseding(self, clause_id: str) -> list[PolicyClause]:
        """Find endorsements that SUPERSEDE the given clause (1-hop)."""
        return self.get_related_clauses(clause_id, edge_types=["SUPERSEDES"], depth=1)

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        edge_type_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "UNKNOWN")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "edge_types": edge_type_counts,
            "source_docs": list({c.source_doc for c in self._clauses.values()}),
        }


# ---------------------------------------------------------------------------
# Seed graph (Week 6: replace with LLM extraction from policy .md files)
# ---------------------------------------------------------------------------

def build_seed_graph() -> PolicyKnowledgeGraph:
    """Build a hand-crafted seed knowledge graph for the 3 policy types."""
    kg = PolicyKnowledgeGraph()

    # Motor clauses
    for clause in [
        PolicyClause("motor:1.1", "motor_policy", "1.1", "Own Damage Coverage",
                     "We cover loss or damage to your vehicle caused by accident, fire, theft, or natural calamity.",
                     ["coverage", "motor", "own_damage"]),
        PolicyClause("motor:1.2", "motor_policy", "1.2", "Third Party Liability",
                     "We cover your legal liability to third parties for bodily injury or property damage.",
                     ["coverage", "motor", "third_party"]),
        PolicyClause("motor:2.1", "motor_policy", "2.1", "General Exclusions",
                     "This policy does not cover: (a) wear and tear; (b) mechanical breakdown; (c) depreciation.",
                     ["exclusion", "motor"]),
        PolicyClause("motor:2.2", "motor_policy", "2.2", "Driving Under Influence Exclusion",
                     "Claims arising from driving under the influence of alcohol or drugs are excluded.",
                     ["exclusion", "motor", "dui"]),
        PolicyClause("motor:3.1", "motor_policy", "3.1", "Definition — Accident",
                     "Accident means a sudden, unforeseen, and involuntary event causing damage.",
                     ["definition", "motor"]),
    ]:
        kg.add_clause(clause)

    # Health clauses
    for clause in [
        PolicyClause("health:1.1", "health_policy", "1.1", "Hospitalisation Cover",
                     "We cover hospitalisation expenses for illness or injury requiring 24+ hours admission.",
                     ["coverage", "health", "hospitalisation"]),
        PolicyClause("health:1.2", "health_policy", "1.2", "Day Care Procedures",
                     "We cover day care procedures that do not require 24-hour hospitalisation.",
                     ["coverage", "health", "daycare"]),
        PolicyClause("health:2.1", "health_policy", "2.1", "Pre-existing Disease Waiting Period",
                     "Pre-existing diseases are covered only after a 48-month waiting period.",
                     ["exclusion", "health", "waiting_period"]),
        PolicyClause("health:2.2", "health_policy", "2.2", "Cosmetic Surgery Exclusion",
                     "Cosmetic, aesthetic, or weight-reduction procedures are not covered.",
                     ["exclusion", "health"]),
        PolicyClause("health:3.1", "health_policy", "3.1", "Definition — Pre-existing Disease",
                     "Pre-existing disease means any condition diagnosed or treated in the 48 months before policy start.",
                     ["definition", "health"]),
    ]:
        kg.add_clause(clause)

    # Property clauses
    for clause in [
        PolicyClause("property:1.1", "property_policy", "1.1", "Fire and Allied Perils",
                     "We cover loss or damage from fire, lightning, explosion, and allied perils.",
                     ["coverage", "property", "fire"]),
        PolicyClause("property:2.1", "property_policy", "2.1", "Flood Exclusion (Standard)",
                     "Loss due to flood, inundation, or water damage from external source is excluded.",
                     ["exclusion", "property", "flood"]),
        PolicyClause("property:2.2", "property_policy", "2.2", "Flood Cover Endorsement FLD-001",
                     "If Endorsement FLD-001 is attached, flood damage up to the sum insured is covered.",
                     ["endorsement", "property", "flood"]),
    ]:
        kg.add_clause(clause)

    # Typed edges
    for edge in [
        PolicyEdge("motor:2.1",    "motor:1.1",    "EXCLUDES",   "General exclusions limit own damage claims"),
        PolicyEdge("motor:2.2",    "motor:1.1",    "EXCLUDES",   "DUI exclusion overrides own damage cover"),
        PolicyEdge("motor:2.2",    "motor:1.2",    "EXCLUDES",   "DUI exclusion overrides third party liability"),
        PolicyEdge("motor:1.1",    "motor:3.1",    "REFERENCES", "Own damage coverage relies on accident definition"),
        PolicyEdge("health:2.1",   "health:1.1",   "EXCLUDES",   "Waiting period applies to hospitalisation cover"),
        PolicyEdge("health:3.1",   "health:2.1",   "DEFINES",    "Definition scopes the pre-existing disease exclusion"),
        PolicyEdge("health:2.2",   "health:1.2",   "EXCLUDES",   "Cosmetic exclusion overrides day care coverage"),
        PolicyEdge("property:2.2", "property:2.1", "SUPERSEDES", "FLD-001 endorsement overrides standard flood exclusion"),
        PolicyEdge("property:1.1", "property:2.1", "REFERENCES", "Fire cover section references flood exclusion scope"),
    ]:
        kg.add_edge(edge)

    return kg


# ---------------------------------------------------------------------------
# Graph-aware retrieval
# ---------------------------------------------------------------------------

def graph_aware_retrieve(
    query: str,
    seed_clause_ids: list[str],
    knowledge_graph: PolicyKnowledgeGraph,
    depth: int = 2,
) -> list[PolicyClause]:
    """Expand seed clauses through the knowledge graph.

    Week 6: seed_clause_ids come from Chroma dense retrieval (rag/retriever.py).
    Graph traversal adds related definitions + exclusions to the LLM context window.

    Args:
        query: Original retrieval query (logged for debugging)
        seed_clause_ids: Clause IDs from initial vector retrieval
        knowledge_graph: Built PolicyKnowledgeGraph
        depth: Graph traversal depth

    Returns:
        Seed + graph-expanded PolicyClause list (deduplicated)
    """
    all_clauses: dict[str, PolicyClause] = {}
    for seed_id in seed_clause_ids:
        clause = knowledge_graph.get_clause(seed_id)
        if clause:
            all_clauses[seed_id] = clause
        for related in knowledge_graph.get_related_clauses(seed_id, depth=depth):
            all_clauses[related.clause_id] = related
    return list(all_clauses.values())


# ---------------------------------------------------------------------------
# Week 6 demo — no API key required (pure Python / NetworkX)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 60
    print("=" * W)
    print(" Week 6 — Policy Knowledge Graph Demo")
    print("=" * W)
    print()

    if not NX_AVAILABLE:
        print("ERROR: networkx not installed. Run: pip install networkx")
        raise SystemExit(1)

    kg = build_seed_graph()
    stats = kg.stats()

    print("Graph Statistics")
    print("-" * 40)
    print(f"  Clauses (nodes) : {stats['nodes']}")
    print(f"  Relationships   : {stats['edges']}")
    print(f"  Source policies : {', '.join(sorted(stats['source_docs']))}")
    print(f"  Edge types:")
    for et, count in sorted(stats["edge_types"].items()):
        desc = EDGE_TYPE_DESCRIPTIONS.get(et, "")
        print(f"    {et:<15} {count}x  — {desc}")
    print()

    # Demo 1: exclusion lookup
    print("Demo 1 — What EXCLUDES motor own-damage coverage?")
    print("-" * 40)
    for clause in kg.find_exclusions("motor:1.1"):
        print(f"  [{clause.clause_id}] {clause.title}")
        print(f"   → {clause.text[:90]}...")
    print()

    # Demo 2: 2-hop graph expansion from health hospitalisation
    print("Demo 2 — 2-hop expansion from 'health:1.1' (hospitalisation)")
    print("-" * 40)
    for clause in kg.get_related_clauses("health:1.1", depth=2):
        print(f"  [{clause.clause_id}] {clause.title}  (tags: {', '.join(clause.tags)})")
    print()

    # Demo 3: endorsement supersedes exclusion
    print("Demo 3 — Does the flood endorsement override the flood exclusion?")
    print("-" * 40)
    for clause in kg.find_superseding("property:2.1"):
        print(f"  [{clause.clause_id}] {clause.title}")
        print(f"   SUPERSEDES standard flood exclusion → coverage restored")
    print()

    # Demo 4: graph_aware_retrieve simulation
    print("Demo 4 — graph_aware_retrieve() with 2-hop expansion")
    print("-" * 40)
    query = "Is my car covered if damaged in an accident?"
    seeds = ["motor:1.1"]
    expanded = graph_aware_retrieve(query, seeds, kg, depth=2)
    print(f"  Query        : {query}")
    print(f"  Vector seeds : {seeds}")
    print(f"  After graph expansion ({len(expanded)} clauses total):")
    for c in expanded:
        print(f"    [{c.clause_id}] {c.title}")
    print()

    print("=" * W)
    print(" Week 6 Next Steps")
    print("=" * W)
    print("  1. Replace hand-crafted edges with LLM-extracted relationships")
    print("     from data/policies/*.md using a structured-output prompt")
    print("  2. Wire graph_aware_retrieve() into rag/retriever.py pipeline")
    print("  3. Add GraphRAG context to COVERAGE_CHECK node in claims_workflow.py")
    print("  4. Measure: does GraphRAG improve CitationQualityJudge score?")
    print("=" * W)
