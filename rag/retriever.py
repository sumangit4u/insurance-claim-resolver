"""RAG retriever abstraction layer.

Local (Week 1-2):  Chroma + GoogleGenerativeAIEmbeddings
Production (Week 2+): Vertex AI Search (behind settings.gcp_ready flag)

Pattern: directly extends rag_utils.py from session10_Agentic_RAG.
- Same two-level chunking (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter)
- Same two Chroma collections: detail (chunk_size=800) and summary
- Same grader schemas: RetrievalGrade, HallucinationGrade, AnswerGrade
- Adds domain-specific: PolicyCitation schema for clause-level grounding

Usage (Week 1 demo):
    python -m rag.retriever
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, Field

from config.settings import get_settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
CORPUS_DIR = DATA_DIR / "policies"
CHROMA_DIR = DATA_DIR / "chroma_db"

CORPUS_FILES: dict[str, Path] = {
    "motor_policy": CORPUS_DIR / "motor_insurance_policy.md",
    "health_policy": CORPUS_DIR / "health_insurance_policy.md",
    "property_policy": CORPUS_DIR / "property_insurance_policy.md",
}

DETAIL_COLLECTION = "insurance_policy_detail"
SUMMARY_COLLECTION = "insurance_policy_summary"

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# ---------------------------------------------------------------------------
# Pydantic schemas (extended from rag_utils.py grader schemas)
# ---------------------------------------------------------------------------


class PolicyCitation(BaseModel):
    """A cited policy clause — core to the 'every decision cited' requirement."""
    section: str = Field(description="Policy section number, e.g. '3.2.1'")
    clause_text: str = Field(description="Verbatim clause text from the policy document")
    source_doc: str = Field(description="Source document name, e.g. 'motor_policy'")
    relevance: Literal["directly_answers", "supports", "exclusion"] = Field(
        description="How this clause relates to the claim decision"
    )


class RetrievalGrade(BaseModel):
    """Per-document relevance grade — same schema as rag_utils.RetrievalGrade."""
    grade: Literal["correct", "ambiguous", "incorrect"] = Field(
        description=(
            "'correct' = document directly answers the question; "
            "'ambiguous' = on-topic but does not fully answer; "
            "'incorrect' = off-topic."
        )
    )
    reason: str = Field(description="One short sentence justifying the grade.")


class HallucinationGrade(BaseModel):
    """Is the generation grounded in retrieved documents?"""
    grounded: bool = Field(
        description="True if every factual claim is supported by the context."
    )
    reason: str = Field(description="One short sentence justifying the verdict.")


class AnswerGrade(BaseModel):
    """Does the generation actually address the user's question?"""
    addresses_question: bool = Field(
        description="True if the answer responds to the question asked."
    )
    reason: str = Field(description="One short sentence justifying the verdict.")


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _chunk_markdown(text: str, source_name: str) -> list[Document]:
    """Split a markdown policy document into chunks with header metadata.

    Two-level split (same as rag_utils.py):
    1. MarkdownHeaderTextSplitter → preserves section hierarchy
    2. RecursiveCharacterTextSplitter (chunk_size=800) → fits context window
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    header_chunks = header_splitter.split_text(text)
    final_chunks = char_splitter.split_documents(header_chunks)
    for doc in final_chunks:
        doc.metadata["source"] = source_name
    return final_chunks


def build_chroma_from_policies(persist: bool = True) -> Chroma:
    """Ingest all policy markdown files into Chroma.

    Run once (or when policies are updated):
        python -m rag.retriever  # Week 1 demo runs this

    Args:
        persist: If True, write to CHROMA_DIR; otherwise in-memory.

    Returns:
        Populated Chroma vector store
    """
    all_chunks: list[Document] = []
    missing = []

    for name, path in CORPUS_FILES.items():
        if path.exists():
            text = path.read_text(encoding="utf-8")
            chunks = _chunk_markdown(text, name)
            all_chunks.extend(chunks)
            print(f"  [+] {name}: {len(chunks)} chunks")
        else:
            missing.append(str(path))
            print(f"  [!] {name}: file not found — skipping")

    if not all_chunks:
        raise FileNotFoundError(
            f"No policy files found. Expected in {CORPUS_DIR}. Missing: {missing}"
        )

    kwargs: dict = {
        "collection_name": DETAIL_COLLECTION,
        "embedding_function": get_embeddings(),
    }
    if persist:
        kwargs["persist_directory"] = str(CHROMA_DIR)

    store = Chroma.from_documents(all_chunks, **kwargs)
    print(f"  [✓] Chroma built: {len(all_chunks)} total chunks → {CHROMA_DIR}")
    return store


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return embeddings — same model as rag_utils.EMBEDDING_MODEL."""
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


def get_policy_retriever(
    k: int = 4,
    policy_filter: Optional[str] = None,
    use_summary: bool = False,
) -> Chroma:
    """Return a retriever over the policy Chroma collection.

    Args:
        k: Number of chunks to retrieve
        policy_filter: Optional source filter ('motor_policy', 'health_policy', etc.)
        use_summary: If True, query the summary collection first (HRAG pattern)

    Returns:
        Chroma retriever instance
    """
    settings = get_settings()

    if settings.gcp_ready:
        # Week 2+: swap to Vertex AI Search
        raise NotImplementedError(
            "Vertex AI Search retriever not yet implemented. "
            "Set GCP_PROJECT_ID and VERTEX_SEARCH_DATASTORE_ID, then implement in Week 2."
        )

    collection = SUMMARY_COLLECTION if use_summary else DETAIL_COLLECTION
    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    search_kwargs: dict = {"k": k}
    if policy_filter:
        search_kwargs["filter"] = {"source": policy_filter}

    return store.as_retriever(search_kwargs=search_kwargs)


def format_docs_with_citations(docs: list[Document]) -> tuple[str, list[str]]:
    """Format retrieved docs and extract citation strings.

    Returns:
        (formatted_context, list_of_citation_strings)
    """
    formatted = []
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        h1 = doc.metadata.get("h1", "")
        h2 = doc.metadata.get("h2", "")
        section = f"{h1} > {h2}".strip(" >")
        citation = f"[{source}: {section}]" if section else f"[{source}]"
        formatted.append(f"{citation}\n{doc.page_content}")
        citations.append(citation)
    return "\n\n---\n\n".join(formatted), citations


# ---------------------------------------------------------------------------
# Week 1 demo — requires GOOGLE_API_KEY in .env
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    W = 60
    print("=" * W)
    print(" Week 1 — RAG Retriever Demo")
    print("=" * W)
    print()

    api_key = os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set in environment.")
        print("         Set it in .env and run:  python -m rag.retriever")
        print()

    # Step 1: show corpus inventory
    print("Step 1 — Policy corpus inventory")
    print("-" * 40)
    for name, path in CORPUS_FILES.items():
        status = "✓ found" if path.exists() else "✗ missing"
        print(f"  {status}  {name}: {path.name}")
    print()

    # Step 2: build Chroma (only if GOOGLE_API_KEY present)
    if api_key:
        print("Step 2 — Ingesting policies into Chroma")
        print("-" * 40)
        try:
            store = build_chroma_from_policies(persist=True)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            store = None
        print()

        # Step 3: run 3 test queries
        if store is not None:
            print("Step 3 — Test queries")
            print("-" * 40)
            queries = [
                ("motor",    "Is flood damage covered under motor insurance?"),
                ("health",   "What is the waiting period for pre-existing diseases?"),
                (None,       "What documents are required to file a property claim?"),
            ]
            retriever = get_policy_retriever(k=3)
            for policy_filter, query in queries:
                print(f"  Query : {query}")
                docs = retriever.get_relevant_documents(query)
                context, citations = format_docs_with_citations(docs)
                print(f"  Found : {len(docs)} chunks")
                for c in citations:
                    print(f"    → {c}")
                print()
    else:
        print("Step 2 & 3 — Skipped (no GOOGLE_API_KEY)")
        print()
        print("Chunking schema (no API key needed):")
        print("-" * 40)
        print("  Level 1: MarkdownHeaderTextSplitter → preserves section hierarchy")
        print("  Level 2: RecursiveCharacterTextSplitter(chunk_size=800) → context window")
        print("  Metadata added: source (policy name), h1, h2, h3")
        print("  Collection: insurance_policy_detail (k=4 by default)")
        print()
        print("  PolicyCitation schema:")
        for field_name, field_info in PolicyCitation.model_fields.items():
            print(f"    {field_name:<20} {field_info.description}")
        print()

    print("=" * W)
    print(" Week 1 Complete → Week 2: RAGAS baseline evaluation")
    print("=" * W)
