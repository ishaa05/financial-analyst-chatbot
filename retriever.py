"""
retriever.py  –  Two-stage retrieval: dense vector search then cross-encoder reranking.

The retriever is the heart of what makes this chatbot feel intelligent rather
than doing keyword search.  Every answer is grounded in the top-8 passages
ranked by *semantic relevance*, not word overlap.

Flow:
  query → embed → cosine ANN search (top-20) → cross-encoder rerank → top-8

The cross-encoder score is computed with a lightweight local model
(cross-encoder/ms-marco-MiniLM-L-6-v2) that sees both the query and the
passage together, giving much better precision than the bi-encoder alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
# from google import genai
# from google.genai import types
from sentence_transformers import SentenceTransformer,CrossEncoder

CHROMA_DIR = Path("embeddings/chroma_db")
COLLECTION_NAME = "infosys_financials"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedder = SentenceTransformer(EMBED_MODEL)

# How many candidates to pull from vector search before reranking
VECTOR_TOPK = 20
# How many to keep after reranking and pass to the LLM
FINAL_TOPK = 8


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_label: str
    content: str
    chunk_type: str
    page: int | None
    section: str
    table_name: str
    vector_score: float      # cosine similarity (0–1, higher is better)
    rerank_score: float      # cross-encoder logit (higher is better)


# ─── Singleton cross-encoder (loaded once per process) ─────────────────────────

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading cross-encoder… (first call only)")
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


# ─── Singleton ChromaDB collection ─────────────────────────────────────────────

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ─── Main retrieve function ────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = FINAL_TOPK,
    filter_doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for *query*.

    Args:
        query: The user question (possibly reformulated to include context).
        top_k: Number of chunks to return after reranking.
        filter_doc_ids: Optional list of doc_id values to restrict search to.

    Returns:
        List of RetrievedChunk sorted by rerank_score descending.
    """
    # 1. Embed the query
    
    query_embedding = _embedder.encode(query, convert_to_numpy=True).tolist()

    # 2. ANN search in ChromaDB
    where_clause = {"doc_id": {"$in": filter_doc_ids}} if filter_doc_ids else None

    results = _get_collection().query(
        query_embeddings=[query_embedding],
        n_results=min(VECTOR_TOPK, _get_collection().count()),
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # ChromaDB returns distance (lower = more similar for cosine); convert
    candidates = []
    for doc, meta, dist in zip(docs, metas, distances):
        vector_score = 1.0 - dist
        candidates.append((doc, meta, vector_score))

    # 3. Cross-encoder reranking
    ce = _get_cross_encoder()
    pairs = [(query, doc) for doc, _, _ in candidates]
    rerank_scores = ce.predict(pairs).tolist()

    # 4. Assemble and sort
    chunks = []
    for (doc, meta, v_score), re_score in zip(candidates, rerank_scores):
        chunks.append(RetrievedChunk(
            chunk_id=meta.get("chunk_id", ""),
            doc_id=meta.get("doc_id", ""),
            doc_label=meta.get("doc_label", ""),
            content=doc,
            chunk_type=meta.get("chunk_type", "text"),
            page=int(meta["page"]) if meta.get("page", -1) != -1 else None,
            section=meta.get("section", ""),
            table_name=meta.get("table_name", ""),
            vector_score=v_score,
            rerank_score=re_score,
        ))

    chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    return chunks[:top_k]


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Serialise retrieved chunks into a context block for the prompt.

    Each chunk is wrapped in <source> tags so the LLM can cite them precisely.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        loc = ""
        if chunk.page:
            loc += f" | page {chunk.page}"
        if chunk.table_name:
            loc += f" | {chunk.table_name}"
        if chunk.section:
            loc += f" | section: {chunk.section}"

        # Truncate content to 1000 chars to stay within token limits
        content = chunk.content[:1000] + ("..." if len(chunk.content) > 1000 else "")

        parts.append(
            f'<source id="{i}" doc="{chunk.doc_label}"{loc}>\n'
            f"{chunk.content}\n"
            f"</source>"
        )
    return "\n\n".join(parts)
