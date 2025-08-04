# -*- coding: utf-8 -*-
"""
Legal RAG Tool Interface Layer

This is the tool that is directly called by the LangChain Agent. It encapsulates the entire complex RAG retrieval process.
"""
from typing import List, Tuple
from hashlib import md5

from langchain.schema import Document

from retrieval.document_loader import load_and_prepare_documents
from retrieval.semantic_retriever import SemanticRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.utils import extract_and_match_law_name

# --- Global loading to avoid reloading and re-indexing on every call ---
print("--- Initializing RAG system, please wait... ---")
# 1. Load and chunk all documents
ALL_DOCS_FOR_RETRIEVAL, ORIGINAL_DOCS_MAP = load_and_prepare_documents()

# 2. Build a mapping of documents by regulation name for subsequent pruning
REG_NAME_TO_DOCS = {}
for doc in ALL_DOCS_FOR_RETRIEVAL:
    reg_name = doc.metadata.get("regulatory_name")
    if reg_name:
        REG_NAME_TO_DOCS.setdefault(reg_name, []).append(doc)
ALL_LAW_NAMES = list(REG_NAME_TO_DOCS.keys())

# 3. Pre-build a retriever instance for the "all laws" scope
print("Building retriever instance for 'all_laws' scope...")
SEMANTIC_RETRIEVER_ALL = SemanticRetriever(ALL_DOCS_FOR_RETRIEVAL, cache_key="all_laws")
HYBRID_RETRIEVER_ALL = HybridRetriever(ALL_DOCS_FOR_RETRIEVAL, SEMANTIC_RETRIEVER_ALL)
print("--- RAG system initialization complete ---")

# --- Cache for already built retrievers for specific laws ---
RETRIEVER_CACHE = {
    "all_laws": HYBRID_RETRIEVER_ALL
}


def _format_docs_for_agent(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """Helper function to format retrieval results into a string readable by the Agent."""
    if not docs_with_scores:
        return "No relevant legal articles found in the knowledge base."
    
    output_lines = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        output_lines.append(
            f"[{i}] Regulation: {doc.metadata.get('regulatory_name', 'N/A')}\n"
            f"Article: {doc.metadata.get('number_items', 'N/A')}\n"
            f"Relevance Score: {score:.4f}\n"
            f"Content Snippet: {doc.page_content[:200]}..."
        )
    return "\n---\n".join(output_lines)

def rag_law_search(query: str, top_k: int = 3) -> str:
    """
    【Legal Knowledge Base Search Tool】
    The main function called by the Agent. It retrieves relevant legal articles from the local knowledge base based on the user's query.
    """
    print(f"\n[RAG Tool] Received query: {query}")
    
    # --- 1. Query Pre-processing and Pruning ---
    selected_law, cleaned_query = extract_and_match_law_name(query, ALL_LAW_NAMES)
    
    search_query = cleaned_query if cleaned_query else query
    
    # --- 2. Select or build the appropriate retriever ---
    if selected_law:
        print(f"[RAG Tool] Identified legal scope: {selected_law}")
        cache_key = md5(selected_law.encode()).hexdigest()
        
        if cache_key not in RETRIEVER_CACHE:
            print(f"[RAG Tool] Building a new retriever instance for '{selected_law}'...")
            scoped_docs = REG_NAME_TO_DOCS[selected_law]
            semantic_retriever_scoped = SemanticRetriever(scoped_docs, cache_key=cache_key)
            hybrid_retriever_scoped = HybridRetriever(scoped_docs, semantic_retriever_scoped)
            RETRIEVER_CACHE[cache_key] = hybrid_retriever_scoped
        
        active_retriever = RETRIEVER_CACHE[cache_key]
    else:
        print("[RAG Tool] No specific law identified, searching within all laws.")
        active_retriever = RETRIEVER_CACHE["all_laws"]

    # --- 3. Execute retrieval ---
    retrieved_chunks = active_retriever.retrieve(search_query, top_k=top_k)

    # --- 4. Post-processing: Map retrieved chunks back to their full parent documents ---
    final_results = []
    seen_parent_ids = set()
    for chunk, score in retrieved_chunks:
        parent_id = chunk.metadata.get("source_id")
        if parent_id and parent_id not in seen_parent_ids:
            parent_doc = ORIGINAL_DOCS_MAP.get(parent_id)
            if parent_doc:
                final_results.append((parent_doc, score))
                seen_parent_ids.add(parent_id)
    
    print(f"[RAG Tool] Retrieved {len(final_results)} unique legal articles.")

    # --- 5. Format the output ---
    return _format_docs_for_agent(final_results)
