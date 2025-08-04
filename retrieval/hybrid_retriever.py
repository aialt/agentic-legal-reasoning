# -*- coding: utf-8 -*-
"""
Hybrid Retriever Module

This module implements one of the core components of the "Divide and Enhance" paper:
the dynamically-weighted hybrid retrieval system. It combines the strengths of
lexical search (BM25) and semantic search (vector-based).
"""
import jieba
import numpy as np
from typing import List, Tuple, Dict
from hashlib import md5
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.schema import Document

from .bm25_retriever import BM25Retriever
from .semantic_retriever import SemanticRetriever
from configs import settings

class HybridRetriever:
    """
    Implements the "Dynamically-weighted Hybrid RAG" architecture from the paper.
    1. Executes BM25 and Semantic retrieval in parallel.
    2. Dynamically calculates weights based on query specificity.
    3. Merges the results using the weighted Reciprocal Rank Fusion (RRF) algorithm.
    """
    def __init__(self, documents_for_retrieval: List[Document], semantic_retriever: SemanticRetriever):
        """
        Initializes the HybridRetriever.

        Args:
            documents_for_retrieval (List[Document]): The list of documents/chunks for retrieval.
            semantic_retriever (SemanticRetriever): An already initialized instance of the semantic retriever.
        """
        self.all_docs_for_retrieval = documents_for_retrieval
        self.doc_content_map = {doc.page_content: doc for doc in documents_for_retrieval}
        
        texts = [doc.page_content for doc in documents_for_retrieval]
        
        # 1. Initialize the BM25 retriever
        self.bm25_retriever = BM25Retriever(texts)
        
        # 2. Receive the initialized semantic retriever
        self.semantic_retriever = semantic_retriever
        
        # 3. Initialize the TF-IDF vectorizer for calculating query specificity
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda text: list(jieba.cut(text)), lowercase=False)
        try:
            if texts:
                self.tfidf_vectorizer.fit(texts)
                self.tfidf_fitted = True
            else:
                self.tfidf_fitted = False
        except ValueError:
            self.tfidf_fitted = False
            print("⚠️ HybridRetriever: TF-IDF fit failed, possibly because the document content is empty.")

    def _calculate_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """
        Calculates dynamic weights for BM25 and semantic retrieval based on query specificity.
        Corresponds to the "Dynamically-weighted Fusion" section of the paper (Section 3.3).
        """
        if not self.tfidf_fitted:
            return (0.5, 0.5) # Return default weights if TF-IDF is not initialized

        # Calculate the TF-IDF vector for the query
        query_tokens = " ".join(list(jieba.cut(query)))
        if not query_tokens: return (0.5, 0.5)
        
        query_vec = self.tfidf_vectorizer.transform([query_tokens]).toarray()[0]
        
        if np.sum(query_vec) == 0:
            return (0.3, 0.7) # If query terms are not in the vocabulary, lean towards semantic search

        # The paper's S(q) is the average tf-idf score; here we use the ratio of non-zero elements as a simplified implementation
        specificity = np.count_nonzero(query_vec) / len(self.tfidf_vectorizer.vocabulary_)

        # Determine weights based on the threshold
        if specificity > settings.QUERY_SPECIFICITY_THRESHOLD:
            # For high-specificity queries (e.g., containing precise legal terms), lexical search is more important
            return (0.7, 0.3)
        else:
            # For low-specificity/general queries, semantic search is more important
            return (0.3, 0.7)

    def _reciprocal_rank_fusion(self, rankings: List[List[Document]], weights: List[float], k: int = 60) -> List[Tuple[Document, float]]:
        """
        Implements the weighted Reciprocal Rank Fusion (RRF) algorithm.
        Corresponds to formula (2) in the paper.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Iterate through all ranked lists
        for rank_list, weight in zip(rankings, weights):
            for rank, doc in enumerate(rank_list):
                # Use a hash of the document content as a unique ID
                doc_id = md5(doc.page_content.encode()).hexdigest()
                doc_map[doc_id] = doc
                
                # Accumulate the RRF score
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += weight * (1 / (rank + k))
        
        # Sort by score in descending order
        sorted_doc_ids = sorted(scores.keys(), key=lambda id: scores[id], reverse=True)
        
        return [(doc_map[doc_id], scores[doc_id]) for doc_id in sorted_doc_ids]

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Executes the complete hybrid retrieval process.
        """
        # 1. Get dynamic weights
        w_bm25, w_semantic = self._calculate_dynamic_weights(query)
        print(f"ℹ️ Dynamic Weights: BM25={w_bm25}, Semantic={w_semantic}")

        # 2. Execute both retrieval methods in parallel, fetching more candidates than top_k for fusion
        candidate_k = top_k * 5
        bm25_texts = self.bm25_retriever.get_relevant_documents(query, k=candidate_k)
        bm25_docs = [self.doc_content_map[text] for text in bm25_texts if text in self.doc_content_map]
        
        semantic_docs_with_scores = self.semantic_retriever.get_relevant_documents(query, k=candidate_k)
        semantic_docs = [doc for doc, score in semantic_docs_with_scores]

        # 3. Fuse the results using RRF
        fused_results = self._reciprocal_rank_fusion(
            rankings=[bm25_docs, semantic_docs],
            weights=[w_bm25, w_semantic]
        )
        
        return fused_results[:top_k]
