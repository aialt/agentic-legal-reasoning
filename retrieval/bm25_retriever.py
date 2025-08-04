# -*- coding: utf-8 -*-
"""
BM25 Lexical Retriever Module
"""
import os
import jieba
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from configs import settings

# A lexical retriever based on the BM25 algorithm. It uses Jieba for Chinese word segmentation
# and can load a custom legal dictionary to improve tokenization accuracy.
class BM25Retriever:
    def __init__(self, documents_texts: List[str]):
        """
        Initializes the BM25Retriever.

        Args:
            documents_texts (List[str]): A list of text documents to build the index from.
        """
        # Load the custom legal dictionary
        if os.path.exists(settings.LEGAL_JIEBA_DICT_PATH):
            jieba.load_userdict(settings.LEGAL_JIEBA_DICT_PATH)
            print("✅ BM25Retriever: Custom legal dictionary loaded.")
        else:
            print(f"⚠️ BM25Retriever: Custom dictionary not found at: {settings.LEGAL_JIEBA_DICT_PATH}")

        self.texts = documents_texts
        # Tokenize all documents to build the BM25 index
        self.tokenized_corpus = [list(jieba.cut(t)) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_relevant_documents(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieves the top-k most relevant documents for a given query.

        Args:
            query (str): The user's query.
            k (int): The number of documents to return.

        Returns:
            List[str]: A list of the most relevant document texts.
        """
        tokenized_query = list(jieba.cut(query))
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get the indices of the top-k highest scoring documents
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        
        return [self.texts[i] for i in top_k_indices]
