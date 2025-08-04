# -*- coding: utf-8 -*-
"""
Semantic Vector Retriever Module
"""
import os
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from hashlib import md5
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.docstore import InMemoryDocstore
from configs import settings

class SemanticRetriever:
    """
    A vector retriever based on semantic similarity, using HuggingFaceEmbeddings and FAISS.
    - Implements a caching mechanism to avoid re-calculating document embeddings.
    - Automatically detects and uses GPU for FAISS index acceleration.
    """
    def __init__(self, documents: List[Document], cache_key: str):
        """
        Initializes the SemanticRetriever.

        Args:
            documents (List[Document]): A list of LangChain Document objects to build the index from.
            cache_key (str): A unique identifier for creating cache filenames (e.g., a hash based on the law's name).
        """
        self.documents = documents
        
        # --- 1. Initialize the embedding model ---
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.SEMANTIC_EMBEDDING_MODEL_PATH,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
            print(f"✅ SemanticRetriever: Successfully initialized embedding model: {settings.SEMANTIC_EMBEDDING_MODEL_PATH}")
        except Exception as e:
            print(f"❌ SemanticRetriever: Failed to initialize embedding model: {e}")
            raise

        # --- 2. Set up caching path and logic ---
        # Generate a hash based on the current document content to ensure cache consistency
        docs_hash = md5(json.dumps([d.dict() for d in self.documents], sort_keys=True).encode()).hexdigest()
        cache_dir = os.path.join(settings.SEMANTIC_CACHE_PATH, cache_key)
        os.makedirs(cache_dir, exist_ok=True)
        
        embed_file = os.path.join(cache_dir, f"embeddings_{docs_hash}.npy")

        # --- 3. Load or generate embedding vectors ---
        if os.path.exists(embed_file):
            print(f"✅ SemanticRetriever: Loading embeddings from cache: {embed_file}")
            embed_matrix = np.load(embed_file).astype(np.float32)
        else:
            print(f"ℹ️ SemanticRetriever: Cache not found, generating new embeddings...")
            texts_for_embedding = [doc.page_content for doc in self.documents]
            if not texts_for_embedding:
                # Get model dimension to create an empty embedding matrix
                dim = self.embeddings.client.get_sentence_embedding_dimension() or 768
                embed_matrix = np.empty((0, dim), dtype=np.float32)
            else:
                # Batch generate embeddings, using no_grad to reduce memory consumption
                with torch.no_grad():
                    embed_matrix = np.array(self.embeddings.embed_documents(texts_for_embedding)).astype(np.float32)
            np.save(embed_file, embed_matrix)
            print(f"✅ SemanticRetriever: Embeddings generated and cached at: {embed_file}")

        # --- 4. Build the FAISS index ---
        if embed_matrix.shape[0] > 0:
            dim = embed_matrix.shape[1]
            if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    config = faiss.GpuIndexFlatConfig()
                    config.device = torch.cuda.current_device()
                    index = faiss.GpuIndexFlatL2(res, dim, config)
                    print(f"✅ SemanticRetriever: FAISS index built on GPU.")
                except Exception as e:
                    print(f"⚠️ SemanticRetriever: FAISS GPU initialization failed, falling back to CPU: {e}")
                    index = faiss.IndexFlatL2(dim)
            else:
                index = faiss.IndexFlatL2(dim)
                print("ℹ️ SemanticRetriever: FAISS index built on CPU.")
            index.add(embed_matrix)
        else:
            # Create an empty index if there are no documents
            dim = self.embeddings.client.get_sentence_embedding_dimension() or 768
            index = faiss.IndexFlatL2(dim)
            print("⚠️ SemanticRetriever: Document list is empty, created an empty FAISS index.")

        # --- 5. Build the LangChain FAISS wrapper ---
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(self.documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(self.documents))}
        
        self.vectorstore = LangChainFAISS(
            index=index,
            embedding_function=self.embeddings,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieves the most relevant documents and their scores based on vector similarity.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, where each tuple contains a Document object and its similarity score.
        """
        if self.vectorstore.index.ntotal == 0:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k)
