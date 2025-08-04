# -*- coding: utf-8 -*-
"""
Legal Document Loading and Preprocessing Module
"""
import os
import json
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configs import settings

def load_and_prepare_documents() -> Tuple[List[Document], Dict[str, Document]]:
    """
    Loads legal data from a JSON file, performs conditional chunking, and creates a map of original documents.

    Returns:
        Tuple[List[Document], Dict[str, Document]]:
        - docs_for_retrieval: A list of documents/chunks to be used for building the retriever.
        - original_docs_map: A mapping from source_id to the complete, original document.
    """
    if not os.path.exists(settings.LAW_DATA_PATH):
        raise FileNotFoundError(f"Error: Legal data file not found at: {settings.LAW_DATA_PATH}")

    with open(settings.LAW_DATA_PATH, "r", encoding="utf-8") as f:
        law_data = json.load(f)

    original_docs_map: Dict[str, Document] = {}
    docs_for_splitting: List[Document] = []

    # 1. Create the original document map and the list of documents to be split
    for i, item in enumerate(law_data):
        if isinstance(item.get("content"), str) and item["content"].strip():
            # Use the item's 'id' as the source_id; if it doesn't exist, use the index as a unique identifier
            source_id = item.get("id", f"doc_{i}")
            
            doc = Document(
                page_content=item["content"],
                metadata={
                    "number_items": item.get("number_items", []),
                    "regulatory_name": item.get("regulatory_name", ""),
                    "type": item.get("type", ""),
                    "source_id": source_id
                }
            )
            original_docs_map[source_id] = doc
            docs_for_splitting.append(doc)

    if not docs_for_splitting:
        print("Warning: The legal data file is empty or contains no valid content.")
        return [], {}

    # 2. Conditionally chunk long documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.DOCUMENT_CHUNK_SIZE,
        chunk_overlap=settings.DOCUMENT_CHUNK_OVERLAP
    )
    
    docs_for_retrieval: List[Document] = []
    print(f"INFO: Checking {len(docs_for_splitting)} documents for chunking...")
    for doc in docs_for_splitting:
        if len(doc.page_content) > settings.DOCUMENT_CHUNK_SIZE:
            # If the document content exceeds the threshold, split it and inherit the parent's metadata
            chunks = text_splitter.split_documents([doc])
            docs_for_retrieval.extend(chunks)
        else:
            # If it does not exceed the threshold, use the original document directly
            docs_for_retrieval.append(doc)
    
    print(f"INFO: A total of {len(docs_for_retrieval)} documents/chunks have been generated for the retriever.")
    
    return docs_for_retrieval, original_docs_map
