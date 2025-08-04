# -*- coding: utf-8 -*-
"""
Unified configuration module.

All hard-coded paths, API keys, model names, and important parameters are centralized here for management.
Please modify the paths and keys in this file according to your actual environment.
"""

# --- Model Service Addresses ---
# LLM service address for Triage and Decompose steps
TRIAGE_DECOMPOSE_MODEL_URL = "http://localhost:4399/generate"

# LLM service address for the Synthesize step and internal Agent reasoning
SYNTHESIZE_AGENT_MODEL_URL = "http://localhost:8080/generate"


# --- Core File Paths ---
# Data source path for the RAG system's laws and regulations
LAW_DATA_PATH = "/path/to/your/Resources/laws_and_regulations.json"

# Custom legal dictionary path for the BM25 retriever
LEGAL_JIEBA_DICT_PATH = "/path/to/your/Resources/legal_dictionary.txt"

# Embedding model path for the semantic retriever
SEMANTIC_EMBEDDING_MODEL_PATH = "/path/to/your/Rag/Model/Lawformer"

# Cache file storage path for semantic retrieval
SEMANTIC_CACHE_PATH = "/path/to/your/.semantic_cache"


# --- API Keys ---
# API Key for the web search tool
BOCHA_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# --- RAG System Parameters ---
# Chunk documents when their content length (in characters) exceeds this value
DOCUMENT_CHUNK_SIZE = 500
DOCUMENT_CHUNK_OVERLAP = 50

# TF-IDF threshold for calculating query specificity to dynamically adjust RRF weights
# The original paper uses a value of 0.05
QUERY_SPECIFICITY_THRESHOLD = 0.05
