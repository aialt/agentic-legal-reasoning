@[TOC](⚖️ Divide and Enhance - Legal Agent Framework)
# [英文版说明]A Code Implementation of the "Divide and Enhance" Paper
This project provides a reference implementation for the "Divide" component of the paper **《Divide and Enhance: Agentic Legal Reasoning with Domain-Adapted LLMs for Chinese Law》**. It establishes a complete client-server architecture, including a local model deployment service and an intelligent agent application that calls this service.

## 🚀 Architecture Overview
The project is divided into two main components, which should be run as separate processes:
 1. <kbd>Model_Server</kbd>: A dedicated FastAPI server responsible for loading the  large language model into GPU memory and exposing it via an API endpoint. Its sole job is to perform text generation.
 2. <kbd>Main Application</kbd>: The core client application that implements the "Divide" workflow. It handles user queries, orchestrates the Triage -> Decompose -> Execute -> Synthesize pipeline, and communicates with the <kbd>model_server</kbd> for AI reasoning.

## 📂 Directory Structure
```javascript
DIVIDE_AGENT_FRAMEWORK/
│
├── configs/
│   └── settings.py           
│
├── divide/                     
│   ├── agent_workflow.py     
│   ├── steps/                  # Implementations for Assess, Decompose, Execute, Synthesize
│   └── tools/                  # Tools available to the LangChain agent (RAG, Web Search)
│
├── Model/                      # Directory for storing model files
│   ├── Base_Model/
│   ├── Fine_Tuned_Model/
│   └── Lawformer/
│
├── model_server/               # FastAPI Model Server
│   ├── config.py               # Configuration for the model server (model path, port)
│   └── server.py               # Main FastAPI server application
│
├── Resources/                  # Data files for the RAG system
│   ├── laws_and_regulations.json
│   └── legal_dictionary.txt
│
├── retrieval/                  # Core implementation of the Hybrid RAG system
│   ├── document_loader.py
│   ├── bm25_retriever.py
│   ├── semantic_retriever.py
│   ├── hybrid_retriever.py
│   └── utils.py
│
├── services/
│   └── llm_interface.py        # Wrapper for all communications with the LLM service
│
├── main.py                     # Entry point for the agent application
├── Readme.md                   # This file
└── requirements.txt            # All project dependencies
```



## 🛠️ Setup and Installation
**Prerequisites**

 - Python 3.9+
 - NVIDIA GPU with CUDA installed
 - <kbd>pip</kbd> package manager

***Step 1: Clone the Repository***
```javascript
git clone <url>
cd DIVIDE_AGENT_FRAMEWORK
```
***Step 2:Install Dependencies***
Install all dependencies for the entire project from the unified <kbd>requirements.txt</kbd> file.
```javascript
pip install -r requirements.txt
```
**Step 3: Configure the Project**
This is the most crucial step. You must configure the paths and keys before running the project.
 1.Configure the Model Server:
 - Open <kbd>model_server/config.py</kbd>.
 - Modify <kbd>MODEL_PATH</kbd> to the absolute path of your fine-tuned language model folder (e.g., pointing to the <kbd>Base_Model</kbd>、<kbd>Fine_Tuned_Model</kbd> directory).
 - Adjust <kbd>HOST</kbd>, <kbd>PORT</kbd>, and <kbd>MAX_GPU_MEMORY</kbd> if needed.

 2.Configure the Agent Application:

 - Open <kbd>configs/settings.py</kbd>.
 - Ensure <kbd>TRIAGE_DECOMPOSE_MODEL_URL</kbd> and <kbd>SYNTHESIZE_AGENT_MODEL_URL</kbd> match the address of your running model server.
 - Modify all file paths (<kbd>LAW_DATA_PATH</kbd>, <kbd>LEGAL_JIEBA_DICT_PATH</kbd>, <kbd>SEMANTIC_EMBEDDING_MODEL_PATH</kbd>) to the absolute paths of your resource and model files.
 - Fill in your <kbd>BOCHA_API_KEY</kbd> for the web search tool.
## ▶️ How to Run
You need to start the two components in two separate terminal windows.
**Terminal 1: Start the Model Server**
```javascript
# Navigate to the model server directory
cd model_server

#Start the server
python server.py
```
Wait until you see the log message indicating that the model has been successfully loaded and the server is running.
**Terminal 2: Run the Agent Application**
```javascript
# From the root directory (DIVIDE_AGENT_FRAMEWORK)
python main.py
```
Once the agent application initializes the RAG system, you can start typing your legal questions in the console.

