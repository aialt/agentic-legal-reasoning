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

# [中文版说明]⚖️ "Divide and Enhance" 法律智能体框架
本项目是论文**《Divide and Enhance: Agentic Legal Reasoning with Domain-Adapted LLMs for Chinese Law》**中 "Divide" 组件的一个参考代码实现。它建立了一套完整的客户端-服务器架构，包含一个本地化部署的模型服务和一个调用该服务的智能体应用。

## 🚀 架构概览
The project is divided into two main components, which should be run as separate processes:
 1. <kbd>Model_Server</kbd>(模型服务):一个专用的FastAPI服务器，负责将大语言模型加载到GPU显存，并通过API接口提供服务。它的唯一职责是执行文本生成。
 2. <kbd>Main Application</kbd>(主应用): 实现 "Divide" 工作流的核心客户端应用。它处理用户查询，调度 Assess -> Decompose -> Execute -> Synthesize 的四阶段流程，并与 <kbd>model_server</kbd> 通信以进行AI推理。

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



## 🛠️  安装与设置
**Prerequisites**

 - Python 3.9+
 - 已安装CUDA的NVIDIA GPU
 - <kbd>pip</kbd>包管理器
 
***第一步: 克隆仓库***
```javascript
git clone <url>
cd DIVIDE_AGENT_FRAMEWORK
```
***第二步: 安装依赖***
从统一的<kbd>requirements.txt</kbd> 文件中安装整个项目的所有依赖。
```javascript
pip install -r requirements.txt
```
**第三步: 配置项目**
这是最关键的一步。 在运行前，你必须配置好所有路径和密钥。
 1.配置模型服务:
 - 打开  <kbd>model_server/config.py</kbd>。
 - 修改  <kbd>MODEL_PATH</kbd>为你本地大模型文件夹的绝对路径（例如，指向 <kbd>Base_Model</kbd>、<kbd>Fine_Tuned_Model</kbd>目录）。
 - 如果需要，调整 <kbd>HOST</kbd>, <kbd>PORT</kbd>, and <kbd>MAX_GPU_MEMORY</kbd> 。
 
 2.配置智能体应用:

 - 打开 <kbd>configs/settings.py</kbd>。
 - 确保 <kbd>TRIAGE_DECOMPOSE_MODEL_URL</kbd> 和 <kbd>SYNTHESIZE_AGENT_MODEL_URL</kbd> 与你运行的模型服务地址一致。
 - 修改所有的文件路径 (<kbd>LAW_DATA_PATH</kbd>, <kbd>LEGAL_JIEBA_DICT_PATH</kbd>, <kbd>SEMANTIC_EMBEDDING_MODEL_PATH</kbd>) 为你资源文件和模型文件的绝对路径。
 - 填入你的 <kbd>BOCHA_API_KEY</kbd> 以使用网页搜索工具。
## ▶️ 如何运行
你需要在两个独立的终端窗口中启动这两个组件。
**终端 1: 启动模型服务**
```javascript
# Navigate to the model server directory
cd model_server

#Start the server
python server.py
```
请耐心等待，直到日志显示模型已成功加载，服务正在运行。
**终端 2: 运行智能体应用**
```javascript
# From the root directory (DIVIDE_AGENT_FRAMEWORK)
python main.py
```
当智能体应用初始化完RAG系统后，你就可以在控制台输入你的法律问题了。
